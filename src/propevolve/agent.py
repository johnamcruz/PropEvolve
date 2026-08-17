"""Recurrent distributional Double-DQN for exact masked action scoring."""

from __future__ import annotations

import copy
from contextlib import nullcontext
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .balance_aware_regime_selectivity import (
    BalanceAwareRegimeSelectivity,
    EXPANSION_REGIME_CONFLUENCE_SEMANTICS,
    PERSISTENT_CHOP_ASSOCIATION_SEMANTICS,
    PERSISTENT_CHOP_NEGATIVE_WEIGHT_SEMANTICS,
    STATIC_STATE_SEMANTICS,
    REGIME_TEACHER_CHANNELS,
    SIDE_CONDITIONED_EXPANSION_REGIME_CONFLUENCE_SEMANTICS,
)
from .decision import Action
from .config import configure_runtime_environment
from .replay import Transition


_REGIME_SELECTIVITY_STRATA = (
    "positive_long_short",
    "positive_long",
    "positive_short",
    "dominant_chop",
    "nonchop",
    "low_headroom_le_0_25",
    "mid_headroom_gt_0_25_lt_0_75",
    "safe_headroom_ge_0_75",
)
_REGIME_ACTION_NAMES = ("wait", "long", "short")
_REGIME_CONFUSION_FIELDS = tuple(
    f"target_{target}_predicted_{prediction}_rows"
    for target in _REGIME_ACTION_NAMES
    for prediction in _REGIME_ACTION_NAMES
)
_REGIME_SELECTIVITY_ADDITIVE_FIELDS = (
    "rows",
    "target_wait_probability_sum",
    "model_wait_probability_sum",
    "wait_absolute_error_sum",
    "target_action_probability_sum",
    "model_target_action_probability_sum",
    "target_action_absolute_error_sum",
    "greedy_wait_rows",
    "declared_side_probability_sum",
    "greedy_entry_rows",
    "correct_rows",
    *_REGIME_CONFUSION_FIELDS,
)
_REGIME_CHANNEL_ADDITIVE_FIELDS = (
    "rows",
    "target_probability_sum",
    "model_probability_sum",
    "absolute_error_sum",
    "squared_error_sum",
)
_ENTRY_BALANCE_ACTION_NAMES = ("wait", "long", "short")
_ENTRY_ACTION_LOSS_REDUCTIONS = {
    "population_weighted_mean_v1",
    "equal_present_class_mean_v1",
}
_ENTRY_BALANCE_ADDITIVE_FIELDS = (
    "rows",
    "weighted_mass",
    "unweighted_ce_sum",
    "weighted_ce_sum",
)
_REGIME_ENTRY_CONFLICT_FIELDS = (
    "rows",
    "target_wait_probability_sum",
    "target_declared_side_probability_sum",
    "model_wait_probability_sum",
    "soft_wait_disagreement_rows",
)
_REGIME_PERSISTENT_ADDITIVE_METRICS = (
    "regime_selectivity_exact_wait_rows",
    "regime_selectivity_exact_wait_weight_sum",
    "regime_selectivity_exact_wait_model_wait_probability_sum",
    "regime_selectivity_persistent_chop_weight_sum",
    "regime_selectivity_persistent_dead_chop_rows",
    "regime_selectivity_persistent_dead_chop_weight_sum",
    "regime_selectivity_persistent_dead_chop_model_wait_probability_sum",
    "regime_selectivity_transition_ready_rows",
    "regime_selectivity_transition_ready_weight_sum",
    "regime_selectivity_transition_ready_model_wait_probability_sum",
    "regime_selectivity_failed_setup_confluence_rows",
    "regime_selectivity_failed_setup_confluence_weight_sum",
    "regime_selectivity_failed_setup_confluence_model_wait_probability_sum",
    "regime_selectivity_failed_long_confluence_rows",
    "regime_selectivity_failed_long_confluence_model_wait_probability_sum",
    "regime_selectivity_failed_short_confluence_rows",
    "regime_selectivity_failed_short_confluence_model_wait_probability_sum",
    "regime_selectivity_transition_positive_long_rows",
    "regime_selectivity_transition_positive_long_declared_side_probability_sum",
    "regime_selectivity_transition_positive_short_rows",
    "regime_selectivity_transition_positive_short_declared_side_probability_sum",
    "regime_selectivity_association_dead_wait_rows",
    "regime_selectivity_association_dead_wait_model_wait_probability_sum",
    "regime_selectivity_association_transition_positive_long_rows",
    "regime_selectivity_association_transition_positive_long_model_wait_probability_sum",
    "regime_selectivity_association_transition_positive_short_rows",
    "regime_selectivity_association_transition_positive_short_model_wait_probability_sum",
)


def resolve_device(device: str) -> torch.device:
    """Resolve the declared accelerator without silently changing explicit requests."""
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS training was requested but this PyTorch runtime cannot use MPS"
        )
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA training was requested but this PyTorch runtime cannot use CUDA"
        )
    if device not in {"cpu", "mps", "cuda"}:
        raise ValueError("device must be auto, cuda, mps, or cpu")
    return torch.device(device)


def centered_entry_search_target(
    probabilities: torch.Tensor,
    *,
    center: float,
    probability_epsilon: float,
    teacher_temperature: float,
) -> torch.Tensor:
    """Center a soft opportunity score on its authenticated fit-only base rate."""
    if (
        not 0 < probability_epsilon < 0.5
        or not probability_epsilon < center < 1.0 - probability_epsilon
        or teacher_temperature <= 0
    ):
        raise ValueError("entry-search probability contract is invalid")
    bounded = probabilities.clamp(probability_epsilon, 1.0 - probability_epsilon)
    centered_log_odds = (
        torch.logit(bounded) - math.log(center / (1.0 - center))
    ) / teacher_temperature
    return torch.sigmoid(centered_log_odds)


def exact_action_margin_losses(
    flat_action_values: torch.Tensor,
    targets: torch.Tensor,
    *,
    margin: float,
) -> torch.Tensor:
    """Require each exact entry action to outrank both alternatives."""
    if (
        flat_action_values.ndim != 2
        or flat_action_values.shape[-1] != 3
        or not torch.is_floating_point(flat_action_values)
        or targets.shape != flat_action_values.shape[:-1]
        or targets.dtype != torch.long
        or isinstance(margin, bool)
        or not math.isfinite(float(margin))
        or float(margin) < 0.0
    ):
        raise ValueError("exact action margin loss contract is invalid")
    if float(margin) == 0.0:
        return torch.zeros(
            flat_action_values.shape[0],
            dtype=flat_action_values.dtype,
            device=flat_action_values.device,
        )
    demonstrated = flat_action_values.gather(1, targets[:, None]).squeeze(1)
    alternative_offsets = torch.full_like(flat_action_values, float(margin))
    alternative_offsets.scatter_(1, targets[:, None], 0.0)
    strongest_margin_action = (
        flat_action_values + alternative_offsets
    ).max(dim=1).values
    return strongest_margin_action - demonstrated


def chop_specific_wait_margin_losses(
    flat_action_values: torch.Tensor,
    *,
    dominant_chop_membership: torch.Tensor,
    failed_long_membership: torch.Tensor,
    failed_short_membership: torch.Tensor,
    chop_margin: float,
    failed_confluence_margin: float,
) -> torch.Tensor:
    """Require WAIT to outrank entry actions only on declared chop failures."""
    memberships = (
        dominant_chop_membership,
        failed_long_membership,
        failed_short_membership,
    )
    margins = (chop_margin, failed_confluence_margin)
    if (
        flat_action_values.ndim != 2
        or flat_action_values.shape[-1] != 3
        or not torch.is_floating_point(flat_action_values)
        or any(value.shape != flat_action_values.shape[:-1] for value in memberships)
        or any(
            isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for value in margins
        )
    ):
        raise ValueError("chop-specific WAIT margin loss contract is invalid")
    wait = flat_action_values[:, int(Action.WAIT)]
    strongest_entry = flat_action_values[:, 1:].max(dim=1).values
    dominant_loss = nn.functional.relu(
        strongest_entry + float(chop_margin) - wait
    ) * dominant_chop_membership * float(chop_margin > 0.0)
    long_loss = nn.functional.relu(
        flat_action_values[:, int(Action.ENTER_LONG_1)]
        + float(failed_confluence_margin)
        - wait
    ) * failed_long_membership * float(failed_confluence_margin > 0.0)
    short_loss = nn.functional.relu(
        flat_action_values[:, int(Action.ENTER_SHORT_1)]
        + float(failed_confluence_margin)
        - wait
    ) * failed_short_membership * float(failed_confluence_margin > 0.0)
    return dominant_loss + long_loss + short_loss


def persistent_chop_association_rank_loss(
    flat_action_values: torch.Tensor,
    *,
    dead_membership: torch.Tensor,
    transition_positive_long_membership: torch.Tensor,
    transition_positive_short_membership: torch.Tensor,
    q_temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rank dead-chop WAIT above exact transition-positive WAIT preference.

    Long and Short values form a detached reference, so this auxiliary can
    only move the WAIT coordinate. Long and Short cohorts are normalized
    separately before their equal-present-side mean.
    """
    memberships = (
        dead_membership,
        transition_positive_long_membership,
        transition_positive_short_membership,
    )
    if (
        flat_action_values.ndim != 2
        or flat_action_values.shape[-1] != 3
        or not torch.is_floating_point(flat_action_values)
        or any(value.shape != flat_action_values.shape[:-1] for value in memberships)
        or isinstance(q_temperature, bool)
        or not math.isfinite(float(q_temperature))
        or float(q_temperature) <= 0.0
    ):
        raise ValueError("persistent-chop association loss contract is invalid")
    dead_mass = dead_membership.sum()
    ready_long_mass = transition_positive_long_membership.sum()
    ready_short_mass = transition_positive_short_membership.sum()
    dead_active = (dead_mass > 0).to(flat_action_values.dtype)
    ready_long_active = (ready_long_mass > 0).to(flat_action_values.dtype)
    ready_short_active = (ready_short_mass > 0).to(flat_action_values.dtype)
    ready_side_count = ready_long_active + ready_short_active
    active = dead_active * (ready_side_count > 0).to(flat_action_values.dtype)
    temperature = float(q_temperature)
    wait_preference = flat_action_values[:, int(Action.WAIT)] - (
        torch.logsumexp(
            flat_action_values[:, 1:].detach() / temperature,
            dim=-1,
        )
        * temperature
    )
    tiny = torch.finfo(flat_action_values.dtype).tiny
    dead_wait_preference = (
        wait_preference * dead_membership
    ).sum() / dead_mass.clamp_min(tiny)
    ready_long_wait_preference = (
        wait_preference * transition_positive_long_membership
    ).sum() / ready_long_mass.clamp_min(tiny)
    ready_short_wait_preference = (
        wait_preference * transition_positive_short_membership
    ).sum() / ready_short_mass.clamp_min(tiny)
    ready_wait_preference = (
        ready_long_wait_preference * ready_long_active
        + ready_short_wait_preference * ready_short_active
    ) / ready_side_count.clamp_min(1.0)
    return (
        nn.functional.softplus(
            ready_wait_preference - dead_wait_preference
        ) * active,
        active,
    )


def side_conditioned_wait_rank_loss(
    flat_action_values: torch.Tensor,
    *,
    failed_long_membership: torch.Tensor,
    failed_short_membership: torch.Tensor,
    valid_long_membership: torch.Tensor,
    valid_short_membership: torch.Tensor,
    q_temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Separate failed-versus-valid WAIT preference for Long and Short."""
    memberships = (
        failed_long_membership,
        failed_short_membership,
        valid_long_membership,
        valid_short_membership,
    )
    if (
        flat_action_values.ndim != 2
        or flat_action_values.shape[-1] != 3
        or not torch.is_floating_point(flat_action_values)
        or any(value.shape != flat_action_values.shape[:-1] for value in memberships)
        or isinstance(q_temperature, bool)
        or not math.isfinite(float(q_temperature))
        or float(q_temperature) <= 0.0
    ):
        raise ValueError("side-conditioned WAIT rank loss contract is invalid")
    tiny = torch.finfo(flat_action_values.dtype).tiny
    loss_sum = torch.zeros(
        (), dtype=flat_action_values.dtype, device=flat_action_values.device
    )
    active_sides = torch.zeros_like(loss_sum)
    for side_index, failed, valid in (
        (int(Action.ENTER_LONG_1), failed_long_membership, valid_long_membership),
        (int(Action.ENTER_SHORT_1), failed_short_membership, valid_short_membership),
    ):
        failed_mass = failed.sum()
        valid_mass = valid.sum()
        active = ((failed_mass > 0) & (valid_mass > 0)).to(
            flat_action_values.dtype
        )
        wait_preference = (
            flat_action_values[:, int(Action.WAIT)]
            - flat_action_values[:, side_index]
        ) / float(q_temperature)
        failed_preference = (
            wait_preference * failed
        ).sum() / failed_mass.clamp_min(tiny)
        valid_preference = (
            wait_preference * valid
        ).sum() / valid_mass.clamp_min(tiny)
        loss_sum = loss_sum + nn.functional.softplus(
            valid_preference - failed_preference
        ) * active
        active_sides = active_sides + active
    return loss_sum, active_sides


class RecurrentC51Network(nn.Module):
    """Compact market/account encoder with recurrent C51 action values."""

    def __init__(
        self,
        observation_dim: int,
        action_count: int,
        atoms: int,
        hidden_dim: int,
        teacher_channels: int = 0,
    ) -> None:
        super().__init__()
        self.observation_dim = int(observation_dim)
        self.action_count = int(action_count)
        self.atoms = int(atoms)
        self.hidden_dim = int(hidden_dim)
        self.teacher_channels = int(teacher_channels)
        self.input = nn.Sequential(
            nn.LayerNorm(observation_dim),
            nn.Linear(observation_dim, hidden_dim),
            nn.SiLU(),
        )
        self.recurrent = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, action_count * atoms)
        self.teacher_output = (
            nn.Linear(hidden_dim, self.teacher_channels)
            if self.teacher_channels
            else None
        )

    def recurrent_features(
        self,
        observations: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.input(observations)
        recurrent_dtype = self.recurrent.weight_ih_l0.dtype
        if encoded.dtype != recurrent_dtype:
            encoded = encoded.to(recurrent_dtype)
        if hidden is not None and hidden.dtype != recurrent_dtype:
            hidden = hidden.to(recurrent_dtype)
        return self.recurrent(encoded, hidden)

    def distribution_logits(self, recurrent: torch.Tensor) -> torch.Tensor:
        return self.output(recurrent).view(
            *recurrent.shape[:2], self.action_count, self.atoms
        )

    def forward(
        self,
        observations: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        recurrent, hidden = self.recurrent_features(observations, hidden)
        return self.distribution_logits(recurrent), hidden


class RecurrentC51Agent:
    """Train and serve one recurrent action-value policy with hard masks."""

    def __init__(
        self,
        observation_dim: int,
        *,
        hidden_dim: int,
        atoms: int,
        value_min: float,
        value_max: float,
        gamma: float,
        learning_rate: float,
        weight_decay: float,
        gradient_clip: float,
        target_sync_updates: int,
        n_step_return: int = 1,
        recurrent_burn_in: int = 0,
        target_update_mode: str = "hard",
        target_soft_tau: float = 1.0,
        device: str,
        seed: int,
        teacher_channels: int = 0,
        teacher_channel_names: Sequence[str] | None = None,
        teacher_loss_weight: float = 0.0,
        teacher_channel_loss_weights: Sequence[float] | None = None,
        teacher_entry_search_loss_weight: float = 0.0,
        teacher_entry_search_objective: str = "raw_probability",
        teacher_entry_search_centers: Sequence[float] = (0.5, 0.5),
        teacher_entry_search_probability_epsilon: float = 1e-6,
        teacher_entry_search_teacher_temperature: float = 1.0,
        teacher_entry_search_q_temperature: float = 1.0,
        entry_action_loss_weight: float = 0.0,
        entry_action_class_weights: Sequence[float] = (1.0, 1.0, 1.0),
        entry_action_loss_reduction: str = "population_weighted_mean_v1",
        entry_action_margin: float = 0.0,
        regime_selectivity_loss_weight: float = 0.0,
        regime_selectivity_expansion_centers: Sequence[float] | None = None,
        regime_selectivity_probability_epsilon: float = 1e-6,
        regime_selectivity_headroom_pressure: float = 1.0,
        regime_selectivity_dominant_chop_pressure: float = 2.0,
        regime_selectivity_chop_wait_margin: float = 0.0,
        regime_selectivity_failed_confluence_margin: float = 0.0,
        regime_selectivity_q_temperature: float = 1.0,
        regime_selectivity_side_balance: str = "none",
        regime_selectivity_semantics: str = STATIC_STATE_SEMANTICS,
        regime_selectivity_persistent_chop_negative_emphasis: float = 0.0,
        policy_retention_loss_weight: float = 0.0,
        mixed_precision: str = "off",
        compile_model: bool = False,
        compile_backend: str = "inductor",
        compile_mode: str = "default",
        mps_prefer_metal: bool = False,
        mps_fast_math: bool = False,
    ) -> None:
        if atoms < 2 or value_min >= value_max:
            raise ValueError("distributional support is invalid")
        if isinstance(n_step_return, bool) or int(n_step_return) < 1:
            raise ValueError("n-step return must be a positive integer")
        if isinstance(recurrent_burn_in, bool) or int(recurrent_burn_in) < 0:
            raise ValueError("recurrent burn-in must be a nonnegative integer")
        if learning_rate <= 0 or weight_decay < 0 or gradient_clip <= 0:
            raise ValueError("optimizer settings are invalid")
        if target_sync_updates < 1:
            raise ValueError("target_sync_updates must be positive")
        if (
            target_update_mode not in {"hard", "soft"}
            or isinstance(target_soft_tau, bool)
            or not 0 < float(target_soft_tau) <= 1
            or (target_update_mode == "hard" and float(target_soft_tau) != 1.0)
        ):
            raise ValueError("target update contract is invalid")
        if (
            teacher_channels < 0
            or teacher_loss_weight < 0
            or teacher_entry_search_loss_weight < 0
            or entry_action_loss_weight < 0
            or isinstance(entry_action_margin, bool)
            or not np.isfinite(entry_action_margin)
            or entry_action_margin < 0
            or not np.isfinite(regime_selectivity_loss_weight)
            or regime_selectivity_loss_weight < 0
            or isinstance(regime_selectivity_chop_wait_margin, bool)
            or not np.isfinite(regime_selectivity_chop_wait_margin)
            or regime_selectivity_chop_wait_margin < 0
            or isinstance(regime_selectivity_failed_confluence_margin, bool)
            or not np.isfinite(regime_selectivity_failed_confluence_margin)
            or regime_selectivity_failed_confluence_margin < 0
            or policy_retention_loss_weight < 0
        ):
            raise ValueError("teacher settings must be nonnegative")
        if bool(teacher_channels) != bool(teacher_loss_weight):
            raise ValueError("teacher channels and loss weight must be enabled together")
        teacher_channel_names = tuple(
            str(value) for value in (teacher_channel_names or ())
        )
        if teacher_channel_names and len(teacher_channel_names) != teacher_channels:
            raise ValueError("teacher channel names do not match channel count")
        teacher_entry_search_centers = tuple(
            float(value) for value in teacher_entry_search_centers
        )
        if (
            teacher_entry_search_objective not in {
                "raw_probability",
                "centered_log_odds",
            }
            or len(teacher_entry_search_centers) != 2
            or not 0 < teacher_entry_search_probability_epsilon < 0.5
            or any(
                not teacher_entry_search_probability_epsilon
                < value
                < 1.0 - teacher_entry_search_probability_epsilon
                for value in teacher_entry_search_centers
            )
            or teacher_entry_search_teacher_temperature <= 0
            or teacher_entry_search_q_temperature <= 0
            or not np.isfinite(regime_selectivity_q_temperature)
            or regime_selectivity_q_temperature <= 0
        ):
            raise ValueError("teacher entry-search contract is invalid")
        if teacher_channel_loss_weights is None:
            teacher_channel_loss_weights = (
                (float(teacher_loss_weight) / int(teacher_channels),) * int(teacher_channels)
                if teacher_channels else ()
            )
        teacher_channel_loss_weights = tuple(
            float(value) for value in teacher_channel_loss_weights
        )
        if (
            len(teacher_channel_loss_weights) != int(teacher_channels)
            or any(value < 0 for value in teacher_channel_loss_weights)
            or (teacher_channels and not any(teacher_channel_loss_weights))
        ):
            raise ValueError("teacher channel loss weights are invalid")
        entry_action_class_weights = tuple(
            float(value) for value in entry_action_class_weights
        )
        if (
            len(entry_action_class_weights) != 3
            or any(
                not np.isfinite(value) or value <= 0.0
                for value in entry_action_class_weights
            )
        ):
            raise ValueError("entry action class weights are invalid")
        if entry_action_loss_reduction not in _ENTRY_ACTION_LOSS_REDUCTIONS:
            raise ValueError("entry action loss reduction is invalid")
        if mixed_precision not in {"off", "fp16"}:
            raise ValueError("mixed precision must be off or fp16")
        if regime_selectivity_side_balance not in {
            "none",
            "equal_long_short_v1",
        }:
            raise ValueError("Regime selectivity side balance is invalid")
        if (
            regime_selectivity_semantics
            not in {
                STATIC_STATE_SEMANTICS,
                PERSISTENT_CHOP_NEGATIVE_WEIGHT_SEMANTICS,
                PERSISTENT_CHOP_ASSOCIATION_SEMANTICS,
                EXPANSION_REGIME_CONFLUENCE_SEMANTICS,
                SIDE_CONDITIONED_EXPANSION_REGIME_CONFLUENCE_SEMANTICS,
            }
            or not np.isfinite(
                regime_selectivity_persistent_chop_negative_emphasis
            )
            or regime_selectivity_persistent_chop_negative_emphasis < 0.0
        ):
            raise ValueError("Regime selectivity semantics are invalid")
        if (
            regime_selectivity_semantics
            in {
                PERSISTENT_CHOP_NEGATIVE_WEIGHT_SEMANTICS,
                PERSISTENT_CHOP_ASSOCIATION_SEMANTICS,
                EXPANSION_REGIME_CONFLUENCE_SEMANTICS,
                SIDE_CONDITIONED_EXPANSION_REGIME_CONFLUENCE_SEMANTICS,
            }
            and regime_selectivity_side_balance != "equal_long_short_v1"
        ):
            raise ValueError(
                "persistent-chop Regime selectivity requires equal Long/Short groups"
            )
        torch.manual_seed(seed)
        self.seed = int(seed)
        self._rng = np.random.default_rng(seed)
        self.device = resolve_device(device)
        if mixed_precision == "fp16" and self.device.type not in {"mps", "cuda"}:
            raise ValueError("fp16 mixed precision requires MPS or CUDA")
        self.mixed_precision = mixed_precision
        self.compile_model = bool(compile_model)
        self.compile_backend = str(compile_backend)
        self.compile_mode = str(compile_mode)
        self.mps_prefer_metal = bool(mps_prefer_metal)
        self.mps_fast_math = bool(mps_fast_math)
        self.runtime_environment: dict[str, str] = {}
        if self.device.type == "mps":
            self.runtime_environment = configure_runtime_environment({
                "mps_prefer_metal": self.mps_prefer_metal,
                "mps_fast_math": self.mps_fast_math,
            })
        self.observation_dim = int(observation_dim)
        self.hidden_dim = int(hidden_dim)
        self.atoms = int(atoms)
        self.gamma = float(gamma)
        self.n_step_return = int(n_step_return)
        self.recurrent_burn_in = int(recurrent_burn_in)
        self.learning_rate = float(learning_rate)
        self.value_min = float(value_min)
        self.value_max = float(value_max)
        self.target_sync_updates = int(target_sync_updates)
        self.target_update_mode = str(target_update_mode)
        self.target_soft_tau = float(target_soft_tau)
        self.weight_decay = float(weight_decay)
        self.gradient_clip = float(gradient_clip)
        self.teacher_channels = int(teacher_channels)
        self.teacher_channel_names = teacher_channel_names
        self.teacher_loss_weight = float(teacher_loss_weight)
        self.teacher_channel_loss_weights = teacher_channel_loss_weights
        self.teacher_entry_search_loss_weight = float(
            teacher_entry_search_loss_weight
        )
        self.teacher_entry_search_objective = str(
            teacher_entry_search_objective
        )
        self.teacher_entry_search_centers = teacher_entry_search_centers
        self.teacher_entry_search_probability_epsilon = float(
            teacher_entry_search_probability_epsilon
        )
        self.teacher_entry_search_teacher_temperature = float(
            teacher_entry_search_teacher_temperature
        )
        self.teacher_entry_search_q_temperature = float(
            teacher_entry_search_q_temperature
        )
        self.entry_action_loss_weight = float(entry_action_loss_weight)
        self.entry_action_class_weights = entry_action_class_weights
        self.entry_action_loss_reduction = str(entry_action_loss_reduction)
        self.entry_action_margin = float(entry_action_margin)
        self.regime_selectivity_loss_weight = float(
            regime_selectivity_loss_weight
        )
        self.regime_selectivity_expansion_centers = (
            None
            if regime_selectivity_expansion_centers is None
            else tuple(float(value) for value in regime_selectivity_expansion_centers)
        )
        self.regime_selectivity_probability_epsilon = float(
            regime_selectivity_probability_epsilon
        )
        self.regime_selectivity_headroom_pressure = float(
            regime_selectivity_headroom_pressure
        )
        self.regime_selectivity_dominant_chop_pressure = float(
            regime_selectivity_dominant_chop_pressure
        )
        self.regime_selectivity_chop_wait_margin = float(
            regime_selectivity_chop_wait_margin
        )
        self.regime_selectivity_failed_confluence_margin = float(
            regime_selectivity_failed_confluence_margin
        )
        self.regime_selectivity_q_temperature = float(
            regime_selectivity_q_temperature
        )
        self.regime_selectivity_side_balance = str(
            regime_selectivity_side_balance
        )
        self.regime_selectivity_semantics = str(regime_selectivity_semantics)
        self.regime_selectivity_persistent_chop_negative_emphasis = float(
            regime_selectivity_persistent_chop_negative_emphasis
        )
        self.regime_selectivity = (
            BalanceAwareRegimeSelectivity(
                channel_names=self.teacher_channel_names,
                expansion_centers=(
                    self.regime_selectivity_expansion_centers or ()
                ),
                probability_epsilon=self.regime_selectivity_probability_epsilon,
                headroom_pressure=self.regime_selectivity_headroom_pressure,
                dominant_chop_pressure=(
                    self.regime_selectivity_dominant_chop_pressure
                ),
                semantics=self.regime_selectivity_semantics,
                persistent_chop_negative_emphasis=(
                    self.regime_selectivity_persistent_chop_negative_emphasis
                ),
            )
            if self.regime_selectivity_loss_weight
            else None
        )
        self.policy_retention_loss_weight = float(policy_retention_loss_weight)
        self.retention_anchor: RecurrentC51Network | None = None
        self.retention_anchor_applies_to_all_management_rows = False
        self.last_train_metrics: dict[str, float] = {}
        self.support = torch.linspace(value_min, value_max, atoms, device=self.device)
        # Immutable training constants stay device-resident across optimizer
        # updates. Recreating them in train_batch forces allocator work and can
        # introduce avoidable MPS synchronization points.
        self._teacher_channel_loss_weights_tensor = torch.tensor(
            self.teacher_channel_loss_weights,
            dtype=torch.float32,
            device=self.device,
        )
        self._entry_action_class_weights_tensor = torch.tensor(
            self.entry_action_class_weights,
            dtype=torch.float32,
            device=self.device,
        )
        self._flat_action_indices = torch.tensor(
            (
                int(Action.WAIT),
                int(Action.ENTER_LONG_1),
                int(Action.ENTER_SHORT_1),
            ),
            dtype=torch.long,
            device=self.device,
        )
        self.regime_teacher_channel_names = tuple(
            channel
            for channel in REGIME_TEACHER_CHANNELS
            if channel in self.teacher_channel_names
        )
        self._regime_teacher_channel_indices_tensor = torch.tensor(
            tuple(
                self.teacher_channel_names.index(channel)
                for channel in self.regime_teacher_channel_names
            ),
            dtype=torch.long,
            device=self.device,
        )
        self.online = RecurrentC51Network(
            observation_dim, len(Action), atoms, hidden_dim, self.teacher_channels
        ).to(self.device)
        self.target = copy.deepcopy(self.online).to(self.device).eval()
        self.optimizer = torch.optim.AdamW(
            self.online.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        self.scaler = torch.amp.GradScaler(
            self.device.type,
            enabled=self.mixed_precision == "fp16",
        )
        self.compile_status = "disabled"
        self.compile_error = ""
        self._configure_execution()
        self._updates = 0

    @torch.no_grad()
    def _update_target_network(self) -> None:
        """Apply the recipe-declared stable target-network update."""
        if self.target_update_mode == "hard":
            if self._updates % self.target_sync_updates == 0:
                self.target.load_state_dict(self.online.state_dict())
            return
        for target_parameter, online_parameter in zip(
            self.target.parameters(), self.online.parameters(), strict=True
        ):
            target_parameter.lerp_(online_parameter, self.target_soft_tau)
        for target_buffer, online_buffer in zip(
            self.target.buffers(), self.online.buffers(), strict=True
        ):
            target_buffer.copy_(online_buffer)

    def _configure_execution(self) -> None:
        """Build optional compiled callables without changing module state keys."""
        self._online_recurrent = self.online.recurrent_features
        self._online_forward = self.online.forward
        self._target_forward = self.target.forward
        if not self.compile_model:
            return
        try:
            self._online_recurrent = torch.compile(
                self.online.recurrent_features,
                backend=self.compile_backend,
                mode=self.compile_mode,
            )
            self._online_forward = torch.compile(
                self.online.forward,
                backend=self.compile_backend,
                mode=self.compile_mode,
            )
            self._target_forward = torch.compile(
                self.target.forward,
                backend=self.compile_backend,
                mode=self.compile_mode,
            )
            self.compile_status = "compiled"
        except Exception as error:  # torch.compile availability varies by platform.
            self.compile_status = "fallback_eager"
            self.compile_error = f"{type(error).__name__}: {error}"

    def _autocast(self):
        if self.mixed_precision == "off":
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=torch.float16)

    @staticmethod
    def _recurrent_features_with_resets(
        network: RecurrentC51Network,
        observations: torch.Tensor,
        reset_rows: Sequence[Sequence[bool]],
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate exact behavior-time GRU state across declared reset points."""
        if (
            len(reset_rows) != observations.shape[0]
            or any(len(row) != observations.shape[1] for row in reset_rows)
        ):
            raise ValueError("recurrent reset mask does not match observations")
        grouped: dict[tuple[int, ...], list[int]] = {}
        for batch_index, row in enumerate(reset_rows):
            pattern = tuple(index for index, reset in enumerate(row) if reset)
            grouped.setdefault(pattern, []).append(batch_index)
        outputs: list[torch.Tensor] = []
        final_hidden: list[torch.Tensor] = []
        order: list[int] = []
        for pattern, indices in grouped.items():
            index = torch.as_tensor(indices, dtype=torch.long, device=observations.device)
            values = observations.index_select(0, index)
            current_hidden = (
                None if hidden is None else hidden.index_select(1, index)
            )
            pieces: list[torch.Tensor] = []
            boundaries = sorted({0, *pattern, observations.shape[1]})
            for start, stop in zip(boundaries, boundaries[1:]):
                if start in pattern:
                    current_hidden = None
                if stop > start:
                    piece, current_hidden = network.recurrent_features(
                        values[:, start:stop], current_hidden
                    )
                    pieces.append(piece)
            outputs.append(torch.cat(pieces, dim=1))
            assert current_hidden is not None
            final_hidden.append(current_hidden)
            order.extend(indices)
        inverse = torch.as_tensor(
            np.argsort(np.asarray(order)), dtype=torch.long, device=observations.device
        )
        return (
            torch.cat(outputs, dim=0).index_select(0, inverse),
            torch.cat(final_hidden, dim=1).index_select(1, inverse),
        )

    def _call_with_compile_fallback(self, compiled, eager, *args):
        if self.compile_status != "compiled":
            return eager(*args)
        try:
            return compiled(*args)
        except Exception as error:  # Lazy graph lowering can fail on MPS.
            self.compile_status = "fallback_eager"
            self.compile_error = f"{type(error).__name__}: {error}"
            self._online_recurrent = self.online.recurrent_features
            self._online_forward = self.online.forward
            self._target_forward = self.target.forward
            return eager(*args)

    def select_action(
        self,
        observation: np.ndarray,
        *,
        hidden: torch.Tensor | None,
        valid_actions: tuple[Action, ...],
        epsilon: float,
        return_action_values: bool = False,
    ) -> tuple[Action, torch.Tensor, np.ndarray | None]:
        if not valid_actions:
            raise ValueError("at least one action must be valid")
        explore = epsilon > 0.0 and self._rng.random() < epsilon
        selected = (
            valid_actions[int(self._rng.integers(len(valid_actions)))]
            if explore else None
        )
        if explore and not return_action_values:
            with torch.no_grad():
                value = torch.as_tensor(
                    observation, dtype=torch.float32, device=self.device
                ).view(1, 1, -1)
                with self._autocast():
                    _, next_hidden = self._call_with_compile_fallback(
                        self._online_forward, self.online.forward, value, hidden
                    )
            return selected, next_hidden.detach(), None
        with torch.no_grad():
            value = torch.as_tensor(
                observation, dtype=torch.float32, device=self.device
            ).view(1, 1, -1)
            with self._autocast():
                logits, next_hidden = self._call_with_compile_fallback(
                    self._online_forward, self.online.forward, value, hidden
                )
            q_values = (logits.float().softmax(-1) * self.support).sum(-1)[0, 0]
            valid = torch.zeros(len(Action), dtype=torch.bool, device=self.device)
            valid[[int(action) for action in valid_actions]] = True
            q_values = q_values.masked_fill(~valid, -torch.inf)
            if selected is None:
                selected = Action(int(q_values.argmax().item()))
        values = q_values.cpu().numpy() if return_action_values else None
        return selected, next_hidden.detach(), values

    @torch.no_grad()
    def greedy_sequence_action_values(
        self,
        sequences: Sequence[Sequence[Transition]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Score fixed recurrent traces without exploration or state mutation."""
        if not sequences:
            raise ValueError("greedy sequence probe cannot be empty")
        lengths = {len(sequence) for sequence in sequences}
        if len(lengths) != 1 or next(iter(lengths)) < 1:
            raise ValueError("greedy sequence probe lengths are inconsistent")
        observation_rows = np.stack([
            [transition.observation for transition in sequence]
            for sequence in sequences
        ])
        if not np.isfinite(observation_rows).all():
            raise ValueError("greedy sequence probe observations are invalid")
        observations = torch.as_tensor(
            observation_rows,
            dtype=torch.float32,
            device=self.device,
        )
        reset_rows = tuple(
            tuple(transition.recurrent_reset for transition in sequence)
            for sequence in sequences
        )
        valid_masks = torch.as_tensor(
            [[
                [action in transition.valid_actions for action in Action]
                for transition in sequence
            ] for sequence in sequences],
            dtype=torch.bool,
            device=self.device,
        )
        if not bool(valid_masks.any(-1).all()):
            raise ValueError("greedy sequence probe has a row without valid actions")
        with self._autocast():
            recurrent, _ = self._recurrent_features_with_resets(
                self.online,
                observations,
                reset_rows,
            )
            logits = self.online.distribution_logits(recurrent)
        values = (logits.float().softmax(-1) * self.support).sum(-1)
        values = values.masked_fill(~valid_masks, -torch.inf)
        actions = values.argmax(-1)
        return (
            actions.detach().cpu().numpy(),
            values.detach().cpu().numpy(),
        )

    def train_batch(
        self,
        sequences: Sequence[Sequence[Transition]],
        *,
        teacher_weight_scale: float = 1.0,
        entry_action_weight_scale: float = 1.0,
    ) -> float:
        if not sequences:
            raise ValueError("training batch cannot be empty")
        teacher_weight_scale = float(teacher_weight_scale)
        entry_action_weight_scale = float(entry_action_weight_scale)
        if (
            not np.isfinite(teacher_weight_scale)
            or not 0 <= teacher_weight_scale <= 1
        ):
            raise ValueError("teacher weight scale must be between zero and one")
        if (
            not np.isfinite(entry_action_weight_scale)
            or not 0 <= entry_action_weight_scale <= 1
        ):
            raise ValueError(
                "entry timing weight scale must be between zero and one"
            )
        lengths = {len(sequence) for sequence in sequences}
        if len(lengths) != 1 or next(iter(lengths)) < 1:
            raise ValueError("training sequences must have one positive length")
        sequence_length = next(iter(lengths))
        if sequence_length < self.recurrent_burn_in + self.n_step_return:
            raise ValueError(
                "training sequence is shorter than recurrent burn-in plus n-step return"
            )
        training_steps = (
            sequence_length - self.recurrent_burn_in - self.n_step_return + 1
        )
        observations = torch.as_tensor(
            np.stack([[item.observation for item in sequence] for sequence in sequences]),
            dtype=torch.float32,
            device=self.device,
        )
        next_observations = torch.as_tensor(
            np.stack([[item.next_observation for item in sequence] for sequence in sequences]),
            dtype=torch.float32,
            device=self.device,
        )
        all_actions = torch.as_tensor(
            [[int(item.action) for item in sequence] for sequence in sequences],
            dtype=torch.long,
            device=self.device,
        )
        reward_rows = np.asarray(
            [[item.reward for item in sequence] for sequence in sequences],
            dtype=np.float32,
        )
        if not np.isfinite(reward_rows).all():
            raise ValueError("training loss is non-finite")
        all_rewards = torch.as_tensor(
            reward_rows,
            dtype=torch.float32,
            device=self.device,
        )
        all_terminated = torch.as_tensor(
            [[item.terminated for item in sequence] for sequence in sequences],
            dtype=torch.bool,
            device=self.device,
        )
        all_next_masks = torch.as_tensor(
            [[
                [action in item.next_valid_actions for action in Action]
                for item in sequence
            ] for sequence in sequences],
            dtype=torch.bool,
            device=self.device,
        )
        recurrent_reset_rows = tuple(
            tuple(item.recurrent_reset for item in sequence)
            for sequence in sequences
        )
        next_recurrent_reset_rows = tuple(
            tuple(item.next_recurrent_reset for item in sequence)
            for sequence in sequences
        )
        valid_masks = torch.as_tensor(
            [[
                [action in item.valid_actions for action in Action]
                for item in sequence
            ] for sequence in sequences],
            dtype=torch.bool,
            device=self.device,
        )
        competence_anchors = torch.as_tensor(
            [[item.competence_anchor for item in sequence] for sequence in sequences],
            dtype=torch.bool,
            device=self.device,
        )
        training_valid = torch.as_tensor(
            [[item.training_valid for item in sequence] for sequence in sequences],
            dtype=torch.bool,
            device=self.device,
        )
        # Process one contiguous causal trace so Q(s_t) and Q(s_{t+1}) use
        # exactly the same recurrent history. Independently resetting the GRU
        # on the shifted next-observation sequence changes the state definition
        # and corrupts recurrent TD targets.
        causal_observations = torch.cat(
            (observations[:, :1], next_observations), dim=1
        )
        causal_reset_rows = tuple(
            (current[0], *following)
            for current, following in zip(
                recurrent_reset_rows, next_recurrent_reset_rows, strict=True
            )
        )
        burn_in_reset_rows = tuple(
            row[:self.recurrent_burn_in] for row in causal_reset_rows
        )
        learning_reset_rows = tuple(
            row[self.recurrent_burn_in:] for row in causal_reset_rows
        )

        learning_start = self.recurrent_burn_in
        actions = all_actions[:, learning_start:learning_start + training_steps]
        immediate_rewards = all_rewards[
            :, learning_start:learning_start + training_steps
        ]
        learning_valid = training_valid[
            :, learning_start:learning_start + training_steps
        ]
        auxiliary_valid = training_valid[:, learning_start:]
        n_step_rewards = torch.zeros_like(immediate_rewards)
        terminal_targets = torch.zeros_like(immediate_rewards, dtype=torch.bool)
        complete_n_step = torch.ones_like(immediate_rewards, dtype=torch.bool)
        target_offsets = torch.ones_like(actions)
        alive = torch.ones_like(immediate_rewards, dtype=torch.bool)
        discount = 1.0
        for offset in range(self.n_step_return):
            available = training_valid[
                :,
                learning_start + offset:learning_start + offset + training_steps,
            ]
            reward_slice = all_rewards[
                :,
                learning_start + offset:learning_start + offset + training_steps,
            ]
            terminated_slice = all_terminated[
                :,
                learning_start + offset:learning_start + offset + training_steps,
            ]
            active = alive & available & complete_n_step
            n_step_rewards = n_step_rewards + (
                discount * active.to(all_rewards.dtype) * reward_slice
            )
            terminal_event = active & terminated_slice
            terminal_targets = terminal_targets | terminal_event
            target_offsets = torch.where(
                terminal_event,
                torch.full_like(target_offsets, offset + 1),
                target_offsets,
            )
            alive = alive & ~terminal_event
            complete_n_step = complete_n_step & (available | terminal_targets)
            discount *= self.gamma
        full_horizon = complete_n_step & ~terminal_targets
        learnable_rows = learning_valid & (terminal_targets | full_horizon)
        # Replay authenticates sampled sequences before device transfer. Keep
        # this fail-closed preflight off the accelerator hot path.
        def authentic_learning_row(
            sequence: Sequence[Transition], candidate_index: int
        ) -> bool:
            if not sequence[candidate_index].training_valid:
                return False
            for offset in range(self.n_step_return):
                transition = sequence[candidate_index + offset]
                if not transition.training_valid:
                    return False
                if transition.terminated:
                    return True
            return True

        if not any(
            authentic_learning_row(sequence, learning_start + time_index)
            for sequence in sequences
            for time_index in range(training_steps)
        ):
            raise ValueError("training batch has no valid learning rows")
        target_offsets = torch.where(
            full_horizon,
            torch.full_like(target_offsets, self.n_step_return),
            target_offsets,
        )
        candidate_indices = torch.arange(
            training_steps, dtype=torch.long, device=self.device
        ).view(1, -1)
        target_state_indices = candidate_indices + target_offsets
        target_transition_indices = target_state_indices - 1
        next_masks = all_next_masks[:, learning_start:].gather(
            1,
            target_transition_indices[..., None].expand(-1, -1, len(Action)),
        )

        online_hidden = None
        if self.recurrent_burn_in:
            with torch.no_grad(), self._autocast():
                _, online_hidden = self._recurrent_features_with_resets(
                    self.online,
                    causal_observations[:, :self.recurrent_burn_in],
                    burn_in_reset_rows,
                )
            online_hidden = online_hidden.detach()
        with self._autocast():
            causal_recurrent, _ = self._recurrent_features_with_resets(
                self.online,
                causal_observations[:, self.recurrent_burn_in:],
                learning_reset_rows,
                online_hidden,
            )
            recurrent = causal_recurrent[:, :-1]
            all_logits = self.online.distribution_logits(recurrent)
            logits = all_logits[:, :training_steps]
            online_causal = self.online.distribution_logits(causal_recurrent)
            online_next = online_causal.gather(
                1,
                target_state_indices[..., None, None].expand(
                    -1, -1, len(Action), self.atoms
                ),
            )
        logits = logits.float()
        online_next = online_next.float()
        chosen_logits = logits.gather(
            2,
            actions[..., None, None].expand(-1, -1, 1, self.atoms),
        ).squeeze(2)
        with torch.no_grad():
            target_hidden = None
            if self.recurrent_burn_in:
                with self._autocast():
                    _, target_hidden = self._recurrent_features_with_resets(
                        self.target,
                        causal_observations[:, :self.recurrent_burn_in],
                        burn_in_reset_rows,
                    )
            with self._autocast():
                target_recurrent, _ = self._recurrent_features_with_resets(
                    self.target,
                    causal_observations[:, self.recurrent_burn_in:],
                    learning_reset_rows,
                    target_hidden,
                )
                target_causal = self.target.distribution_logits(target_recurrent)
                target_next = target_causal.gather(
                    1,
                    target_state_indices[..., None, None].expand(
                        -1, -1, len(Action), self.atoms
                    ),
                )
            online_q = (online_next.float().softmax(-1) * self.support).sum(-1)
            online_q = online_q.masked_fill(~next_masks, -torch.inf)
            next_actions = online_q.argmax(-1)
            target_distribution = target_next.float().softmax(-1).gather(
                2,
                next_actions[..., None, None].expand(-1, -1, 1, self.atoms),
            ).squeeze(2)
            projected = self._project_distribution(
                target_distribution,
                n_step_rewards,
                terminal_targets,
                bootstrap_discount=self.gamma**self.n_step_return,
            )
        td_losses = -(projected * chosen_logits.log_softmax(-1)).sum(-1)
        rl_loss = td_losses[learnable_rows].mean()
        loss = rl_loss
        teacher_loss = torch.zeros((), dtype=torch.float32, device=self.device)
        entry_search_loss = torch.zeros((), dtype=torch.float32, device=self.device)
        entry_action_loss = torch.zeros((), dtype=torch.float32, device=self.device)
        entry_action_margin_loss = torch.zeros(
            (), dtype=torch.float32, device=self.device
        )
        regime_selectivity_loss = teacher_loss
        regime_selectivity_rows = teacher_loss
        regime_selectivity_target_wait_mean = teacher_loss
        regime_selectivity_low_headroom_rows = teacher_loss
        regime_selectivity_low_headroom_wait_mean = teacher_loss
        regime_selectivity_dominant_chop_rows = teacher_loss
        regime_selectivity_dominant_chop_wait_mean = teacher_loss
        regime_selectivity_positive_long_loss = teacher_loss
        regime_selectivity_positive_short_loss = teacher_loss
        regime_selectivity_exact_wait_loss = teacher_loss
        regime_selectivity_association_loss = teacher_loss
        regime_selectivity_association_active = teacher_loss
        regime_selectivity_association_skipped = teacher_loss
        regime_selectivity_side_conditioned_loss = teacher_loss
        regime_selectivity_side_conditioned_active_sides = teacher_loss
        regime_selectivity_additive: dict[str, torch.Tensor] = {}
        regime_channel_names = self.regime_teacher_channel_names
        regime_channel_additive = torch.zeros(
            (
                len(regime_channel_names),
                len(_REGIME_CHANNEL_ADDITIVE_FIELDS),
            ),
            dtype=torch.float32,
            device=self.device,
        )
        regime_persistent_additive = {
            name: torch.zeros((), dtype=torch.float32, device=self.device)
            for name in _REGIME_PERSISTENT_ADDITIVE_METRICS
        }
        entry_action_supervised_rows = torch.zeros(
            (), dtype=torch.float32, device=self.device
        )
        entry_action_target_counts = torch.zeros(
            3, dtype=torch.float32, device=self.device
        )
        entry_action_prediction_counts = torch.zeros(
            3, dtype=torch.float32, device=self.device
        )
        entry_action_correct_counts = torch.zeros(
            3, dtype=torch.float32, device=self.device
        )
        entry_balance_additive = {
            f"entry_balance_{action}_{field}": torch.zeros(
                (), dtype=torch.float32, device=self.device
            )
            for action in _ENTRY_BALANCE_ACTION_NAMES
            for field in _ENTRY_BALANCE_ADDITIVE_FIELDS
        }
        regime_entry_conflict_additive = {
            f"regime_entry_conflict_{side}_{field}": torch.zeros(
                (), dtype=torch.float32, device=self.device
            )
            for side in ("long", "short")
            for field in _REGIME_ENTRY_CONFLICT_FIELDS
        }
        entry_diagnostics_active = (
            (
                self.regime_selectivity is not None
                and entry_action_weight_scale > 0.0
            )
            or (
                self.entry_action_loss_weight > 0.0
                and entry_action_weight_scale > 0.0
            )
        )
        diagnostic_action_targets: np.ndarray | None = None
        diagnostic_targets: torch.Tensor | None = None
        diagnostic_flat_rows: torch.Tensor | None = None
        diagnostic_target_rows: torch.Tensor | None = None
        if entry_diagnostics_active:
            diagnostic_action_targets = np.full(
                observations.shape[:2], -1, dtype=np.int64
            )
            for batch_index, sequence in enumerate(sequences):
                for time_index, transition in enumerate(sequence):
                    if transition.entry_action_target is not None:
                        try:
                            target = Action(transition.entry_action_target)
                        except (TypeError, ValueError) as error:
                            raise ValueError(
                                "entry timing target is invalid"
                            ) from error
                        if target not in {
                            Action.WAIT,
                            Action.ENTER_LONG_1,
                            Action.ENTER_SHORT_1,
                        }:
                            raise ValueError("entry timing target is invalid")
                        diagnostic_action_targets[batch_index, time_index] = int(
                            target
                        )
            diagnostic_targets = torch.as_tensor(
                diagnostic_action_targets[
                    :,
                    self.recurrent_burn_in:
                    self.recurrent_burn_in + training_steps,
                ],
                dtype=torch.long,
                device=self.device,
            )
            diagnostic_flat_rows = (
                valid_masks[
                    :,
                    self.recurrent_burn_in:
                    self.recurrent_burn_in + training_steps,
                    int(Action.WAIT),
                ]
                & valid_masks[
                    :,
                    self.recurrent_burn_in:
                    self.recurrent_burn_in + training_steps,
                    int(Action.ENTER_LONG_1),
                ]
                & valid_masks[
                    :,
                    self.recurrent_burn_in:
                    self.recurrent_burn_in + training_steps,
                    int(Action.ENTER_SHORT_1),
                ]
            )
            diagnostic_target_rows = (
                (diagnostic_targets >= 0)
                & diagnostic_flat_rows
                & learnable_rows
            )
            entry_action_target_counts = torch.bincount(
                diagnostic_targets[diagnostic_target_rows], minlength=3
            ).to(torch.float32)
        policy_retention_loss = torch.zeros(
            (), dtype=torch.float32, device=self.device
        )
        # Teacher dropout and its curriculum scale govern optional imitation
        # only. Exact action and Regime-confluence supervision remain training
        # labels and are never policy observations.
        if self.teacher_channels and (
            teacher_weight_scale > 0.0
            or (
                self.regime_selectivity is not None
                and entry_action_weight_scale > 0.0
            )
        ):
            teacher_targets = np.full(
                (*observations.shape[:2], self.teacher_channels),
                np.nan,
                dtype=np.float32,
            )
            teacher_imitation_visible = np.zeros(
                observations.shape[:2], dtype=np.bool_
            )
            for batch_index, sequence in enumerate(sequences):
                for time_index, transition in enumerate(sequence):
                    if transition.teacher_target is not None:
                        target = np.asarray(
                            transition.teacher_target, dtype=np.float32
                        ).reshape(-1)
                        if target.shape != (self.teacher_channels,):
                            raise ValueError("teacher target width drifted")
                        teacher_targets[batch_index, time_index] = target
                        teacher_imitation_visible[batch_index, time_index] = bool(
                            transition.teacher_imitation_visible
                        )
            all_teacher_rows_numpy = np.isfinite(teacher_targets).all(axis=-1)
            teacher_rows_numpy = (
                all_teacher_rows_numpy & teacher_imitation_visible
            )
            teacher_targets_tensor = torch.as_tensor(
                teacher_targets, dtype=torch.float32, device=self.device
            )[:, self.recurrent_burn_in:]
            teacher_rows = torch.as_tensor(
                teacher_rows_numpy[:, self.recurrent_burn_in:],
                device=self.device,
            ) & auxiliary_valid
            all_teacher_rows = torch.as_tensor(
                all_teacher_rows_numpy[:, self.recurrent_burn_in:],
                device=self.device,
            ) & auxiliary_valid
            if bool(all_teacher_rows.any().item()):
                if bool(teacher_rows.any().item()):
                    assert self.online.teacher_output is not None
                    with self._autocast():
                        teacher_logits = self.online.teacher_output(recurrent)
                    teacher_losses = nn.functional.binary_cross_entropy_with_logits(
                        teacher_logits.float()[teacher_rows],
                        teacher_targets_tensor[teacher_rows],
                        reduction="none",
                    )
                    teacher_probabilities = (
                        teacher_logits.float()[teacher_rows].sigmoid()
                    )
                    teacher_probability_targets = teacher_targets_tensor[teacher_rows]
                    if regime_channel_names:
                        regime_targets = teacher_probability_targets.index_select(
                            -1, self._regime_teacher_channel_indices_tensor
                        )
                        regime_predictions = teacher_probabilities.index_select(
                            -1, self._regime_teacher_channel_indices_tensor
                        )
                        regime_errors = regime_predictions - regime_targets
                        teacher_row_count = teacher_rows.sum().to(torch.float32)
                        regime_channel_additive = torch.stack((
                            teacher_row_count.expand(len(regime_channel_names)),
                            regime_targets.sum(dim=0),
                            regime_predictions.sum(dim=0),
                            regime_errors.abs().sum(dim=0),
                            regime_errors.square().sum(dim=0),
                        ), dim=-1)
                    teacher_loss = (
                        teacher_losses * self._teacher_channel_loss_weights_tensor
                    ).sum(dim=-1).mean()
                    loss = loss + teacher_weight_scale * teacher_loss
                if (
                    self.teacher_entry_search_loss_weight
                    and bool(teacher_rows.any().item())
                ):
                    entry_rows = (
                        teacher_rows
                        & valid_masks[
                            :, self.recurrent_burn_in:, int(Action.WAIT)
                        ]
                        & valid_masks[
                            :, self.recurrent_burn_in:, int(Action.ENTER_LONG_1)
                        ]
                        & valid_masks[
                            :, self.recurrent_burn_in:, int(Action.ENTER_SHORT_1)
                        ]
                    )
                    if bool(entry_rows.any().item()):
                        q_values = (all_logits.float().softmax(-1) * self.support).sum(-1)
                        long_target = (
                            teacher_targets_tensor[..., 0]
                            * teacher_targets_tensor[..., 1]
                        ).nan_to_num(0.0).clamp(0.0, 1.0)
                        short_target = (
                            teacher_targets_tensor[..., 2]
                            * teacher_targets_tensor[..., 3]
                        ).nan_to_num(0.0).clamp(0.0, 1.0)
                        if self.teacher_entry_search_objective == "centered_log_odds":
                            long_target = centered_entry_search_target(
                                long_target,
                                center=self.teacher_entry_search_centers[0],
                                probability_epsilon=(
                                    self.teacher_entry_search_probability_epsilon
                                ),
                                teacher_temperature=(
                                    self.teacher_entry_search_teacher_temperature
                                ),
                            )
                            short_target = centered_entry_search_target(
                                short_target,
                                center=self.teacher_entry_search_centers[1],
                                probability_epsilon=(
                                    self.teacher_entry_search_probability_epsilon
                                ),
                                teacher_temperature=(
                                    self.teacher_entry_search_teacher_temperature
                                ),
                            )
                        long_advantage = (
                            q_values[..., int(Action.ENTER_LONG_1)]
                            - q_values[..., int(Action.WAIT)]
                        ) / self.teacher_entry_search_q_temperature
                        short_advantage = (
                            q_values[..., int(Action.ENTER_SHORT_1)]
                            - q_values[..., int(Action.WAIT)]
                        ) / self.teacher_entry_search_q_temperature
                        entry_weights = entry_rows.to(q_values.dtype)
                        entry_count = entry_weights.sum().clamp_min(1.0)
                        entry_search_loss = 0.5 * (
                            (
                                nn.functional.binary_cross_entropy_with_logits(
                                    long_advantage,
                                    long_target,
                                    reduction="none",
                                )
                                * entry_weights
                            ).sum()
                            / entry_count
                            + (
                                nn.functional.binary_cross_entropy_with_logits(
                                    short_advantage,
                                    short_target,
                                    reduction="none",
                                )
                                * entry_weights
                            ).sum()
                            / entry_count
                        )
                        loss = loss + teacher_weight_scale * (
                            self.teacher_entry_search_loss_weight
                            * entry_search_loss
                        )
                if self.regime_selectivity is not None:
                    assert diagnostic_targets is not None
                    headroom = np.full(
                        observations.shape[:2], np.nan, dtype=np.float32
                    )
                    for batch_index, sequence in enumerate(sequences):
                        for time_index, transition in enumerate(sequence):
                            value = transition.regime_selectivity_headroom_fraction
                            if value is not None:
                                headroom[batch_index, time_index] = float(value)
                    headroom_tensor = torch.as_tensor(
                        headroom,
                        dtype=torch.float32,
                        device=self.device,
                    )[:, self.recurrent_burn_in:]
                    selectivity_action_targets_tensor = diagnostic_targets
                    selectivity_teacher_targets = teacher_targets_tensor[
                        :, :training_steps
                    ]
                    selectivity_headroom = headroom_tensor[:, :training_steps]
                    exact_flat_rows = (
                        all_teacher_rows[:, :training_steps]
                        & learnable_rows
                        & torch.isfinite(selectivity_headroom)
                        & (selectivity_action_targets_tensor >= int(Action.WAIT))
                        & (
                            selectivity_action_targets_tensor
                            <= int(Action.ENTER_SHORT_1)
                        )
                        & valid_masks[
                            :,
                            self.recurrent_burn_in:
                            self.recurrent_burn_in + training_steps,
                            int(Action.WAIT),
                        ]
                        & valid_masks[
                            :,
                            self.recurrent_burn_in:
                            self.recurrent_burn_in + training_steps,
                            int(Action.ENTER_LONG_1),
                        ]
                        & valid_masks[
                            :,
                            self.recurrent_burn_in:
                            self.recurrent_burn_in + training_steps,
                            int(Action.ENTER_SHORT_1),
                        ]
                    )
                    positive_rows_mask = exact_flat_rows & (
                        selectivity_action_targets_tensor != int(Action.WAIT)
                    )
                    selectivity_rows = (
                        exact_flat_rows
                        if self.regime_selectivity_semantics
                        in {
                            PERSISTENT_CHOP_NEGATIVE_WEIGHT_SEMANTICS,
                            PERSISTENT_CHOP_ASSOCIATION_SEMANTICS,
                            EXPANSION_REGIME_CONFLUENCE_SEMANTICS,
                            SIDE_CONDITIONED_EXPANSION_REGIME_CONFLUENCE_SEMANTICS,
                        }
                        else positive_rows_mask
                    )
                    selected_teachers = selectivity_teacher_targets[
                        selectivity_rows
                    ]
                    selected_headroom = selectivity_headroom[selectivity_rows]
                    selected_actions = selectivity_action_targets_tensor[
                        selectivity_rows
                    ]
                    q_values = (
                        all_logits[:, :training_steps].float().softmax(-1)
                        * self.support
                    ).sum(-1)
                    flat_q = q_values.index_select(
                        -1, self._flat_action_indices
                    )[selectivity_rows]
                    model_log_probabilities = nn.functional.log_softmax(
                        flat_q / self.regime_selectivity_q_temperature,
                        dim=-1,
                    )
                    positive_rows = selectivity_rows.sum().to(torch.float32)
                    if (
                        self.regime_selectivity_semantics
                        in {
                            PERSISTENT_CHOP_NEGATIVE_WEIGHT_SEMANTICS,
                            PERSISTENT_CHOP_ASSOCIATION_SEMANTICS,
                            EXPANSION_REGIME_CONFLUENCE_SEMANTICS,
                            SIDE_CONDITIONED_EXPANSION_REGIME_CONFLUENCE_SEMANTICS,
                        }
                    ):
                        compiler = self.regime_selectivity
                        persistent_evidence = (
                            compiler.exact_wait_negative_weight_evidence(
                                selected_teachers,
                                selected_actions,
                            )
                        )
                        wait_weights = persistent_evidence.exact_wait_weights
                        exact_losses = nn.functional.nll_loss(
                            model_log_probabilities,
                            selected_actions,
                            reduction="none",
                        )
                        wait_rows = (
                            selected_actions == int(Action.WAIT)
                        ).to(exact_losses.dtype)
                        long_rows = (
                            selected_actions == int(Action.ENTER_LONG_1)
                        ).to(exact_losses.dtype)
                        short_rows = (
                            selected_actions == int(Action.ENTER_SHORT_1)
                        ).to(exact_losses.dtype)
                        wait_mass = wait_weights.sum()
                        wait_count = wait_rows.sum()
                        long_count = long_rows.sum()
                        short_count = short_rows.sum()
                        wait_active = (wait_mass > 0).to(exact_losses.dtype)
                        long_active = (long_count > 0).to(exact_losses.dtype)
                        short_active = (short_count > 0).to(exact_losses.dtype)
                        regime_selectivity_exact_wait_loss = (
                            exact_losses * wait_weights
                        ).sum() / (
                            wait_count
                            if self.regime_selectivity_semantics
                            in {
                                EXPANSION_REGIME_CONFLUENCE_SEMANTICS,
                                SIDE_CONDITIONED_EXPANSION_REGIME_CONFLUENCE_SEMANTICS,
                            }
                            else wait_mass
                        ).clamp_min(1.0)
                        regime_selectivity_positive_long_loss = (
                            exact_losses * long_rows
                        ).sum() / long_count.clamp_min(1.0)
                        regime_selectivity_positive_short_loss = (
                            exact_losses * short_rows
                        ).sum() / short_count.clamp_min(1.0)
                        selectivity_targets = nn.functional.one_hot(
                            selected_actions,
                            num_classes=3,
                        ).to(model_log_probabilities.dtype)
                        selectivity_row_losses = exact_losses
                        model_wait_probability = model_log_probabilities[
                            :, int(Action.WAIT)
                        ].exp()
                        declared_side_probability = model_log_probabilities.gather(
                            -1, selected_actions[:, None]
                        ).squeeze(-1).exp()
                        dead_membership = (
                            persistent_evidence.persistent_dead_chop_membership
                        )
                        ready_membership = (
                            persistent_evidence.transition_ready_membership
                        )
                        ready_long_membership = (
                            persistent_evidence.transition_positive_long_membership
                        )
                        ready_short_membership = (
                            persistent_evidence.transition_positive_short_membership
                        )
                        failed_setup_confluence_membership = (
                            persistent_evidence.failed_setup_confluence_membership
                        )
                        failed_long_confluence_membership = (
                            persistent_evidence.failed_long_confluence_membership
                        )
                        failed_short_confluence_membership = (
                            persistent_evidence.failed_short_confluence_membership
                        )
                        association_group_active = torch.zeros_like(wait_active)
                        if (
                            self.regime_selectivity_semantics
                            in {
                                PERSISTENT_CHOP_ASSOCIATION_SEMANTICS,
                                EXPANSION_REGIME_CONFLUENCE_SEMANTICS,
                                SIDE_CONDITIONED_EXPANSION_REGIME_CONFLUENCE_SEMANTICS,
                            }
                        ):
                            (
                                regime_selectivity_association_loss,
                                association_group_active,
                            ) = persistent_chop_association_rank_loss(
                                flat_q,
                                dead_membership=dead_membership,
                                transition_positive_long_membership=(
                                    ready_long_membership
                                ),
                                transition_positive_short_membership=(
                                    ready_short_membership
                                ),
                                q_temperature=(
                                    self.regime_selectivity_q_temperature
                                ),
                            )
                            regime_selectivity_association_active = (
                                association_group_active
                            )
                            regime_selectivity_association_skipped = (
                                1.0 - association_group_active
                            )
                        side_conditioned_group_active = torch.zeros_like(wait_active)
                        if (
                            self.regime_selectivity_semantics
                            == SIDE_CONDITIONED_EXPANSION_REGIME_CONFLUENCE_SEMANTICS
                        ):
                            (
                                side_conditioned_loss_sum,
                                side_conditioned_group_active,
                            ) = side_conditioned_wait_rank_loss(
                                flat_q,
                                failed_long_membership=(
                                    failed_long_confluence_membership
                                ),
                                failed_short_membership=(
                                    failed_short_confluence_membership
                                ),
                                valid_long_membership=ready_long_membership,
                                valid_short_membership=ready_short_membership,
                                q_temperature=(
                                    self.regime_selectivity_q_temperature
                                ),
                            )
                            regime_selectivity_side_conditioned_loss = (
                                side_conditioned_loss_sum
                                / side_conditioned_group_active.clamp_min(1.0)
                            )
                            regime_selectivity_side_conditioned_active_sides = (
                                side_conditioned_group_active
                            )
                        chop_margin_membership = (
                            dead_membership
                            + failed_long_confluence_membership
                            + failed_short_confluence_membership
                        )
                        chop_margin_active = (
                            (chop_margin_membership.sum() > 0).to(exact_losses.dtype)
                            * float(
                                self.regime_selectivity_chop_wait_margin > 0.0
                                or self.regime_selectivity_failed_confluence_margin
                                > 0.0
                            )
                        )
                        chop_margin_loss = (
                            chop_specific_wait_margin_losses(
                                flat_q,
                                dominant_chop_membership=dead_membership,
                                failed_long_membership=(
                                    failed_long_confluence_membership
                                ),
                                failed_short_membership=(
                                    failed_short_confluence_membership
                                ),
                                chop_margin=(
                                    self.regime_selectivity_chop_wait_margin
                                ),
                                failed_confluence_margin=(
                                    self.regime_selectivity_failed_confluence_margin
                                ),
                            ).sum()
                            / chop_margin_membership.sum().clamp_min(1.0)
                        )
                        regime_selectivity_loss = (
                            regime_selectivity_exact_wait_loss * wait_active
                            + regime_selectivity_positive_long_loss * long_active
                            + regime_selectivity_positive_short_loss * short_active
                            + regime_selectivity_association_loss
                            + regime_selectivity_side_conditioned_loss
                            * side_conditioned_group_active
                        ) / (
                            wait_active
                            + long_active
                            + short_active
                            + association_group_active
                            + side_conditioned_group_active
                        ).clamp_min(1.0)
                        regime_selectivity_loss = (
                            regime_selectivity_loss
                            + chop_margin_loss * chop_margin_active
                        )
                        regime_persistent_additive.update({
                            "regime_selectivity_exact_wait_rows": (
                                wait_rows.sum()
                            ),
                            "regime_selectivity_exact_wait_weight_sum": (
                                wait_weights.sum()
                            ),
                            "regime_selectivity_exact_wait_"
                            "model_wait_probability_sum": (
                                model_wait_probability * wait_rows
                            ).sum(),
                            "regime_selectivity_persistent_chop_weight_sum": (
                                wait_weights.sum()
                            ),
                            "regime_selectivity_persistent_dead_chop_rows": (
                                dead_membership.sum()
                            ),
                            "regime_selectivity_persistent_dead_chop_"
                            "weight_sum": (
                                dead_membership * wait_weights
                            ).sum(),
                            "regime_selectivity_persistent_dead_chop_"
                            "model_wait_probability_sum": (
                                dead_membership * model_wait_probability
                            ).sum(),
                            "regime_selectivity_transition_ready_rows": (
                                ready_membership.sum()
                            ),
                            "regime_selectivity_transition_ready_weight_sum": (
                                ready_membership * wait_weights
                            ).sum(),
                            "regime_selectivity_transition_ready_"
                            "model_wait_probability_sum": (
                                ready_membership * model_wait_probability
                            ).sum(),
                            "regime_selectivity_failed_setup_confluence_rows": (
                                failed_setup_confluence_membership.sum()
                            ),
                            "regime_selectivity_failed_setup_confluence_"
                            "weight_sum": (
                                failed_setup_confluence_membership * wait_weights
                            ).sum(),
                            "regime_selectivity_failed_setup_confluence_"
                            "model_wait_probability_sum": (
                                failed_setup_confluence_membership
                                * model_wait_probability
                            ).sum(),
                            "regime_selectivity_failed_long_confluence_rows": (
                                failed_long_confluence_membership.sum()
                            ),
                            "regime_selectivity_failed_long_confluence_"
                            "model_wait_probability_sum": (
                                failed_long_confluence_membership
                                * model_wait_probability
                            ).sum(),
                            "regime_selectivity_failed_short_confluence_rows": (
                                failed_short_confluence_membership.sum()
                            ),
                            "regime_selectivity_failed_short_confluence_"
                            "model_wait_probability_sum": (
                                failed_short_confluence_membership
                                * model_wait_probability
                            ).sum(),
                            "regime_selectivity_transition_positive_long_rows": (
                                ready_long_membership.sum()
                            ),
                            "regime_selectivity_transition_positive_long_"
                            "declared_side_probability_sum": (
                                ready_long_membership * declared_side_probability
                            ).sum(),
                            "regime_selectivity_transition_positive_short_rows": (
                                ready_short_membership.sum()
                            ),
                            "regime_selectivity_transition_positive_short_"
                            "declared_side_probability_sum": (
                                ready_short_membership * declared_side_probability
                            ).sum(),
                            "regime_selectivity_association_dead_wait_rows": (
                                dead_membership.sum()
                            ),
                            "regime_selectivity_association_dead_wait_"
                            "model_wait_probability_sum": (
                                dead_membership * model_wait_probability
                            ).sum(),
                            "regime_selectivity_association_"
                            "transition_positive_long_rows": (
                                ready_long_membership.sum()
                            ),
                            "regime_selectivity_association_"
                            "transition_positive_long_"
                            "model_wait_probability_sum": (
                                ready_long_membership * model_wait_probability
                            ).sum(),
                            "regime_selectivity_association_"
                            "transition_positive_short_rows": (
                                ready_short_membership.sum()
                            ),
                            "regime_selectivity_association_"
                            "transition_positive_short_"
                            "model_wait_probability_sum": (
                                ready_short_membership * model_wait_probability
                            ).sum(),
                        })
                    else:
                        selectivity_targets = (
                            self.regime_selectivity.target_probabilities(
                                selected_teachers,
                                selected_headroom,
                                selected_actions,
                            )
                        )
                        selectivity_row_losses = -(
                            selectivity_targets * model_log_probabilities
                        ).sum(-1)
                    if (
                        self.regime_selectivity_semantics == STATIC_STATE_SEMANTICS
                        and self.regime_selectivity_side_balance
                        == "equal_long_short_v1"
                    ):
                        long_rows = (
                            selected_actions == int(Action.ENTER_LONG_1)
                        ).to(selectivity_row_losses.dtype)
                        short_rows = (
                            selected_actions == int(Action.ENTER_SHORT_1)
                        ).to(selectivity_row_losses.dtype)
                        long_count = long_rows.sum()
                        short_count = short_rows.sum()
                        long_active = (long_count > 0).to(long_count.dtype)
                        short_active = (short_count > 0).to(short_count.dtype)
                        regime_selectivity_loss = (
                            (
                                (selectivity_row_losses * long_rows).sum()
                                / long_count.clamp_min(1.0)
                            )
                            * long_active
                            + (
                                (selectivity_row_losses * short_rows).sum()
                                / short_count.clamp_min(1.0)
                            )
                            * short_active
                        ) / (long_active + short_active).clamp_min(1.0)
                    elif self.regime_selectivity_semantics == STATIC_STATE_SEMANTICS:
                        regime_selectivity_loss = (
                            selectivity_row_losses.sum()
                            / positive_rows.clamp_min(1.0)
                        )
                    long_loss_rows = (
                        selected_actions == int(Action.ENTER_LONG_1)
                    ).to(selectivity_row_losses.dtype)
                    short_loss_rows = (
                        selected_actions == int(Action.ENTER_SHORT_1)
                    ).to(selectivity_row_losses.dtype)
                    if self.regime_selectivity_semantics == STATIC_STATE_SEMANTICS:
                        regime_selectivity_positive_long_loss = (
                            selectivity_row_losses * long_loss_rows
                        ).sum() / long_loss_rows.sum().clamp_min(1.0)
                        regime_selectivity_positive_short_loss = (
                            selectivity_row_losses * short_loss_rows
                        ).sum() / short_loss_rows.sum().clamp_min(1.0)
                    loss = loss + entry_action_weight_scale * (
                        self.regime_selectivity_loss_weight
                        * regime_selectivity_loss
                    )

                    target_wait = selectivity_targets[:, int(Action.WAIT)]
                    model_probabilities = model_log_probabilities.exp()
                    model_wait = model_probabilities[:, int(Action.WAIT)]
                    target_action_probability = selectivity_targets.gather(
                        -1, selected_actions[:, None]
                    ).squeeze(-1)
                    model_target_action_probability = model_probabilities.gather(
                        -1, selected_actions[:, None]
                    ).squeeze(-1)
                    declared_side = model_probabilities.gather(
                        -1, selected_actions[:, None]
                    ).squeeze(-1)
                    greedy_actions = model_probabilities.argmax(-1)
                    greedy_wait = greedy_actions == int(Action.WAIT)
                    greedy_entry = greedy_actions == selected_actions
                    for side, action_index in (
                        ("long", int(Action.ENTER_LONG_1)),
                        ("short", int(Action.ENTER_SHORT_1)),
                    ):
                        side_rows = (
                            selected_actions == action_index
                        ).to(model_wait.dtype)
                        prefix = f"regime_entry_conflict_{side}_"
                        regime_entry_conflict_additive.update({
                            prefix + "rows": side_rows.sum(),
                            prefix + "target_wait_probability_sum": (
                                target_wait * side_rows
                            ).sum(),
                            prefix + "target_declared_side_probability_sum": (
                                target_action_probability * side_rows
                            ).sum(),
                            prefix + "model_wait_probability_sum": (
                                model_wait * side_rows
                            ).sum(),
                            prefix + "soft_wait_disagreement_rows": (
                                (
                                    target_wait
                                    > target_action_probability
                                ).to(model_wait.dtype) * side_rows
                            ).sum(),
                        })
                    names = self.teacher_channel_names
                    chop = selected_teachers[
                        :, names.index("chop_no_trend_probability")
                    ]
                    chop_end_transition = selected_teachers[
                        :, names.index("chop_end_transition_probability")
                    ]
                    expansion_trend = selected_teachers[
                        :, names.index("expansion_trend_probability")
                    ]
                    dominant_chop = chop > torch.maximum(
                        chop_end_transition, expansion_trend
                    )
                    stratum_rows = {
                        "positive_long_short": torch.ones_like(
                            selected_actions, dtype=torch.bool
                        ) & (
                            selected_actions != int(Action.WAIT)
                        ),
                        "positive_long": (
                            selected_actions == int(Action.ENTER_LONG_1)
                        ),
                        "positive_short": (
                            selected_actions == int(Action.ENTER_SHORT_1)
                        ),
                        "dominant_chop": dominant_chop,
                        "nonchop": ~dominant_chop,
                        "low_headroom_le_0_25": selected_headroom <= 0.25,
                        "mid_headroom_gt_0_25_lt_0_75": (
                            (selected_headroom > 0.25)
                            & (selected_headroom < 0.75)
                        ),
                        "safe_headroom_ge_0_75": selected_headroom >= 0.75,
                    }
                    stratum_weights = torch.stack(
                        tuple(stratum_rows.values()), dim=-1
                    ).to(torch.float32)
                    row_diagnostics = torch.stack((
                        torch.ones_like(target_wait),
                        target_wait,
                        model_wait,
                        (model_wait - target_wait).abs(),
                        target_action_probability,
                        model_target_action_probability,
                        (
                            model_target_action_probability
                            - target_action_probability
                        ).abs(),
                        greedy_wait.to(torch.float32),
                        declared_side,
                        greedy_entry.to(torch.float32),
                        (greedy_actions == selected_actions).to(torch.float32),
                        *(
                            (
                                (selected_actions == target_index)
                                & (greedy_actions == prediction_index)
                            ).to(torch.float32)
                            for target_index in range(3)
                            for prediction_index in range(3)
                        ),
                    ), dim=-1)
                    additive_matrix = stratum_weights.transpose(0, 1).matmul(
                        row_diagnostics
                    )
                    regime_selectivity_additive = {
                        f"regime_selectivity_{stratum}_{field}": (
                            additive_matrix[stratum_index, field_index]
                        )
                        for stratum_index, stratum in enumerate(stratum_rows)
                        for field_index, field in enumerate(
                            _REGIME_SELECTIVITY_ADDITIVE_FIELDS
                        )
                    }

                    positive_side_rows = (
                        selected_actions != int(Action.WAIT)
                    ).sum().to(torch.float32)
                    regime_selectivity_rows = positive_rows
                    regime_selectivity_target_wait_mean = (
                        regime_selectivity_additive[
                            "regime_selectivity_positive_long_short_"
                            "target_wait_probability_sum"
                        ]
                        / positive_side_rows.clamp_min(1.0)
                    )
                    regime_selectivity_low_headroom_rows = (
                        regime_selectivity_additive[
                            "regime_selectivity_low_headroom_le_0_25_rows"
                        ]
                    )
                    regime_selectivity_low_headroom_wait_mean = (
                        regime_selectivity_additive[
                            "regime_selectivity_low_headroom_le_0_25_"
                            "target_wait_probability_sum"
                        ]
                        / regime_selectivity_low_headroom_rows.clamp_min(1.0)
                    )
                    regime_selectivity_dominant_chop_rows = (
                        regime_selectivity_additive[
                            "regime_selectivity_dominant_chop_rows"
                        ]
                    )
                    regime_selectivity_dominant_chop_wait_mean = (
                        regime_selectivity_additive[
                            "regime_selectivity_dominant_chop_"
                            "target_wait_probability_sum"
                        ]
                        / regime_selectivity_dominant_chop_rows.clamp_min(1.0)
                    )
        if self.entry_action_loss_weight and entry_action_weight_scale > 0.0:
            assert diagnostic_targets is not None
            assert diagnostic_target_rows is not None
            timing_targets = diagnostic_targets
            flat_rows = (
                valid_masks[
                    :,
                    self.recurrent_burn_in:
                    self.recurrent_burn_in + training_steps,
                    int(Action.WAIT),
                ]
                & valid_masks[
                    :,
                    self.recurrent_burn_in:
                    self.recurrent_burn_in + training_steps,
                    int(Action.ENTER_LONG_1),
                ]
                & valid_masks[
                    :,
                    self.recurrent_burn_in:
                    self.recurrent_burn_in + training_steps,
                    int(Action.ENTER_SHORT_1),
                ]
            )
            timing_rows = diagnostic_target_rows
            if bool((timing_rows & ~flat_rows).any().item()):
                raise ValueError("entry timing target requires a flat decision")
            if bool(timing_rows.any().item()):
                q_values = (all_logits.float().softmax(-1) * self.support).sum(-1)
                entry_q = torch.stack(
                    tuple(
                        q_values[:, :training_steps, int(action)]
                        for action in (
                            Action.WAIT,
                            Action.ENTER_LONG_1,
                            Action.ENTER_SHORT_1,
                        )
                    ),
                    dim=-1,
                )
                selected_entry_q = entry_q[timing_rows]
                selected_timing_targets = timing_targets[timing_rows]
                unweighted_entry_ce = nn.functional.cross_entropy(
                    selected_entry_q,
                    selected_timing_targets,
                    reduction="none",
                )
                unweighted_margin = exact_action_margin_losses(
                    selected_entry_q,
                    selected_timing_targets,
                    margin=self.entry_action_margin,
                )
                selected_class_counts = torch.bincount(
                    selected_timing_targets, minlength=3
                ).to(unweighted_entry_ce.dtype)
                present_classes = selected_class_counts > 0
                if (
                    self.entry_action_loss_reduction
                    == "equal_present_class_mean_v1"
                ):
                    class_ce_means = torch.stack(tuple(
                        unweighted_entry_ce[
                            selected_timing_targets == class_index
                        ].sum() / selected_class_counts[class_index].clamp_min(1.0)
                        for class_index in range(3)
                    ))
                    entry_action_loss = class_ce_means[present_classes].mean()
                    class_margin_means = torch.stack(tuple(
                        unweighted_margin[
                            selected_timing_targets == class_index
                        ].sum() / selected_class_counts[class_index].clamp_min(1.0)
                        for class_index in range(3)
                    ))
                    entry_action_margin_loss = class_margin_means[
                        present_classes
                    ].mean()
                else:
                    entry_action_loss = nn.functional.cross_entropy(
                        selected_entry_q,
                        selected_timing_targets,
                        weight=self._entry_action_class_weights_tensor,
                    )
                    selected_class_weights = self._entry_action_class_weights_tensor[
                        selected_timing_targets
                    ]
                    entry_action_margin_loss = (
                        unweighted_margin * selected_class_weights
                    ).sum() / selected_class_weights.sum()
                entry_action_supervised_rows = timing_rows.sum().to(
                    dtype=torch.float32
                )
                selected_targets = selected_timing_targets
                selected_predictions = selected_entry_q.argmax(-1)
                entry_action_target_counts = torch.bincount(
                    selected_targets, minlength=3
                ).to(torch.float32)
                entry_action_prediction_counts = torch.bincount(
                    selected_predictions, minlength=3
                ).to(torch.float32)
                for class_index in range(3):
                    action_name = _ENTRY_BALANCE_ACTION_NAMES[class_index]
                    class_rows = (
                        selected_targets == class_index
                    ).to(unweighted_entry_ce.dtype)
                    class_weight = self._entry_action_class_weights_tensor[
                        class_index
                    ]
                    if (
                        self.entry_action_loss_reduction
                        == "equal_present_class_mean_v1"
                    ):
                        effective_row_weight = torch.where(
                            present_classes[class_index],
                            1.0
                            / (
                                present_classes.sum().to(unweighted_entry_ce.dtype)
                                * selected_class_counts[class_index].clamp_min(1.0)
                            ),
                            torch.zeros((), device=self.device),
                        )
                    else:
                        effective_row_weight = class_weight
                    prefix = f"entry_balance_{action_name}_"
                    entry_balance_additive.update({
                        prefix + "rows": class_rows.sum(),
                        prefix + "weighted_mass": (
                            class_rows.sum() * effective_row_weight
                        ),
                        prefix + "unweighted_ce_sum": (
                            unweighted_entry_ce * class_rows
                        ).sum(),
                        prefix + "weighted_ce_sum": (
                            unweighted_entry_ce
                            * class_rows
                            * effective_row_weight
                        ).sum(),
                    })
                    entry_action_correct_counts[class_index] = (
                        (selected_targets == class_index)
                        & (selected_predictions == class_index)
                    ).sum()
                loss = loss + (
                    entry_action_weight_scale
                    * self.entry_action_loss_weight
                    * (entry_action_loss + entry_action_margin_loss)
                )
        management_valid_rows = auxiliary_valid & (
            valid_masks[:, self.recurrent_burn_in:, int(Action.HOLD)]
            & valid_masks[:, self.recurrent_burn_in:, int(Action.CLOSE)]
            & (valid_masks[:, self.recurrent_burn_in:].sum(-1) == 2)
        )
        retention_rows = management_valid_rows
        if not self.retention_anchor_applies_to_all_management_rows:
            retention_rows = (
                competence_anchors[:, self.recurrent_burn_in:]
                & retention_rows
            )
        if (
            self.retention_anchor is not None
            and self.policy_retention_loss_weight
            and bool(retention_rows.any().item())
        ):
            with torch.no_grad():
                anchor_hidden = None
                if self.recurrent_burn_in:
                    _, anchor_hidden = self._recurrent_features_with_resets(
                        self.retention_anchor,
                        causal_observations[:, :self.recurrent_burn_in],
                        burn_in_reset_rows,
                    )
                anchor_recurrent, _ = self._recurrent_features_with_resets(
                    self.retention_anchor,
                    causal_observations[:, self.recurrent_burn_in:],
                    learning_reset_rows,
                    anchor_hidden,
                )
                anchor_logits = self.retention_anchor.distribution_logits(
                    anchor_recurrent[:, :-1]
                ).float()
                anchor_q = (anchor_logits.softmax(-1) * self.support).sum(-1)
                anchor_management_q = torch.stack((
                    anchor_q[..., int(Action.HOLD)],
                    anchor_q[..., int(Action.CLOSE)],
                ), dim=-1)
            current_q = (all_logits.float().softmax(-1) * self.support).sum(-1)
            current_management_q = torch.stack((
                current_q[..., int(Action.HOLD)],
                current_q[..., int(Action.CLOSE)],
            ), dim=-1)
            policy_retention_loss = nn.functional.smooth_l1_loss(
                current_management_q[retention_rows],
                anchor_management_q[retention_rows],
            )
            loss = loss + (
                self.policy_retention_loss_weight * policy_retention_loss
            )
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        try:
            gradient_norm = nn.utils.clip_grad_norm_(
                self.online.parameters(),
                max_norm=self.gradient_clip,
                error_if_nonfinite=True,
            )
        except RuntimeError as error:
            if "non-finite" not in str(error):
                raise
            self.optimizer.zero_grad(set_to_none=True)
            raise ValueError("training loss is non-finite") from error
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self._updates += 1
        self._update_target_network()
        q_values = (logits.softmax(-1) * self.support).sum(-1)
        rl_valid_masks = valid_masks[
            :, learning_start:learning_start + training_steps
        ]
        management_rows = learnable_rows & (
            rl_valid_masks[..., int(Action.HOLD)]
            & rl_valid_masks[..., int(Action.CLOSE)]
            & (rl_valid_masks.sum(-1) == 2)
        )
        hold_rows = learnable_rows & (actions == int(Action.HOLD))
        close_rows = learnable_rows & (actions == int(Action.CLOSE))

        def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            weights = mask.to(values.dtype)
            return (values * weights).sum() / weights.sum().clamp_min(1.0)

        real_reset_rows = training_valid & torch.as_tensor(
            recurrent_reset_rows, dtype=torch.bool, device=self.device
        )
        valid_row_count = training_valid.sum().to(torch.float32)
        burn_in_reset_coverage = torch.as_tensor(
            sum(
                any(
                    transition.training_valid and transition.recurrent_reset
                    for transition in sequence
                )
                for sequence in sequences
            ) / len(sequences),
            dtype=torch.float32,
            device=self.device,
        )
        terminal_truncated_rows = learnable_rows & terminal_targets

        diagnostic_values = torch.stack((
            rl_loss,
            teacher_loss,
            entry_search_loss,
            entry_action_loss,
            entry_action_margin_loss,
            regime_selectivity_loss,
            regime_selectivity_rows,
            regime_selectivity_target_wait_mean,
            regime_selectivity_low_headroom_rows,
            regime_selectivity_low_headroom_wait_mean,
            regime_selectivity_dominant_chop_rows,
            regime_selectivity_dominant_chop_wait_mean,
            regime_selectivity_positive_long_loss,
            regime_selectivity_positive_short_loss,
            regime_selectivity_exact_wait_loss,
            regime_selectivity_association_loss,
            regime_selectivity_association_active,
            regime_selectivity_association_skipped,
            regime_selectivity_side_conditioned_loss,
            regime_selectivity_side_conditioned_active_sides,
            entry_action_supervised_rows,
            *entry_action_target_counts.unbind(),
            *entry_action_prediction_counts.unbind(),
            *entry_action_correct_counts.unbind(),
            policy_retention_loss,
            loss,
            gradient_norm.float(),
            management_rows.sum().to(torch.float32)
            / learnable_rows.sum().to(torch.float32),
            masked_mean(immediate_rewards, hold_rows),
            masked_mean(immediate_rewards, close_rows),
            masked_mean(n_step_rewards, hold_rows),
            masked_mean(n_step_rewards, close_rows),
            masked_mean(td_losses, hold_rows),
            masked_mean(td_losses, close_rows),
            masked_mean(
                q_values[..., int(Action.HOLD)]
                - q_values[..., int(Action.CLOSE)],
                management_rows,
            ),
            masked_mean(close_rows.float(), management_rows),
            real_reset_rows.sum().to(torch.float32) / valid_row_count,
            burn_in_reset_coverage,
            learnable_rows.sum().to(torch.float32),
            (~training_valid).sum().to(torch.float32),
            terminal_truncated_rows.sum().to(torch.float32),
            *regime_selectivity_additive.values(),
            *regime_channel_additive.flatten().unbind(),
            *entry_balance_additive.values(),
            *regime_entry_conflict_additive.values(),
            *regime_persistent_additive.values(),
        ))
        (
            rl_loss_value,
            teacher_loss_value,
            entry_search_loss_value,
            entry_action_loss_value,
            entry_action_margin_loss_value,
            regime_selectivity_loss_value,
            regime_selectivity_rows_value,
            regime_selectivity_target_wait_mean_value,
            regime_selectivity_low_headroom_rows_value,
            regime_selectivity_low_headroom_wait_mean_value,
            regime_selectivity_dominant_chop_rows_value,
            regime_selectivity_dominant_chop_wait_mean_value,
            regime_selectivity_positive_long_loss_value,
            regime_selectivity_positive_short_loss_value,
            regime_selectivity_exact_wait_loss_value,
            regime_selectivity_association_loss_value,
            regime_selectivity_association_active_value,
            regime_selectivity_association_skipped_value,
            regime_selectivity_side_conditioned_loss_value,
            regime_selectivity_side_conditioned_active_sides_value,
            entry_action_supervised_rows_value,
            entry_target_wait_rows,
            entry_target_long_rows,
            entry_target_short_rows,
            entry_prediction_wait_rows,
            entry_prediction_long_rows,
            entry_prediction_short_rows,
            entry_correct_wait_rows,
            entry_correct_long_rows,
            entry_correct_short_rows,
            policy_retention_loss_value,
            total_loss,
            gradient_norm_value,
            management_row_fraction,
            sampled_hold_reward,
            sampled_close_reward,
            sampled_hold_n_step_return,
            sampled_close_n_step_return,
            sampled_hold_td_loss,
            sampled_close_td_loss,
            management_hold_minus_close_q,
            sampled_management_close_fraction,
            sampled_recurrent_reset_fraction,
            sampled_burn_in_reset_coverage,
            sampled_valid_learning_rows,
            sampled_padding_rows,
            sampled_terminal_truncated_rows,
            *all_regime_additive_values,
        ) = (
            float(value)
            for value in diagnostic_values.detach().float().cpu().tolist()
        )
        selectivity_additive_count = len(regime_selectivity_additive)
        regime_selectivity_additive_values = all_regime_additive_values[
            :selectivity_additive_count
        ]
        regime_channel_additive_count = (
            len(regime_channel_names) * len(_REGIME_CHANNEL_ADDITIVE_FIELDS)
        )
        regime_channel_additive_values = all_regime_additive_values[
            selectivity_additive_count:
            selectivity_additive_count + regime_channel_additive_count
        ]
        entry_balance_start = (
            selectivity_additive_count + regime_channel_additive_count
        )
        entry_balance_count = len(entry_balance_additive)
        entry_balance_values = all_regime_additive_values[
            entry_balance_start:entry_balance_start + entry_balance_count
        ]
        regime_conflict_start = entry_balance_start + entry_balance_count
        regime_conflict_count = len(regime_entry_conflict_additive)
        regime_conflict_values = all_regime_additive_values[
            regime_conflict_start:regime_conflict_start + regime_conflict_count
        ]
        regime_persistent_additive_values = all_regime_additive_values[
            regime_conflict_start + regime_conflict_count:
        ]
        regime_selectivity_metric_values = {
            f"regime_selectivity_{stratum}_{field}": 0.0
            for stratum in _REGIME_SELECTIVITY_STRATA
            for field in _REGIME_SELECTIVITY_ADDITIVE_FIELDS
        }
        regime_selectivity_metric_values.update(dict(zip(
            regime_selectivity_additive,
            regime_selectivity_additive_values,
            strict=True,
        )))
        regime_channel_metric_values = {
            f"regime_teacher_channel_{channel}_{field}": (
                regime_channel_additive_values[
                    channel_index * len(_REGIME_CHANNEL_ADDITIVE_FIELDS)
                    + field_index
                ]
            )
            for channel_index, channel in enumerate(regime_channel_names)
            for field_index, field in enumerate(
                _REGIME_CHANNEL_ADDITIVE_FIELDS
            )
        }
        for channel in regime_channel_names:
            prefix = f"regime_teacher_channel_{channel}_"
            rows = regime_channel_metric_values[prefix + "rows"]
            for total_field, mean_field in (
                ("target_probability_sum", "target_probability_mean"),
                ("model_probability_sum", "model_probability_mean"),
                ("absolute_error_sum", "mean_absolute_error"),
                ("squared_error_sum", "mean_squared_error"),
            ):
                regime_channel_metric_values[prefix + mean_field] = (
                    regime_channel_metric_values[prefix + total_field] / rows
                    if rows else 0.0
                )
        entry_balance_metric_values = dict(zip(
            entry_balance_additive,
            entry_balance_values,
            strict=True,
        ))
        total_weighted_mass = sum(
            entry_balance_metric_values[
                f"entry_balance_{action}_weighted_mass"
            ]
            for action in _ENTRY_BALANCE_ACTION_NAMES
        )
        total_weighted_ce = sum(
            entry_balance_metric_values[
                f"entry_balance_{action}_weighted_ce_sum"
            ]
            for action in _ENTRY_BALANCE_ACTION_NAMES
        )
        for class_index, action in enumerate(_ENTRY_BALANCE_ACTION_NAMES):
            prefix = f"entry_balance_{action}_"
            rows = entry_balance_metric_values[prefix + "rows"]
            entry_balance_metric_values[prefix + "configured_weight"] = float(
                self.entry_action_class_weights[class_index]
            )
            entry_balance_metric_values[prefix + "weighted_mass_fraction"] = (
                entry_balance_metric_values[prefix + "weighted_mass"]
                / total_weighted_mass if total_weighted_mass else 0.0
            )
            entry_balance_metric_values[prefix + "unweighted_ce_mean"] = (
                entry_balance_metric_values[prefix + "unweighted_ce_sum"]
                / rows if rows else 0.0
            )
            entry_balance_metric_values[
                prefix + "weighted_loss_contribution"
            ] = (
                entry_balance_metric_values[prefix + "weighted_ce_sum"]
                / total_weighted_mass if total_weighted_mass else 0.0
            )
            entry_balance_metric_values[prefix + "weighted_ce_fraction"] = (
                entry_balance_metric_values[prefix + "weighted_ce_sum"]
                / total_weighted_ce if total_weighted_ce else 0.0
            )
        regime_entry_conflict_metric_values = dict(zip(
            regime_entry_conflict_additive,
            regime_conflict_values,
            strict=True,
        ))
        for side in ("long", "short"):
            prefix = f"regime_entry_conflict_{side}_"
            rows = regime_entry_conflict_metric_values[prefix + "rows"]
            for total_field, mean_field in (
                ("target_wait_probability_sum", "target_wait_probability_mean"),
                (
                    "target_declared_side_probability_sum",
                    "target_declared_side_probability_mean",
                ),
                ("model_wait_probability_sum", "model_wait_probability_mean"),
                ("soft_wait_disagreement_rows", "soft_wait_disagreement_rate"),
            ):
                regime_entry_conflict_metric_values[prefix + mean_field] = (
                    regime_entry_conflict_metric_values[prefix + total_field]
                    / rows if rows else 0.0
                )
        regime_persistent_metric_values = dict(zip(
            regime_persistent_additive,
            regime_persistent_additive_values,
            strict=True,
        ))
        exact_wait_rows = regime_persistent_metric_values[
            "regime_selectivity_exact_wait_rows"
        ]
        dead_chop_rows = regime_persistent_metric_values[
            "regime_selectivity_persistent_dead_chop_rows"
        ]
        transition_ready_rows = regime_persistent_metric_values[
            "regime_selectivity_transition_ready_rows"
        ]
        failed_setup_confluence_rows = regime_persistent_metric_values[
            "regime_selectivity_failed_setup_confluence_rows"
        ]
        failed_long_confluence_rows = regime_persistent_metric_values[
            "regime_selectivity_failed_long_confluence_rows"
        ]
        failed_short_confluence_rows = regime_persistent_metric_values[
            "regime_selectivity_failed_short_confluence_rows"
        ]
        for rows, sum_name, mean_name in (
            (
                exact_wait_rows,
                "regime_selectivity_exact_wait_weight_sum",
                "regime_selectivity_exact_wait_weight_mean",
            ),
            (
                exact_wait_rows,
                "regime_selectivity_exact_wait_model_wait_probability_sum",
                "regime_selectivity_exact_wait_model_wait_probability_mean",
            ),
            (
                exact_wait_rows,
                "regime_selectivity_persistent_chop_weight_sum",
                "regime_selectivity_persistent_chop_weight_mean",
            ),
            (
                dead_chop_rows,
                "regime_selectivity_persistent_dead_chop_weight_sum",
                "regime_selectivity_persistent_dead_chop_weight_mean",
            ),
            (
                dead_chop_rows,
                "regime_selectivity_persistent_dead_chop_"
                "model_wait_probability_sum",
                "regime_selectivity_persistent_dead_chop_"
                "model_wait_probability_mean",
            ),
            (
                transition_ready_rows,
                "regime_selectivity_transition_ready_weight_sum",
                "regime_selectivity_transition_ready_weight_mean",
            ),
            (
                transition_ready_rows,
                "regime_selectivity_transition_ready_"
                "model_wait_probability_sum",
                "regime_selectivity_transition_ready_"
                "model_wait_probability_mean",
            ),
            (
                failed_setup_confluence_rows,
                "regime_selectivity_failed_setup_confluence_weight_sum",
                "regime_selectivity_failed_setup_confluence_weight_mean",
            ),
            (
                failed_setup_confluence_rows,
                "regime_selectivity_failed_setup_confluence_"
                "model_wait_probability_sum",
                "regime_selectivity_failed_setup_confluence_"
                "model_wait_probability_mean",
            ),
            (
                failed_long_confluence_rows,
                "regime_selectivity_failed_long_confluence_"
                "model_wait_probability_sum",
                "regime_selectivity_failed_long_confluence_"
                "model_wait_probability_mean",
            ),
            (
                failed_short_confluence_rows,
                "regime_selectivity_failed_short_confluence_"
                "model_wait_probability_sum",
                "regime_selectivity_failed_short_confluence_"
                "model_wait_probability_mean",
            ),
            (
                regime_persistent_metric_values[
                    "regime_selectivity_transition_positive_long_rows"
                ],
                "regime_selectivity_transition_positive_long_"
                "declared_side_probability_sum",
                "regime_selectivity_transition_positive_long_"
                "declared_side_probability_mean",
            ),
            (
                regime_persistent_metric_values[
                    "regime_selectivity_transition_positive_short_rows"
                ],
                "regime_selectivity_transition_positive_short_"
                "declared_side_probability_sum",
                "regime_selectivity_transition_positive_short_"
                "declared_side_probability_mean",
            ),
        ):
            regime_persistent_metric_values[mean_name] = (
                regime_persistent_metric_values[sum_name] / rows
                if rows
                else 0.0
            )
        association_dead_rows = regime_persistent_metric_values[
            "regime_selectivity_association_dead_wait_rows"
        ]
        association_dead_wait = (
            regime_persistent_metric_values[
                "regime_selectivity_association_dead_wait_"
                "model_wait_probability_sum"
            ]
            / association_dead_rows
            if association_dead_rows
            else 0.0
        )
        association_ready_means = []
        for side in ("long", "short"):
            prefix = (
                "regime_selectivity_association_transition_positive_"
                f"{side}_"
            )
            rows = regime_persistent_metric_values[prefix + "rows"]
            if rows:
                association_ready_means.append(
                    regime_persistent_metric_values[
                        prefix + "model_wait_probability_sum"
                    ] / rows
                )
        regime_persistent_metric_values[
            "regime_selectivity_dead_wait_minus_"
            "transition_positive_model_wait"
        ] = (
            association_dead_wait
            - sum(association_ready_means) / len(association_ready_means)
            if association_dead_rows and association_ready_means
            else 0.0
        )
        for stratum in _REGIME_SELECTIVITY_STRATA:
            prefix = f"regime_selectivity_{stratum}_"
            rows = regime_selectivity_metric_values[prefix + "rows"]
            for total_field, derived_field in (
                (
                    "target_wait_probability_sum",
                    "target_wait_probability_mean",
                ),
                (
                    "model_wait_probability_sum",
                    "model_wait_probability_mean",
                ),
                (
                    "wait_absolute_error_sum",
                    "wait_mean_absolute_error",
                ),
                (
                    "target_action_probability_sum",
                    "target_action_probability_mean",
                ),
                (
                    "model_target_action_probability_sum",
                    "model_target_action_probability_mean",
                ),
                (
                    "target_action_absolute_error_sum",
                    "target_action_mean_absolute_error",
                ),
                (
                    "greedy_wait_rows",
                    "greedy_wait_rate",
                ),
                (
                    "declared_side_probability_sum",
                    "declared_side_probability_mean",
                ),
                (
                    "greedy_entry_rows",
                    "greedy_entry_rate",
                ),
                (
                    "correct_rows",
                    "accuracy",
                ),
            ):
                regime_selectivity_metric_values[prefix + derived_field] = (
                    regime_selectivity_metric_values[prefix + total_field] / rows
                    if rows else 0.0
                )
        self.last_train_metrics = {
            "rl_loss": rl_loss_value,
            "teacher_loss": teacher_loss_value,
            "entry_search_loss": entry_search_loss_value,
            "entry_action_loss": entry_action_loss_value,
            "entry_action_margin_loss": entry_action_margin_loss_value,
            "regime_selectivity_loss": regime_selectivity_loss_value,
            "regime_selectivity_supervised_rows": regime_selectivity_rows_value,
            "regime_selectivity_target_wait_mean": (
                regime_selectivity_target_wait_mean_value
            ),
            "regime_selectivity_low_headroom_rows": (
                regime_selectivity_low_headroom_rows_value
            ),
            "regime_selectivity_low_headroom_wait_mean": (
                regime_selectivity_low_headroom_wait_mean_value
            ),
            "regime_selectivity_dominant_chop_rows": (
                regime_selectivity_dominant_chop_rows_value
            ),
            "regime_selectivity_dominant_chop_wait_mean": (
                regime_selectivity_dominant_chop_wait_mean_value
            ),
            "regime_selectivity_positive_long_loss": (
                regime_selectivity_positive_long_loss_value
            ),
            "regime_selectivity_positive_short_loss": (
                regime_selectivity_positive_short_loss_value
            ),
            "regime_selectivity_exact_wait_loss": (
                regime_selectivity_exact_wait_loss_value
            ),
            "regime_selectivity_association_loss": (
                regime_selectivity_association_loss_value
            ),
            "regime_selectivity_association_active": (
                regime_selectivity_association_active_value
            ),
            "regime_selectivity_association_skipped": (
                regime_selectivity_association_skipped_value
            ),
            "regime_selectivity_side_conditioned_loss": (
                regime_selectivity_side_conditioned_loss_value
            ),
            "regime_selectivity_side_conditioned_active_sides": (
                regime_selectivity_side_conditioned_active_sides_value
            ),
            **regime_selectivity_metric_values,
            **regime_channel_metric_values,
            **entry_balance_metric_values,
            **regime_entry_conflict_metric_values,
            **regime_persistent_metric_values,
            "entry_action_supervised_rows": entry_action_supervised_rows_value,
            "entry_action_target_wait_rows": entry_target_wait_rows,
            "entry_action_target_long_rows": entry_target_long_rows,
            "entry_action_target_short_rows": entry_target_short_rows,
            "entry_action_prediction_wait_rows": entry_prediction_wait_rows,
            "entry_action_prediction_long_rows": entry_prediction_long_rows,
            "entry_action_prediction_short_rows": entry_prediction_short_rows,
            "entry_action_correct_wait_rows": entry_correct_wait_rows,
            "entry_action_correct_long_rows": entry_correct_long_rows,
            "entry_action_correct_short_rows": entry_correct_short_rows,
            "policy_retention_loss": policy_retention_loss_value,
            "teacher_weight_scale": teacher_weight_scale,
            "entry_action_weight_scale": entry_action_weight_scale,
            "total_loss": total_loss,
            "gradient_norm": gradient_norm_value,
            "sampled_management_row_fraction": management_row_fraction,
            "sampled_hold_reward": sampled_hold_reward,
            "sampled_close_reward": sampled_close_reward,
            "sampled_hold_n_step_return": sampled_hold_n_step_return,
            "sampled_close_n_step_return": sampled_close_n_step_return,
            "sampled_hold_td_loss": sampled_hold_td_loss,
            "sampled_close_td_loss": sampled_close_td_loss,
            "management_hold_minus_close_q": management_hold_minus_close_q,
            "sampled_management_close_fraction": (
                sampled_management_close_fraction
            ),
            "sampled_recurrent_reset_fraction": (
                sampled_recurrent_reset_fraction
            ),
            "sampled_burn_in_reset_coverage": sampled_burn_in_reset_coverage,
            "sampled_valid_learning_rows": sampled_valid_learning_rows,
            "sampled_padding_rows": sampled_padding_rows,
            "sampled_terminal_truncated_rows": sampled_terminal_truncated_rows,
            "sampled_recurrent_reset_pattern_count": float(
                len({
                    tuple(
                        item.recurrent_reset
                        for item in sequence
                        if item.training_valid
                    )
                    for sequence in sequences
                })
            ),
            "n_step_return": float(self.n_step_return),
            "recurrent_burn_in": float(self.recurrent_burn_in),
        }
        return total_loss

    def retain_policy(
        self,
        *,
        apply_to_all_management_rows: bool = False,
    ) -> None:
        """Freeze a training-only copy of demonstrated pass competence."""
        if (
            self.retention_anchor is not None
            and self.retention_anchor_applies_to_all_management_rows
            and not apply_to_all_management_rows
        ):
            # A warm-start parent is the immutable Stage-1 competence boundary.
            # Later pass evidence may be checkpointed independently, but must
            # never replace or narrow the parent protection scope.
            return
        anchor = RecurrentC51Network(
            self.observation_dim,
            len(Action),
            self.atoms,
            self.hidden_dim,
        ).to(self.device)
        anchor.load_state_dict({
            key: value
            for key, value in self.online.state_dict().items()
            if not key.startswith("teacher_output.")
        })
        self.retention_anchor = anchor.eval()
        self.retention_anchor.requires_grad_(False)
        self.retention_anchor_applies_to_all_management_rows = bool(
            apply_to_all_management_rows
        )

    def discard_retention_anchor(self) -> None:
        """Remove training-only competence state before evaluation or shipping."""
        self.retention_anchor = None
        self.retention_anchor_applies_to_all_management_rows = False

    def discard_teacher(self) -> None:
        """Remove the training-only head while retaining shared learned weights."""
        self.entry_action_loss_weight = 0.0
        self.teacher_loss_weight = 0.0
        self.teacher_entry_search_loss_weight = 0.0
        self.teacher_entry_search_objective = "raw_probability"
        self.teacher_entry_search_centers = (0.5, 0.5)
        self.regime_selectivity_loss_weight = 0.0
        self.regime_selectivity_side_balance = "none"
        self.regime_selectivity_semantics = STATIC_STATE_SEMANTICS
        self.regime_selectivity_persistent_chop_negative_emphasis = 0.0
        self.regime_selectivity = None
        self.last_train_metrics = {}
        self.teacher_channel_names = ()
        self.teacher_channel_loss_weights = ()
        self._teacher_channel_loss_weights_tensor = torch.empty(
            0,
            dtype=torch.float32,
            device=self.device,
        )
        self.regime_teacher_channel_names = ()
        self._regime_teacher_channel_indices_tensor = torch.empty(
            0,
            dtype=torch.long,
            device=self.device,
        )
        if not self.teacher_channels:
            return
        online = RecurrentC51Network(
            self.observation_dim, len(Action), self.atoms, self.hidden_dim
        ).to(self.device)
        target = copy.deepcopy(online).to(self.device).eval()
        online.load_state_dict({
            key: value
            for key, value in self.online.state_dict().items()
            if not key.startswith("teacher_output.")
        })
        target.load_state_dict({
            key: value
            for key, value in self.target.state_dict().items()
            if not key.startswith("teacher_output.")
        })
        self.online = online
        self.target = target
        self.teacher_channels = 0
        self.optimizer = torch.optim.AdamW(
            self.online.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        self.scaler = torch.amp.GradScaler(
            self.device.type,
            enabled=self.mixed_precision == "fp16",
        )
        self._configure_execution()

    def assert_teacher_free(self) -> None:
        """Fail closed unless this policy is safe for validation or shipping."""
        if (
            self.teacher_channels != 0
            or self.teacher_channel_names
            or self.teacher_channel_loss_weights
            or self.teacher_loss_weight != 0.0
            or self.teacher_entry_search_loss_weight != 0.0
            or self.entry_action_loss_weight != 0.0
            or self.regime_selectivity_loss_weight != 0.0
            or self.regime_selectivity is not None
            or self.retention_anchor is not None
            or self.online.teacher_output is not None
            or self.target.teacher_output is not None
            or self._teacher_channel_loss_weights_tensor.numel() != 0
            or self.regime_teacher_channel_names
            or self._regime_teacher_channel_indices_tensor.numel() != 0
        ):
            raise ValueError(
                "validation policy still contains training-only teacher state"
            )

    def _project_distribution(
        self,
        next_distribution: torch.Tensor,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        *,
        bootstrap_discount: float | None = None,
    ) -> torch.Tensor:
        delta = (self.value_max - self.value_min) / (self.atoms - 1)
        discount = (
            self.gamma
            if bootstrap_discount is None
            else float(bootstrap_discount)
        )
        target_support = rewards[..., None] + (
            discount * (~terminated).float()[..., None] * self.support
        )
        target_support.clamp_(self.value_min, self.value_max)
        positions = (target_support - self.value_min) / delta
        lower = positions.floor().long()
        upper = positions.ceil().long()
        projected = torch.zeros_like(next_distribution)
        projected.scatter_add_(-1, lower, next_distribution * (upper.float() - positions))
        projected.scatter_add_(-1, upper, next_distribution * (positions - lower.float()))
        equal = lower == upper
        projected.scatter_add_(-1, lower, next_distribution * equal.float())
        return projected

    def save(self, path: str | Path, *, manifest: dict) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "schema": "propevolve_recurrent_c51_v1",
            "manifest": dict(manifest),
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "updates": self._updates,
            "rng_state": self._rng.bit_generator.state,
            "support": self.support.cpu(),
            "config": {
                "observation_dim": self.observation_dim,
                "hidden_dim": self.hidden_dim,
                "atoms": self.atoms,
                "value_min": self.value_min,
                "value_max": self.value_max,
                "gamma": self.gamma,
                "n_step_return": self.n_step_return,
                "recurrent_burn_in": self.recurrent_burn_in,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "gradient_clip": self.gradient_clip,
                "target_sync_updates": self.target_sync_updates,
                "target_update_mode": self.target_update_mode,
                "target_soft_tau": self.target_soft_tau,
                "seed": self.seed,
                "teacher_channels": self.teacher_channels,
                "teacher_channel_names": self.teacher_channel_names,
                "teacher_loss_weight": self.teacher_loss_weight,
                "teacher_channel_loss_weights": self.teacher_channel_loss_weights,
                "teacher_entry_search_loss_weight": (
                    self.teacher_entry_search_loss_weight
                ),
                "teacher_entry_search_objective": (
                    self.teacher_entry_search_objective
                ),
                "teacher_entry_search_centers": self.teacher_entry_search_centers,
                "teacher_entry_search_probability_epsilon": (
                    self.teacher_entry_search_probability_epsilon
                ),
                "teacher_entry_search_teacher_temperature": (
                    self.teacher_entry_search_teacher_temperature
                ),
                "teacher_entry_search_q_temperature": (
                    self.teacher_entry_search_q_temperature
                ),
                "entry_action_loss_weight": self.entry_action_loss_weight,
                "entry_action_class_weights": self.entry_action_class_weights,
                "entry_action_loss_reduction": self.entry_action_loss_reduction,
                "entry_action_margin": self.entry_action_margin,
                "regime_selectivity_loss_weight": (
                    self.regime_selectivity_loss_weight
                ),
                "regime_selectivity_expansion_centers": (
                    self.regime_selectivity_expansion_centers
                ),
                "regime_selectivity_probability_epsilon": (
                    self.regime_selectivity_probability_epsilon
                ),
                "regime_selectivity_headroom_pressure": (
                    self.regime_selectivity_headroom_pressure
                ),
                "regime_selectivity_dominant_chop_pressure": (
                    self.regime_selectivity_dominant_chop_pressure
                ),
                "regime_selectivity_chop_wait_margin": (
                    self.regime_selectivity_chop_wait_margin
                ),
                "regime_selectivity_failed_confluence_margin": (
                    self.regime_selectivity_failed_confluence_margin
                ),
                "regime_selectivity_q_temperature": (
                    self.regime_selectivity_q_temperature
                ),
                "regime_selectivity_side_balance": (
                    self.regime_selectivity_side_balance
                ),
                "regime_selectivity_semantics": (
                    self.regime_selectivity_semantics
                ),
                "regime_selectivity_persistent_chop_negative_emphasis": (
                    self.regime_selectivity_persistent_chop_negative_emphasis
                ),
                "policy_retention_loss_weight": self.policy_retention_loss_weight,
                "mixed_precision": self.mixed_precision,
                "compile_model": self.compile_model,
                "compile_backend": self.compile_backend,
                "compile_mode": self.compile_mode,
                "mps_prefer_metal": self.mps_prefer_metal,
                "mps_fast_math": self.mps_fast_math,
            },
            "retention_anchor": (
                None
                if self.retention_anchor is None
                else self.retention_anchor.state_dict()
            ),
            "retention_anchor_applies_to_all_management_rows": (
                self.retention_anchor_applies_to_all_management_rows
            ),
        }, path)
        return path

    @classmethod
    def warm_start(
        cls,
        path: str | Path,
        *,
        config: Mapping[str, object],
    ) -> tuple["RecurrentC51Agent", dict]:
        """Load policy weights while resetting optimization and teacher state."""
        payload = torch.load(
            Path(path),
            map_location=str(config["device"]),
            weights_only=False,
        )
        if payload.get("schema") != "propevolve_recurrent_c51_v1":
            raise ValueError("unsupported PropEvolve model bundle")
        parent = dict(payload["config"])
        requested = dict(config)
        structural = (
            "observation_dim",
            "hidden_dim",
            "atoms",
            "value_min",
            "value_max",
        )
        if any(parent[field] != requested[field] for field in structural):
            raise ValueError("warm-start policy architecture drifted")
        agent = cls(**requested)

        def load_shared(network: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
            shared = {
                key: value
                for key, value in state.items()
                if not key.startswith("teacher_output.")
            }
            missing, unexpected = network.load_state_dict(shared, strict=False)
            invalid_missing = [
                key for key in missing if not key.startswith("teacher_output.")
            ]
            if invalid_missing or unexpected:
                raise ValueError("warm-start policy state drifted")

        load_shared(agent.online, payload["online"])
        load_shared(agent.target, payload["target"])
        agent.retain_policy(apply_to_all_management_rows=True)
        return agent, dict(payload["manifest"])

    @classmethod
    def load(cls, path: str | Path, *, device: str) -> tuple["RecurrentC51Agent", dict]:
        payload = torch.load(Path(path), map_location=device, weights_only=False)
        if payload.get("schema") != "propevolve_recurrent_c51_v1":
            raise ValueError("unsupported PropEvolve model bundle")
        config = dict(payload["config"])
        config.setdefault("mixed_precision", "off")
        config.setdefault("compile_model", False)
        config.setdefault("compile_backend", "inductor")
        config.setdefault("compile_mode", "default")
        config.setdefault("mps_prefer_metal", False)
        config.setdefault("mps_fast_math", False)
        config.setdefault("target_update_mode", "hard")
        config.setdefault("target_soft_tau", 1.0)
        config.setdefault("n_step_return", 1)
        config.setdefault("recurrent_burn_in", 0)
        config.setdefault("policy_retention_loss_weight", 0.0)
        config.setdefault("teacher_entry_search_objective", "raw_probability")
        config.setdefault("teacher_entry_search_centers", (0.5, 0.5))
        config.setdefault("teacher_entry_search_probability_epsilon", 1e-6)
        config.setdefault("teacher_entry_search_teacher_temperature", 1.0)
        config.setdefault("teacher_entry_search_q_temperature", 1.0)
        config.setdefault("entry_action_loss_weight", 0.0)
        config.setdefault("entry_action_class_weights", (1.0, 1.0, 1.0))
        config.setdefault(
            "entry_action_loss_reduction", "population_weighted_mean_v1"
        )
        config.setdefault("entry_action_margin", 0.0)
        config.setdefault("teacher_channel_names", ())
        config.setdefault("regime_selectivity_loss_weight", 0.0)
        config.setdefault("regime_selectivity_chop_wait_margin", 0.0)
        config.setdefault("regime_selectivity_failed_confluence_margin", 0.0)
        config.setdefault("regime_selectivity_expansion_centers", None)
        config.setdefault("regime_selectivity_probability_epsilon", 1e-6)
        config.setdefault("regime_selectivity_headroom_pressure", 1.0)
        config.setdefault("regime_selectivity_dominant_chop_pressure", 2.0)
        config.setdefault("regime_selectivity_q_temperature", 1.0)
        config.setdefault("regime_selectivity_side_balance", "none")
        config.setdefault(
            "regime_selectivity_semantics", STATIC_STATE_SEMANTICS
        )
        config.setdefault(
            "regime_selectivity_persistent_chop_negative_emphasis", 0.0
        )
        agent = cls(**config, device=device)
        agent.online.load_state_dict(payload["online"])
        agent.target.load_state_dict(payload["target"])
        agent.optimizer.load_state_dict(payload["optimizer"])
        if "scaler" in payload:
            agent.scaler.load_state_dict(payload["scaler"])
        agent._updates = int(payload["updates"])
        if "rng_state" in payload:
            agent._rng.bit_generator.state = payload["rng_state"]
        if payload.get("retention_anchor") is not None:
            agent.retain_policy(
                apply_to_all_management_rows=bool(
                    payload.get(
                        "retention_anchor_applies_to_all_management_rows",
                        False,
                    )
                )
            )
            assert agent.retention_anchor is not None
            agent.retention_anchor.load_state_dict(payload["retention_anchor"])
        return agent, dict(payload["manifest"])
