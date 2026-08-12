"""Recurrent distributional Double-DQN for exact masked action scoring."""

from __future__ import annotations

import copy
from contextlib import nullcontext
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .decision import Action
from .config import configure_runtime_environment
from .replay import Transition


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
    center_tensor = torch.as_tensor(
        center, dtype=bounded.dtype, device=bounded.device
    )
    centered_log_odds = (
        torch.logit(bounded) - torch.logit(center_tensor)
    ) / teacher_temperature
    return torch.sigmoid(centered_log_odds)


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
        teacher_loss_weight: float = 0.0,
        teacher_channel_loss_weights: Sequence[float] | None = None,
        teacher_entry_search_loss_weight: float = 0.0,
        teacher_entry_search_objective: str = "raw_probability",
        teacher_entry_search_centers: Sequence[float] = (0.5, 0.5),
        teacher_entry_search_probability_epsilon: float = 1e-6,
        teacher_entry_search_teacher_temperature: float = 1.0,
        teacher_entry_search_q_temperature: float = 1.0,
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
            or policy_retention_loss_weight < 0
        ):
            raise ValueError("teacher settings must be nonnegative")
        if bool(teacher_channels) != bool(teacher_loss_weight):
            raise ValueError("teacher channels and loss weight must be enabled together")
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
        if mixed_precision not in {"off", "fp16"}:
            raise ValueError("mixed precision must be off or fp16")
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
        self.policy_retention_loss_weight = float(policy_retention_loss_weight)
        self.retention_anchor: RecurrentC51Network | None = None
        self.last_train_metrics: dict[str, float] = {}
        self.support = torch.linspace(value_min, value_max, atoms, device=self.device)
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
        explore = self._rng.random() < epsilon
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

    def train_batch(
        self,
        sequences: Sequence[Sequence[Transition]],
        *,
        teacher_weight_scale: float = 1.0,
    ) -> float:
        if not sequences:
            raise ValueError("training batch cannot be empty")
        teacher_weight_scale = float(teacher_weight_scale)
        if (
            not np.isfinite(teacher_weight_scale)
            or not 0 <= teacher_weight_scale <= 1
        ):
            raise ValueError("teacher weight scale must be between zero and one")
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
        all_rewards = torch.as_tensor(
            [[item.reward for item in sequence] for sequence in sequences],
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
        n_step_rewards = torch.zeros_like(immediate_rewards)
        bootstrap_alive = torch.ones_like(immediate_rewards, dtype=torch.bool)
        discount = 1.0
        for offset in range(self.n_step_return):
            reward_slice = all_rewards[
                :,
                learning_start + offset:learning_start + offset + training_steps,
            ]
            terminated_slice = all_terminated[
                :,
                learning_start + offset:learning_start + offset + training_steps,
            ]
            n_step_rewards = n_step_rewards + (
                discount * bootstrap_alive.to(all_rewards.dtype) * reward_slice
            )
            bootstrap_alive = bootstrap_alive & ~terminated_slice
            discount *= self.gamma
        terminated = ~bootstrap_alive
        next_masks = all_next_masks[
            :,
            learning_start + self.n_step_return - 1:
            learning_start + self.n_step_return - 1 + training_steps,
        ]

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
            online_next = self.online.distribution_logits(
                causal_recurrent[
                    :,
                    self.n_step_return:self.n_step_return + training_steps,
                ]
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
                target_next = target_causal[
                    :,
                    self.n_step_return:self.n_step_return + training_steps,
                ]
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
                terminated,
                bootstrap_discount=self.gamma**self.n_step_return,
            )
        td_losses = -(projected * chosen_logits.log_softmax(-1)).sum(-1)
        rl_loss = td_losses.mean()
        loss = rl_loss
        teacher_loss = torch.zeros((), dtype=torch.float32, device=self.device)
        entry_search_loss = torch.zeros((), dtype=torch.float32, device=self.device)
        policy_retention_loss = torch.zeros(
            (), dtype=torch.float32, device=self.device
        )
        # A zero curriculum scale is the explicit autonomy boundary.  Do not
        # even read replayed teacher targets or execute the discarded-use
        # auxiliary head once that boundary has been reached.
        if self.teacher_channels and teacher_weight_scale > 0.0:
            teacher_targets = np.full(
                (*observations.shape[:2], self.teacher_channels),
                np.nan,
                dtype=np.float32,
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
            teacher_rows_numpy = np.isfinite(teacher_targets).all(axis=-1)
            teacher_targets_tensor = torch.as_tensor(
                teacher_targets, dtype=torch.float32, device=self.device
            )[:, self.recurrent_burn_in:]
            teacher_rows = torch.as_tensor(
                teacher_rows_numpy[:, self.recurrent_burn_in:],
                device=self.device,
            )
            if bool(teacher_rows.any().item()):
                assert self.online.teacher_output is not None
                with self._autocast():
                    teacher_logits = self.online.teacher_output(recurrent)
                teacher_losses = nn.functional.binary_cross_entropy_with_logits(
                    teacher_logits.float()[teacher_rows],
                    teacher_targets_tensor[teacher_rows],
                    reduction="none",
                )
                channel_weights = torch.as_tensor(
                    self.teacher_channel_loss_weights,
                    dtype=teacher_losses.dtype,
                    device=self.device,
                )
                teacher_loss = (teacher_losses * channel_weights).sum(dim=-1).mean()
                loss = loss + teacher_weight_scale * teacher_loss
                if self.teacher_entry_search_loss_weight:
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
        management_valid_rows = (
            valid_masks[:, self.recurrent_burn_in:, int(Action.HOLD)]
            & valid_masks[:, self.recurrent_burn_in:, int(Action.CLOSE)]
            & (valid_masks[:, self.recurrent_burn_in:].sum(-1) == 2)
        )
        retention_rows = (
            competence_anchors[:, self.recurrent_burn_in:]
            & management_valid_rows
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
        gradient_norm = nn.utils.clip_grad_norm_(
            self.online.parameters(), max_norm=self.gradient_clip
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self._updates += 1
        self._update_target_network()
        q_values = (logits.softmax(-1) * self.support).sum(-1)
        rl_valid_masks = valid_masks[
            :, learning_start:learning_start + training_steps
        ]
        management_rows = (
            rl_valid_masks[..., int(Action.HOLD)]
            & rl_valid_masks[..., int(Action.CLOSE)]
            & (rl_valid_masks.sum(-1) == 2)
        )
        hold_rows = actions == int(Action.HOLD)
        close_rows = actions == int(Action.CLOSE)

        def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            weights = mask.to(values.dtype)
            return (values * weights).sum() / weights.sum().clamp_min(1.0)

        diagnostic_values = torch.stack((
            rl_loss,
            teacher_loss,
            entry_search_loss,
            policy_retention_loss,
            loss,
            gradient_norm.float(),
            management_rows.float().mean(),
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
            torch.as_tensor(
                sum(sum(row) for row in recurrent_reset_rows)
                / (len(recurrent_reset_rows) * sequence_length),
                dtype=torch.float32,
                device=self.device,
            ),
            torch.as_tensor(
                (
                    sum(any(row) for row in burn_in_reset_rows)
                    / len(burn_in_reset_rows)
                    if self.recurrent_burn_in else 1.0
                ),
                dtype=torch.float32,
                device=self.device,
            ),
        ))
        (
            rl_loss_value,
            teacher_loss_value,
            entry_search_loss_value,
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
        ) = (
            float(value)
            for value in diagnostic_values.detach().float().cpu().tolist()
        )
        self.last_train_metrics = {
            "rl_loss": rl_loss_value,
            "teacher_loss": teacher_loss_value,
            "entry_search_loss": entry_search_loss_value,
            "policy_retention_loss": policy_retention_loss_value,
            "teacher_weight_scale": teacher_weight_scale,
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
            "sampled_recurrent_reset_pattern_count": float(
                len(set(recurrent_reset_rows))
            ),
            "n_step_return": float(self.n_step_return),
            "recurrent_burn_in": float(self.recurrent_burn_in),
        }
        return total_loss

    def retain_policy(self) -> None:
        """Freeze a training-only copy of demonstrated pass competence."""
        self.retention_anchor = copy.deepcopy(self.online).to(self.device).eval()
        self.retention_anchor.requires_grad_(False)

    def discard_retention_anchor(self) -> None:
        """Remove training-only competence state before evaluation or shipping."""
        self.retention_anchor = None

    def discard_teacher(self) -> None:
        """Remove the training-only head while retaining shared learned weights."""
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
        self.teacher_loss_weight = 0.0
        self.teacher_channel_loss_weights = ()
        self.teacher_entry_search_loss_weight = 0.0
        self.teacher_entry_search_objective = "raw_probability"
        self.teacher_entry_search_centers = (0.5, 0.5)
        self.optimizer = torch.optim.AdamW(
            self.online.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        self.scaler = torch.amp.GradScaler(
            self.device.type,
            enabled=self.mixed_precision == "fp16",
        )
        self._configure_execution()

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
            agent.retain_policy()
            assert agent.retention_anchor is not None
            agent.retention_anchor.load_state_dict(payload["retention_anchor"])
        return agent, dict(payload["manifest"])
