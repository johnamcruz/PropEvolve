"""Recurrent distributional Double-DQN for exact masked action scoring."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .decision import Action
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
        return self.recurrent(self.input(observations), hidden)

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
        device: str,
        seed: int,
        teacher_channels: int = 0,
        teacher_loss_weight: float = 0.0,
        teacher_entry_search_loss_weight: float = 0.0,
    ) -> None:
        if atoms < 2 or value_min >= value_max:
            raise ValueError("distributional support is invalid")
        if learning_rate <= 0 or weight_decay < 0 or gradient_clip <= 0:
            raise ValueError("optimizer settings are invalid")
        if target_sync_updates < 1:
            raise ValueError("target_sync_updates must be positive")
        if (
            teacher_channels < 0
            or teacher_loss_weight < 0
            or teacher_entry_search_loss_weight < 0
        ):
            raise ValueError("teacher settings must be nonnegative")
        if bool(teacher_channels) != bool(teacher_loss_weight):
            raise ValueError("teacher channels and loss weight must be enabled together")
        torch.manual_seed(seed)
        self.seed = int(seed)
        self._rng = np.random.default_rng(seed)
        self.device = resolve_device(device)
        self.observation_dim = int(observation_dim)
        self.hidden_dim = int(hidden_dim)
        self.atoms = int(atoms)
        self.gamma = float(gamma)
        self.learning_rate = float(learning_rate)
        self.value_min = float(value_min)
        self.value_max = float(value_max)
        self.target_sync_updates = int(target_sync_updates)
        self.weight_decay = float(weight_decay)
        self.gradient_clip = float(gradient_clip)
        self.teacher_channels = int(teacher_channels)
        self.teacher_loss_weight = float(teacher_loss_weight)
        self.teacher_entry_search_loss_weight = float(
            teacher_entry_search_loss_weight
        )
        self.last_train_metrics: dict[str, float] = {}
        self.support = torch.linspace(value_min, value_max, atoms, device=self.device)
        self.online = RecurrentC51Network(
            observation_dim, len(Action), atoms, hidden_dim, self.teacher_channels
        ).to(self.device)
        self.target = copy.deepcopy(self.online).to(self.device).eval()
        self.optimizer = torch.optim.AdamW(
            self.online.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        self._updates = 0

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
        if self._rng.random() < epsilon:
            selected = valid_actions[int(self._rng.integers(len(valid_actions)))]
            with torch.no_grad():
                value = torch.as_tensor(observation, device=self.device).view(1, 1, -1)
                _, next_hidden = self.online(value, hidden)
            values = np.full(len(Action), np.nan) if return_action_values else None
            return selected, next_hidden.detach(), values
        with torch.no_grad():
            value = torch.as_tensor(
                observation, dtype=torch.float32, device=self.device
            ).view(1, 1, -1)
            logits, next_hidden = self.online(value, hidden)
            q_values = (logits.softmax(-1) * self.support).sum(-1)[0, 0]
            valid = torch.zeros(len(Action), dtype=torch.bool, device=self.device)
            valid[[int(action) for action in valid_actions]] = True
            q_values = q_values.masked_fill(~valid, -torch.inf)
            selected = Action(int(q_values.argmax().item()))
        values = q_values.cpu().numpy() if return_action_values else None
        return selected, next_hidden.detach(), values

    def train_batch(self, sequences: Sequence[Sequence[Transition]]) -> float:
        if not sequences:
            raise ValueError("training batch cannot be empty")
        lengths = {len(sequence) for sequence in sequences}
        if len(lengths) != 1 or next(iter(lengths)) < 1:
            raise ValueError("training sequences must have one positive length")
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
        actions = torch.as_tensor(
            [[int(item.action) for item in sequence] for sequence in sequences],
            dtype=torch.long,
            device=self.device,
        )
        rewards = torch.as_tensor(
            [[item.reward for item in sequence] for sequence in sequences],
            dtype=torch.float32,
            device=self.device,
        )
        terminated = torch.as_tensor(
            [[item.terminated for item in sequence] for sequence in sequences],
            dtype=torch.bool,
            device=self.device,
        )
        next_masks = torch.as_tensor(
            [[
                [action in item.next_valid_actions for action in Action]
                for item in sequence
            ] for sequence in sequences],
            dtype=torch.bool,
            device=self.device,
        )

        recurrent, _ = self.online.recurrent_features(observations)
        logits = self.online.distribution_logits(recurrent)
        chosen_logits = logits.gather(
            2,
            actions[..., None, None].expand(-1, -1, 1, self.atoms),
        ).squeeze(2)
        with torch.no_grad():
            online_next, _ = self.online(next_observations)
            online_q = (online_next.softmax(-1) * self.support).sum(-1)
            online_q = online_q.masked_fill(~next_masks, -torch.inf)
            next_actions = online_q.argmax(-1)
            target_next, _ = self.target(next_observations)
            target_distribution = target_next.softmax(-1).gather(
                2,
                next_actions[..., None, None].expand(-1, -1, 1, self.atoms),
            ).squeeze(2)
            projected = self._project_distribution(
                target_distribution, rewards, terminated
            )
        rl_loss = -(projected * chosen_logits.log_softmax(-1)).sum(-1).mean()
        loss = rl_loss
        teacher_loss_value = 0.0
        entry_search_loss_value = 0.0
        if self.teacher_channels:
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
            teacher_targets_tensor = torch.as_tensor(
                teacher_targets, dtype=torch.float32, device=self.device
            )
            teacher_rows = torch.isfinite(teacher_targets_tensor).all(dim=-1)
            if teacher_rows.any():
                assert self.online.teacher_output is not None
                teacher_logits = self.online.teacher_output(recurrent)
                teacher_loss = nn.functional.binary_cross_entropy_with_logits(
                    teacher_logits[teacher_rows],
                    teacher_targets_tensor[teacher_rows],
                )
                teacher_loss_value = float(teacher_loss.detach().cpu())
                loss = loss + self.teacher_loss_weight * teacher_loss
                if self.teacher_entry_search_loss_weight:
                    valid_masks = torch.as_tensor(
                        [[
                            [action in item.valid_actions for action in Action]
                            for item in sequence
                        ] for sequence in sequences],
                        dtype=torch.bool,
                        device=self.device,
                    )
                    entry_rows = (
                        teacher_rows
                        & valid_masks[..., int(Action.WAIT)]
                        & valid_masks[..., int(Action.ENTER_LONG_1)]
                        & valid_masks[..., int(Action.ENTER_SHORT_1)]
                    )
                    if entry_rows.any():
                        q_values = (logits.softmax(-1) * self.support).sum(-1)
                        long_target = (
                            teacher_targets_tensor[..., 0]
                            * teacher_targets_tensor[..., 1]
                        ).clamp(0.0, 1.0)
                        short_target = (
                            teacher_targets_tensor[..., 2]
                            * teacher_targets_tensor[..., 3]
                        ).clamp(0.0, 1.0)
                        long_advantage = (
                            q_values[..., int(Action.ENTER_LONG_1)]
                            - q_values[..., int(Action.WAIT)]
                        )
                        short_advantage = (
                            q_values[..., int(Action.ENTER_SHORT_1)]
                            - q_values[..., int(Action.WAIT)]
                        )
                        entry_search_loss = 0.5 * (
                            nn.functional.binary_cross_entropy_with_logits(
                                long_advantage[entry_rows], long_target[entry_rows]
                            )
                            + nn.functional.binary_cross_entropy_with_logits(
                                short_advantage[entry_rows], short_target[entry_rows]
                            )
                        )
                        entry_search_loss_value = float(
                            entry_search_loss.detach().cpu()
                        )
                        loss = loss + (
                            self.teacher_entry_search_loss_weight
                            * entry_search_loss
                        )
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=self.gradient_clip)
        self.optimizer.step()
        self._updates += 1
        if self._updates % self.target_sync_updates == 0:
            self.target.load_state_dict(self.online.state_dict())
        total_loss = float(loss.detach().cpu())
        self.last_train_metrics = {
            "rl_loss": float(rl_loss.detach().cpu()),
            "teacher_loss": teacher_loss_value,
            "entry_search_loss": entry_search_loss_value,
            "total_loss": total_loss,
        }
        return total_loss

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
        self.teacher_entry_search_loss_weight = 0.0
        self.optimizer = torch.optim.AdamW(
            self.online.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

    def _project_distribution(
        self,
        next_distribution: torch.Tensor,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
    ) -> torch.Tensor:
        delta = (self.value_max - self.value_min) / (self.atoms - 1)
        target_support = rewards[..., None] + (
            self.gamma * (~terminated).float()[..., None] * self.support
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
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "gradient_clip": self.gradient_clip,
                "target_sync_updates": self.target_sync_updates,
                "seed": self.seed,
                "teacher_channels": self.teacher_channels,
                "teacher_loss_weight": self.teacher_loss_weight,
                "teacher_entry_search_loss_weight": (
                    self.teacher_entry_search_loss_weight
                ),
            },
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
        agent = cls(**config, device=device)
        agent.online.load_state_dict(payload["online"])
        agent.target.load_state_dict(payload["target"])
        agent.optimizer.load_state_dict(payload["optimizer"])
        agent._updates = int(payload["updates"])
        if "rng_state" in payload:
            agent._rng.bit_generator.state = payload["rng_state"]
        return agent, dict(payload["manifest"])
