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
        device: str,
        seed: int,
        teacher_channels: int = 0,
        teacher_loss_weight: float = 0.0,
        teacher_channel_loss_weights: Sequence[float] | None = None,
        teacher_entry_search_loss_weight: float = 0.0,
        mixed_precision: str = "off",
        compile_model: bool = False,
        compile_backend: str = "inductor",
        compile_mode: str = "default",
        mps_prefer_metal: bool = False,
        mps_fast_math: bool = False,
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
        self.learning_rate = float(learning_rate)
        self.value_min = float(value_min)
        self.value_max = float(value_max)
        self.target_sync_updates = int(target_sync_updates)
        self.weight_decay = float(weight_decay)
        self.gradient_clip = float(gradient_clip)
        self.teacher_channels = int(teacher_channels)
        self.teacher_loss_weight = float(teacher_loss_weight)
        self.teacher_channel_loss_weights = teacher_channel_loss_weights
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
        self.scaler = torch.amp.GradScaler(
            self.device.type,
            enabled=self.mixed_precision == "fp16",
        )
        self.compile_status = "disabled"
        self.compile_error = ""
        self._configure_execution()
        self._updates = 0

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
        if self._rng.random() < epsilon:
            selected = valid_actions[int(self._rng.integers(len(valid_actions)))]
            with torch.no_grad():
                value = torch.as_tensor(observation, device=self.device).view(1, 1, -1)
                with self._autocast():
                    _, next_hidden = self._call_with_compile_fallback(
                        self._online_forward, self.online.forward, value, hidden
                    )
            values = np.full(len(Action), np.nan) if return_action_values else None
            return selected, next_hidden.detach(), values
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

        with self._autocast():
            recurrent, _ = self._call_with_compile_fallback(
                self._online_recurrent,
                self.online.recurrent_features,
                observations,
            )
            logits = self.online.distribution_logits(recurrent)
        logits = logits.float()
        chosen_logits = logits.gather(
            2,
            actions[..., None, None].expand(-1, -1, 1, self.atoms),
        ).squeeze(2)
        with torch.no_grad():
            with self._autocast():
                online_next, _ = self._call_with_compile_fallback(
                    self._online_forward,
                    self.online.forward,
                    next_observations,
                    None,
                )
                target_next, _ = self._call_with_compile_fallback(
                    self._target_forward,
                    self.target.forward,
                    next_observations,
                    None,
                )
            online_q = (online_next.float().softmax(-1) * self.support).sum(-1)
            online_q = online_q.masked_fill(~next_masks, -torch.inf)
            next_actions = online_q.argmax(-1)
            target_distribution = target_next.float().softmax(-1).gather(
                2,
                next_actions[..., None, None].expand(-1, -1, 1, self.atoms),
            ).squeeze(2)
            projected = self._project_distribution(
                target_distribution, rewards, terminated
            )
        rl_loss = -(projected * chosen_logits.log_softmax(-1)).sum(-1).mean()
        loss = rl_loss
        teacher_loss = torch.zeros((), dtype=torch.float32, device=self.device)
        entry_search_loss = torch.zeros((), dtype=torch.float32, device=self.device)
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
            teacher_rows_numpy = np.isfinite(teacher_targets).all(axis=-1)
            teacher_targets_tensor = torch.as_tensor(
                teacher_targets, dtype=torch.float32, device=self.device
            )
            teacher_rows = torch.as_tensor(teacher_rows_numpy, device=self.device)
            if teacher_rows_numpy.any():
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
                loss = loss + teacher_loss
                if self.teacher_entry_search_loss_weight:
                    valid_masks_numpy = np.asarray(
                        [[
                            [action in item.valid_actions for action in Action]
                            for item in sequence
                        ] for sequence in sequences],
                        dtype=np.bool_,
                    )
                    entry_rows_numpy = (
                        teacher_rows_numpy
                        & valid_masks_numpy[..., int(Action.WAIT)]
                        & valid_masks_numpy[..., int(Action.ENTER_LONG_1)]
                        & valid_masks_numpy[..., int(Action.ENTER_SHORT_1)]
                    )
                    if entry_rows_numpy.any():
                        entry_rows = torch.as_tensor(
                            entry_rows_numpy, device=self.device
                        )
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
                        loss = loss + (
                            self.teacher_entry_search_loss_weight
                            * entry_search_loss
                        )
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=self.gradient_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self._updates += 1
        if self._updates % self.target_sync_updates == 0:
            self.target.load_state_dict(self.online.state_dict())
        metrics = torch.stack((rl_loss, teacher_loss, entry_search_loss, loss))
        rl_loss_value, teacher_loss_value, entry_search_loss_value, total_loss = (
            float(value) for value in metrics.detach().float().cpu().tolist()
        )
        self.last_train_metrics = {
            "rl_loss": rl_loss_value,
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
        self.teacher_channel_loss_weights = ()
        self.teacher_entry_search_loss_weight = 0.0
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
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "gradient_clip": self.gradient_clip,
                "target_sync_updates": self.target_sync_updates,
                "seed": self.seed,
                "teacher_channels": self.teacher_channels,
                "teacher_loss_weight": self.teacher_loss_weight,
                "teacher_channel_loss_weights": self.teacher_channel_loss_weights,
                "teacher_entry_search_loss_weight": (
                    self.teacher_entry_search_loss_weight
                ),
                "mixed_precision": self.mixed_precision,
                "compile_model": self.compile_model,
                "compile_backend": self.compile_backend,
                "compile_mode": self.compile_mode,
                "mps_prefer_metal": self.mps_prefer_metal,
                "mps_fast_math": self.mps_fast_math,
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
        config.setdefault("mixed_precision", "off")
        config.setdefault("compile_model", False)
        config.setdefault("compile_backend", "inductor")
        config.setdefault("compile_mode", "default")
        config.setdefault("mps_prefer_metal", False)
        config.setdefault("mps_fast_math", False)
        agent = cls(**config, device=device)
        agent.online.load_state_dict(payload["online"])
        agent.target.load_state_dict(payload["target"])
        agent.optimizer.load_state_dict(payload["optimizer"])
        if "scaler" in payload:
            agent.scaler.load_state_dict(payload["scaler"])
        agent._updates = int(payload["updates"])
        if "rng_state" in payload:
            agent._rng.bit_generator.state = payload["rng_state"]
        return agent, dict(payload["manifest"])
