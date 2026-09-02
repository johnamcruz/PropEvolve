from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from propevolve.agent import RecurrentC51Agent
from propevolve.balance_aware_regime_selectivity import (
    PAIRED_RECURRENT_A_PLUS_CONTRASTIVE_SEMANTICS,
)
from propevolve.decision import Action
from propevolve.entry_supervision import EntryTargetMetadata
from propevolve.environment import ChallengeStartState
from propevolve.replay import BalancedSequenceReplay, Episode, Transition
from propevolve.teachers.expansion import CHANNELS as EXPANSION_CHANNELS
from propevolve.teachers.regime import CHANNELS as REGIME_CHANNELS
from propevolve.teachers.trend import CHANNELS as TREND_CHANNELS
from propevolve.training import (
    BalanceCurriculumSettings,
    evaluate_agent,
    train_agent,
)
from propevolve.training import _prioritize_paired_a_plus_violations


TEACHER_CHANNELS = (*EXPANSION_CHANNELS, *REGIME_CHANNELS)
TREND_CONFLUENCE_CHANNELS = (*TEACHER_CHANNELS, *TREND_CHANNELS)
FLAT_ACTIONS = (
    Action.WAIT,
    Action.ENTER_LONG_1,
    Action.ENTER_SHORT_1,
)


def _economic_episode(
    *,
    episode_id: str,
    side: Action,
    economic_win: bool,
    context: tuple[float, ...],
    offset: float,
    expansion_history: tuple[tuple[float, float, float, float], ...] | None = None,
) -> Episode:
    target = side if economic_win else Action.WAIT
    outcome = "pass" if economic_win else "timeout"
    transitions = []
    for index in range(12):
        anchor = index == 5
        expansion = (
            expansion_history[index]
            if expansion_history is not None and index <= 5
            else context[:4]
        )
        observation = np.asarray(
            (offset + index / 100.0, expansion[0], expansion[2]),
            dtype=np.float32,
        )
        next_expansion = (
            expansion_history[index + 1]
            if expansion_history is not None and index < 5
            else context[:4]
        )
        next_observation = np.asarray(
            (
                offset + (index + 1) / 100.0,
                next_expansion[0],
                next_expansion[2],
            ),
            dtype=np.float32,
        )
        transitions.append(Transition(
            observation=observation,
            action=side if anchor else Action.WAIT,
            reward=2.0 if economic_win and anchor else (-1.0 if anchor else 0.0),
            next_observation=next_observation,
            terminated=index == 11,
            valid_actions=FLAT_ACTIONS,
            next_valid_actions=() if index == 11 else FLAT_ACTIONS,
            teacher_target=(
                np.asarray((*expansion, *context[4:]), dtype=np.float32)
                if anchor or expansion_history is not None and index <= 5
                else None
            ),
            entry_action_target=target if anchor else None,
            regime_selectivity_headroom_fraction=1.0 if anchor else None,
            paired_a_plus_context=(
                np.asarray(context, dtype=np.float32) if anchor else None
            ),
            paired_a_plus_side=side if anchor else None,
            paired_a_plus_economic_win=economic_win if anchor else None,
            source_decision_index=int(offset * 100) + index,
        ))
    return Episode(
        episode_id=episode_id,
        ticker="NQ",
        outcome=outcome,
        primary_side="long" if side == Action.ENTER_LONG_1 else "short",
        ended_at_ns=int(offset * 1_000),
        transitions=tuple(transitions),
        terminal_pnl=6_100.0 if economic_win else -2_800.0,
    )


def _replay(
    *,
    seed: int,
    paired_a_plus_population_weighting: str = "population_proportional_v1",
    paired_a_plus_context_matching: str = "static_expansion_regime_v1",
    sequence_length: int = 6,
    recurrent_burn_in: int = 2,
) -> BalancedSequenceReplay:
    return BalancedSequenceReplay(
        capacity_episodes=8,
        sequence_length=sequence_length,
        entry_opportunity_sequence_fraction=1.0,
        entry_opportunity_side_balance="paired_recurrent_long_short_v1",
        paired_a_plus_population_weighting=(
            paired_a_plus_population_weighting
        ),
        paired_a_plus_context_matching=paired_a_plus_context_matching,
        recurrent_burn_in=recurrent_burn_in,
        n_step_return=2,
        seed=seed,
    )


def _agent(*, recurrent_burn_in: int = 2) -> RecurrentC51Agent:
    return RecurrentC51Agent(
        3,
        hidden_dim=24,
        atoms=11,
        value_min=-3.0,
        value_max=3.0,
        gamma=0.997,
        learning_rate=0.03,
        weight_decay=0.0,
        gradient_clip=10.0,
        target_sync_updates=250,
        n_step_return=2,
        recurrent_burn_in=recurrent_burn_in,
        device="cpu",
        seed=601,
        teacher_channels=len(TEACHER_CHANNELS),
        teacher_channel_names=TEACHER_CHANNELS,
        teacher_loss_weight=1e-6,
        teacher_entry_search_centers=(0.10, 0.10),
        entry_action_loss_weight=1.0,
        entry_action_margin=0.25,
        regime_selectivity_loss_weight=1.0,
        regime_selectivity_expansion_centers=(0.10, 0.10),
        regime_selectivity_chop_wait_margin=0.25,
        regime_selectivity_failed_confluence_margin=0.25,
        regime_selectivity_paired_a_plus_margin=0.25,
        regime_selectivity_side_balance="paired_recurrent_long_short_v1",
        regime_selectivity_semantics=(
            PAIRED_RECURRENT_A_PLUS_CONTRASTIVE_SEMANTICS
        ),
        regime_selectivity_persistent_chop_negative_emphasis=2.0,
        auxiliary_gradient_conflict_mode=(
            "pcgrad_preserve_economic_boundaries_v3"
        ),
        exclude_economic_winners_from_chop_wait=True,
    )


def _trend_confluence_agent(*, recurrent_burn_in: int = 2) -> RecurrentC51Agent:
    return RecurrentC51Agent(
        3,
        hidden_dim=24,
        atoms=11,
        value_min=-3.0,
        value_max=3.0,
        gamma=0.997,
        learning_rate=0.02,
        weight_decay=0.0,
        gradient_clip=10.0,
        target_sync_updates=250,
        n_step_return=2,
        recurrent_burn_in=recurrent_burn_in,
        device="cpu",
        seed=611,
        teacher_channels=len(TREND_CONFLUENCE_CHANNELS),
        teacher_channel_names=TREND_CONFLUENCE_CHANNELS,
        teacher_loss_weight=1e-6,
        teacher_channel_loss_weights=(
            *((1e-6 / len(TEACHER_CHANNELS),) * len(TEACHER_CHANNELS)),
            *((0.0,) * len(TREND_CHANNELS)),
        ),
        entry_action_loss_weight=1.0,
        entry_action_margin=0.25,
        regime_selectivity_loss_weight=1.0,
        regime_selectivity_expansion_centers=(0.10, 0.10),
        regime_selectivity_chop_wait_margin=0.25,
        regime_selectivity_failed_confluence_margin=0.25,
        regime_selectivity_paired_a_plus_margin=0.25,
        regime_selectivity_side_balance="paired_recurrent_long_short_v1",
        regime_selectivity_semantics=(
            PAIRED_RECURRENT_A_PLUS_CONTRASTIVE_SEMANTICS
        ),
        regime_selectivity_persistent_chop_negative_emphasis=2.0,
        trend_start_confluence_loss_weight=3.0,
        trend_start_confluence_margin=0.25,
        trend_start_confluence_confirmation_lookback_bars=3,
        auxiliary_gradient_conflict_mode=(
            "pcgrad_preserve_economic_boundaries_v3"
        ),
        exclude_economic_winners_from_chop_wait=True,
    )


def _with_trend_targets(
    episode: Episode,
    targets: tuple[float, float, float, float],
    *,
    anchor_targets: tuple[float, float, float, float] | None = None,
) -> Episode:
    return replace(
        episode,
        transitions=tuple(
            replace(
                transition,
                teacher_target=(
                    None
                    if transition.teacher_target is None
                    else np.concatenate((
                        transition.teacher_target,
                        np.asarray(
                            anchor_targets
                            if transition.entry_action_target is not None
                            and anchor_targets is not None
                            else targets,
                            dtype=np.float32,
                        ),
                    ))
                ),
            )
            for transition in episode.transitions
        ),
    )


def _boundary_margins(
    agent: RecurrentC51Agent,
    paired_sequences: tuple[tuple[Transition, ...], ...],
) -> dict[tuple[int, str], float]:
    margins: dict[tuple[int, str], float] = {}
    for sequence in paired_sequences:
        hidden = None
        action_values = None
        anchor_action_values = None
        for transition_index, transition in enumerate(sequence):
            _, hidden, action_values = agent.select_action(
                transition.observation,
                hidden=hidden,
                valid_actions=transition.valid_actions,
                epsilon=0.0,
                return_action_values=True,
            )
            if transition_index == agent.recurrent_burn_in:
                anchor_action_values = action_values
        assert anchor_action_values is not None
        anchor = sequence[agent.recurrent_burn_in]
        assert anchor.paired_a_plus_pair_id is not None
        assert anchor.paired_a_plus_pair_side is not None
        side = anchor.paired_a_plus_pair_side
        pair_id = anchor.paired_a_plus_pair_id
        if anchor.paired_a_plus_economic_win:
            opposite = (
                Action.ENTER_SHORT_1
                if side == Action.ENTER_LONG_1
                else Action.ENTER_LONG_1
            )
            margins[(pair_id, "winner_vs_wait")] = float(
                anchor_action_values[int(side)]
                - anchor_action_values[int(Action.WAIT)]
            )
            margins[(pair_id, "winner_vs_opposite")] = float(
                anchor_action_values[int(side)]
                - anchor_action_values[int(opposite)]
            )
        else:
            margins[(pair_id, "wait_vs_failure")] = float(
                anchor_action_values[int(Action.WAIT)]
                - anchor_action_values[int(side)]
            )
    return margins


def _grouped_boundary_margins(
    agent: RecurrentC51Agent,
    paired_sequences: tuple[tuple[Transition, ...], ...],
) -> dict[str, float]:
    per_group: dict[str, list[float]] = {}
    individual = _boundary_margins(agent, paired_sequences)
    for sequence in paired_sequences:
        anchor = sequence[agent.recurrent_burn_in]
        assert anchor.paired_a_plus_pair_id is not None
        assert anchor.paired_a_plus_pair_side is not None
        side_name = (
            "long"
            if anchor.paired_a_plus_pair_side == Action.ENTER_LONG_1
            else "short"
        )
        if anchor.paired_a_plus_economic_win:
            suffixes = ("winner_vs_wait", "winner_vs_opposite")
        else:
            suffixes = ("wait_vs_failure",)
        for suffix in suffixes:
            per_group.setdefault(f"{side_name}_{suffix}", []).append(
                individual[(anchor.paired_a_plus_pair_id, suffix)]
            )
    return {
        name: float(np.mean(values)) for name, values in per_group.items()
    }


def test_pass_replay_round_trip_drives_balanced_contrastive_recurrent_update(
) -> None:
    """Exercise pass promotion through replay pairing and the real V30 update."""
    long_context = (0.90, 0.85, 0.10, 0.10, 0.10, 0.70, 0.20)
    short_context = (0.10, 0.10, 0.90, 0.85, 0.10, 0.70, 0.20)
    ordinary = _replay(seed=602)
    for episode in (
        _economic_episode(
            episode_id="long-pass",
            side=Action.ENTER_LONG_1,
            economic_win=True,
            context=long_context,
            offset=0.0,
        ),
        _economic_episode(
            episode_id="long-failure",
            side=Action.ENTER_LONG_1,
            economic_win=False,
            context=long_context,
            offset=1.0,
        ),
        _economic_episode(
            episode_id="short-pass",
            side=Action.ENTER_SHORT_1,
            economic_win=True,
            context=short_context,
            offset=2.0,
        ),
        _economic_episode(
            episode_id="short-failure",
            side=Action.ENTER_SHORT_1,
            economic_win=False,
            context=short_context,
            offset=3.0,
        ),
    ):
        ordinary.add(episode)

    pass_replay = _replay(seed=603)
    assert pass_replay.absorb_recent_passes(ordinary, max_examples=8) == 2

    restored_ordinary = _replay(seed=999)
    restored_ordinary.load_state_dict(ordinary.state_dict())
    restored_pass_replay = _replay(seed=998)
    restored_pass_replay.load_state_dict(pass_replay.state_dict())

    pass_sequences = restored_pass_replay.sample_balance_pass_entry_sequences(2)
    pass_anchors = {sequence[2].entry_action_target for sequence in pass_sequences}
    assert pass_anchors == {Action.ENTER_LONG_1, Action.ENTER_SHORT_1}
    assert all(
        sequence[2].paired_a_plus_economic_win is True
        for sequence in pass_sequences
    )

    paired_sequences = restored_ordinary.sample(4)
    pair_anchors = [sequence[2] for sequence in paired_sequences]
    assert len({anchor.paired_a_plus_pair_id for anchor in pair_anchors}) == 2
    for side in (Action.ENTER_LONG_1, Action.ENTER_SHORT_1):
        side_anchors = [
            anchor
            for anchor in pair_anchors
            if anchor.paired_a_plus_pair_side == side
        ]
        assert len(side_anchors) == 2
        assert {anchor.paired_a_plus_economic_win for anchor in side_anchors} == {
            True,
            False,
        }
        assert {anchor.entry_action_target for anchor in side_anchors} == {
            side,
            Action.WAIT,
        }

    agent = _agent()
    before = _boundary_margins(agent, paired_sequences)
    agent.train_batch((*paired_sequences, *pass_sequences))
    after = _boundary_margins(agent, paired_sequences)

    assert before.keys() == after.keys()
    assert all(after[key] >= before[key] - 1e-7 for key in before)
    assert any(after[key] > before[key] for key in before)
    # Six paired A+ boundaries and six exact Long/Short/WAIT boundaries must
    # coexist.  Paired replay must never disable exact failure supervision.
    assert agent.last_train_metrics["economic_boundary_count"] == 12.0
    assert (
        agent.last_train_metrics["economic_boundary_active_constraint_count"]
        == 12.0
    )
    assert agent.last_train_metrics["economic_boundary_backtracks"] == 0.0


def test_trend_confluence_survives_replay_and_production_learner_teacher_free(
    tmp_path,
) -> None:
    replay = _replay(seed=612)
    contexts = {
        Action.ENTER_LONG_1: (
            0.90, 0.85, 0.10, 0.10, 0.10, 0.70, 0.20,
        ),
        Action.ENTER_SHORT_1: (
            0.10, 0.10, 0.90, 0.85, 0.10, 0.70, 0.20,
        ),
    }
    aligned = {
        Action.ENTER_LONG_1: (0.9, 0.1, 0.8, 0.2),
        Action.ENTER_SHORT_1: (0.1, 0.9, 0.2, 0.8),
    }
    countertrend = {
        Action.ENTER_LONG_1: aligned[Action.ENTER_SHORT_1],
        Action.ENTER_SHORT_1: aligned[Action.ENTER_LONG_1],
    }
    offset = 0.0
    for side in (Action.ENTER_LONG_1, Action.ENTER_SHORT_1):
        for economic_win, trend in (
            (True, aligned[side]),
            (False, countertrend[side]),
        ):
            replay.add(_with_trend_targets(
                _economic_episode(
                    episode_id=(
                        f"trend-{side.name}-"
                        f"{'winner' if economic_win else 'failure'}"
                    ),
                    side=side,
                    economic_win=economic_win,
                    context=contexts[side],
                    expansion_history=tuple(
                        contexts[side][:4] for _ in range(6)
                    ),
                    offset=offset,
                ),
                trend,
                anchor_targets=(0.5, 0.5, 0.5, 0.5),
            ))
            offset += 1.0

    paired_sequences = replay.sample(4)
    for sequence in paired_sequences:
        anchor = sequence[2]
        assert anchor.teacher_target is not None
        assert anchor.teacher_target.shape == (len(TREND_CONFLUENCE_CHANNELS),)

    agent = _trend_confluence_agent()
    before = _grouped_boundary_margins(agent, paired_sequences)
    for _ in range(128):
        paired_sequences = replay.sample(4)
        agent.train_batch(paired_sequences)
    metrics = agent.last_train_metrics
    # Only the four authenticated economic anchors receive Trend pressure.
    # Other replay bars carry the same Trend targets to reconstruct recurrent
    # history, but Trend alone must never create an entry label.
    assert metrics["trend_start_confluence_active_rows"] == 4.0
    assert metrics["trend_start_confluence_aligned_long_winner_rows"] > 0
    assert metrics["trend_start_confluence_aligned_short_winner_rows"] > 0
    assert metrics["trend_start_confluence_countertrend_long_failure_rows"] > 0
    assert metrics["trend_start_confluence_countertrend_short_failure_rows"] > 0

    agent.discard_teacher()
    agent.assert_teacher_free()
    policy_path = agent.save(tmp_path / "trend-free-policy.pt", manifest={})
    agent, _ = RecurrentC51Agent.load(policy_path, device="cpu")
    agent.assert_teacher_free()
    after = _grouped_boundary_margins(agent, paired_sequences)
    for side in ("long", "short"):
        assert after[f"{side}_winner_vs_wait"] > before[
            f"{side}_winner_vs_wait"
        ]
        assert after[f"{side}_winner_vs_wait"] >= 0.25
        assert after[f"{side}_winner_vs_opposite"] >= 0.25
        assert after[f"{side}_wait_vs_failure"] >= 0.25

    class TeacherFreeTrendValidation:
        def __init__(self) -> None:
            self.index = 0

        def teacher_lookup(self, ticker: str, decision_index: int):
            raise AssertionError("teacher-free validation accessed Trend")

        def reset(self):
            self.index = 0
            return paired_sequences[0][0].observation.copy(), {
                "ticker": "NQ",
                "valid_actions": FLAT_ACTIONS,
                "realized_pnl": 0.0,
            }

        def step(self, action):
            self.index += 1
            terminated = self.index == len(paired_sequences[0])
            next_index = min(self.index, len(paired_sequences[0]) - 1)
            return (
                paired_sequences[0][next_index].observation.copy(),
                0.0,
                terminated,
                False,
                {
                    "ticker": "NQ",
                    "valid_actions": () if terminated else FLAT_ACTIONS,
                    "outcome": "timeout" if terminated else None,
                    "primary_side": "flat",
                    "trade_count": 0,
                    "win_count": 0,
                    "winning_r_sum": 0.0,
                    "equity_pnl": 0.0,
                    "realized_pnl": 0.0,
                },
            )

    result = evaluate_agent(
        agent,
        TeacherFreeTrendValidation(),
        episodes=1,
        recurrent_horizon=6,
    )
    assert result.timeouts == 1


def test_trend_learner_reuses_legacy_expansion_regime_replay_without_trend_pressure(
) -> None:
    legacy_replay = _replay(seed=613)
    contexts = {
        Action.ENTER_LONG_1: (
            0.90, 0.85, 0.10, 0.10, 0.10, 0.70, 0.20,
        ),
        Action.ENTER_SHORT_1: (
            0.10, 0.10, 0.90, 0.85, 0.10, 0.70, 0.20,
        ),
    }
    offset = 0.0
    for side in (Action.ENTER_LONG_1, Action.ENTER_SHORT_1):
        for economic_win in (True, False):
            legacy_replay.add(_economic_episode(
                episode_id=(
                    f"legacy-{side.name}-"
                    f"{'winner' if economic_win else 'failure'}"
                ),
                side=side,
                economic_win=economic_win,
                context=contexts[side],
                offset=offset,
            ))
            offset += 1.0

    restored_replay = _replay(seed=614)
    restored_replay.load_state_dict(legacy_replay.state_dict())
    sequences = restored_replay.sample(4)
    assert {
        np.asarray(transition.teacher_target).size
        for sequence in sequences
        for transition in sequence
        if transition.teacher_target is not None
    } == {len(TEACHER_CHANNELS)}

    agent = _trend_confluence_agent()
    agent.train_batch(sequences)

    metrics = agent.last_train_metrics
    assert metrics["regime_selectivity_paired_a_plus_pair_count"] == 2.0
    assert metrics["regime_selectivity_paired_a_plus_long_pair_count"] == 1.0
    assert metrics["regime_selectivity_paired_a_plus_short_pair_count"] == 1.0
    assert metrics["trend_start_confluence_active_rows"] == 0.0


def test_trend_learner_rejects_unknown_legacy_teacher_width() -> None:
    episode = _economic_episode(
        episode_id="invalid-legacy-width",
        side=Action.ENTER_LONG_1,
        economic_win=True,
        context=(0.90, 0.85, 0.10, 0.10, 0.10, 0.70, 0.20),
        offset=0.0,
    )
    episode = replace(
        episode,
        transitions=tuple(
            replace(
                transition,
                teacher_target=(
                    None
                    if transition.teacher_target is None
                    else np.concatenate((
                        transition.teacher_target,
                        np.asarray((0.5,), dtype=np.float32),
                    ))
                ),
            )
            for transition in episode.transitions
        ),
    )

    with pytest.raises(ValueError, match="teacher target width drifted"):
        _trend_confluence_agent().train_batch((episode.transitions[2:8],))


def test_lifecycle_violation_replay_transfers_both_sides_teacher_free() -> None:
    """The V34 path must learn lifecycle winners without weakening failures."""
    replay = _replay(
        seed=610,
        paired_a_plus_context_matching=(
            "regime_control_expansion_lifecycle_v1"
        ),
        sequence_length=9,
        recurrent_burn_in=5,
    )
    contexts = {
        Action.ENTER_LONG_1: (
            0.90, 0.85, 0.10, 0.10, 0.10, 0.70, 0.20,
        ),
        Action.ENTER_SHORT_1: (
            0.10, 0.10, 0.90, 0.85, 0.10, 0.70, 0.20,
        ),
    }
    rising = {
        Action.ENTER_LONG_1: tuple(
            (0.20 + 0.14 * index, 0.18 + 0.13 * index, 0.10, 0.10)
            for index in range(6)
        ),
        Action.ENTER_SHORT_1: tuple(
            (0.10, 0.10, 0.20 + 0.14 * index, 0.18 + 0.13 * index)
            for index in range(6)
        ),
    }
    for side, offset in (
        (Action.ENTER_LONG_1, 0.0),
        (Action.ENTER_SHORT_1, 2.0),
    ):
        context = contexts[side]
        replay.add(_economic_episode(
            episode_id=f"{side.name}-lifecycle-winner",
            side=side,
            economic_win=True,
            context=context,
            expansion_history=rising[side],
            offset=offset,
        ))
        replay.add(_economic_episode(
            episode_id=f"{side.name}-lifecycle-failure",
            side=side,
            economic_win=False,
            context=context,
            expansion_history=tuple(context[:4] for _ in range(6)),
            offset=offset,
        ))

    candidates = replay.sample_paired_a_plus_candidate_pairs(1)
    agent = _agent(recurrent_burn_in=5)
    agent.regime_selectivity_paired_a_plus_winner_loss_weight = 3.0
    for group in agent.optimizer.param_groups:
        group["lr"] = 0.01
    before = _grouped_boundary_margins(agent, candidates)
    for _ in range(256):
        candidates = replay.sample_paired_a_plus_candidate_pairs(1)
        selected, diagnostic = _prioritize_paired_a_plus_violations(
            agent,
            candidates,
            pairs_per_side=1,
        )
        assert diagnostic["long_selected_pairs"] == 1.0
        assert diagnostic["short_selected_pairs"] == 1.0
        agent.train_batch(selected)

    agent.discard_teacher()
    agent.assert_teacher_free()
    after = _grouped_boundary_margins(agent, candidates)
    for side in ("long", "short"):
        assert after[f"{side}_winner_vs_wait"] > before[
            f"{side}_winner_vs_wait"
        ]
        assert after[f"{side}_winner_vs_wait"] >= 0.25, {
            "before": before,
            "after": after,
        }
        assert after[f"{side}_winner_vs_opposite"] >= 0.25
        assert after[f"{side}_wait_vs_failure"] >= 0.25

    winner_sequences = tuple(
        sequence
        for sequence in candidates
        if sequence[agent.recurrent_burn_in].paired_a_plus_economic_win
    )
    assert tuple(
        sequence[agent.recurrent_burn_in].paired_a_plus_pair_side
        for sequence in winner_sequences
    ) == (Action.ENTER_LONG_1, Action.ENTER_SHORT_1)

    class TeacherFreeLifecycleValidation:
        def __init__(self) -> None:
            self.episode_index = -1
            self.step_index = 0
            self.actions: list[list[Action]] = []

        def teacher_lookup(self, ticker: str, decision_index: int):
            raise AssertionError("teacher-free validation accessed a teacher")

        def reset(self):
            self.episode_index += 1
            self.step_index = 0
            self.actions.append([])
            sequence = winner_sequences[self.episode_index]
            return sequence[0].observation.copy(), {
                "ticker": "NQ",
                "valid_actions": FLAT_ACTIONS,
                "realized_pnl": 0.0,
            }

        def step(self, action):
            sequence = winner_sequences[self.episode_index]
            self.actions[self.episode_index].append(Action(action))
            self.step_index += 1
            terminated = self.step_index == len(sequence)
            next_index = min(self.step_index, len(sequence) - 1)
            return (
                sequence[next_index].observation.copy(),
                0.0,
                terminated,
                False,
                {
                    "ticker": "NQ",
                    "valid_actions": () if terminated else FLAT_ACTIONS,
                    "outcome": "timeout" if terminated else None,
                    "primary_side": "flat",
                    "trade_count": 0,
                    "win_count": 0,
                    "winning_r_sum": 0.0,
                    "equity_pnl": 0.0,
                    "realized_pnl": 0.0,
                },
            )

    validation = TeacherFreeLifecycleValidation()
    result = evaluate_agent(
        agent,
        validation,
        episodes=2,
        recurrent_horizon=9,
    )

    assert result.timeouts == 2
    assert validation.actions[0][agent.recurrent_burn_in] == (
        Action.ENTER_LONG_1
    )
    assert validation.actions[1][agent.recurrent_burn_in] == (
        Action.ENTER_SHORT_1
    )


def test_equal_pair_mass_survives_replay_resume_and_real_optimizer_update(
) -> None:
    replay = _replay(
        seed=604,
        paired_a_plus_population_weighting="equal_pair_mass_v1",
    )
    contexts = {
        Action.ENTER_LONG_1: (
            0.90, 0.85, 0.10, 0.10, 0.10, 0.70, 0.20,
        ),
        Action.ENTER_SHORT_1: (
            0.10, 0.10, 0.90, 0.85, 0.10, 0.70, 0.20,
        ),
    }
    offset = 0.0
    for side, context in contexts.items():
        replay.add(_economic_episode(
            episode_id=f"{side.name}-winner",
            side=side,
            economic_win=True,
            context=context,
            offset=offset,
        ))
        offset += 1.0
        for failure_index in range(3):
            replay.add(_economic_episode(
                episode_id=f"{side.name}-failure-{failure_index}",
                side=side,
                economic_win=False,
                context=context,
                offset=offset,
            ))
            offset += 1.0

    restored = _replay(
        seed=999,
        paired_a_plus_population_weighting="equal_pair_mass_v1",
    )
    restored.load_state_dict(replay.state_dict())
    paired_sequences = restored.sample(4)
    anchors = [sequence[2] for sequence in paired_sequences]
    assert {anchor.paired_a_plus_pair_side for anchor in anchors} == {
        Action.ENTER_LONG_1,
        Action.ENTER_SHORT_1,
    }
    assert all(
        anchor.paired_a_plus_population_weight == pytest.approx(1.0)
        for anchor in anchors
    )

    agent = _agent()
    before = _grouped_boundary_margins(agent, paired_sequences)
    agent.train_batch(paired_sequences)
    after = _grouped_boundary_margins(agent, paired_sequences)

    assert before.keys() == after.keys()
    assert after["long_winner_vs_wait"] > before["long_winner_vs_wait"]
    assert after["short_winner_vs_wait"] > before["short_winner_vs_wait"]
    assert after["long_wait_vs_failure"] > 0.0
    assert after["short_wait_vs_failure"] > 0.0


def test_four_economic_pairs_update_all_optimizer_boundaries_without_stalling(
) -> None:
    """More pair examples must strengthen learning, not reject the update."""
    contexts = (
        (Action.ENTER_LONG_1, (0.90, 0.85, 0.10, 0.10, 0.10, 0.70, 0.20)),
        (Action.ENTER_SHORT_1, (0.10, 0.10, 0.90, 0.85, 0.10, 0.70, 0.20)),
        (Action.ENTER_LONG_1, (0.62, 0.51, 0.42, 0.39, 0.15, 0.55, 0.30)),
        (Action.ENTER_SHORT_1, (0.35, 0.25, 0.65, 0.58, 0.20, 0.50, 0.30)),
    )
    replay = _replay(seed=91)
    for index, (side, context) in enumerate(contexts):
        replay.add(_economic_episode(
            episode_id=f"winner-{index}",
            side=side,
            economic_win=True,
            context=context,
            offset=float(index * 2),
        ))
        replay.add(_economic_episode(
            episode_id=f"failure-{index}",
            side=side,
            economic_win=False,
            context=context,
            offset=float(index * 2 + 1),
        ))
    paired_sequences = replay.sample(8)
    agent = _agent()
    before = _grouped_boundary_margins(agent, paired_sequences)
    satisfied = {
        name for name, margin in before.items() if margin >= 0.25
    }
    parameters_before = tuple(
        parameter.detach().clone() for parameter in agent.online.parameters()
    )

    agent.train_batch(paired_sequences)

    after = _grouped_boundary_margins(agent, paired_sequences)
    parameter_delta = sum(
        ((updated.detach() - original) ** 2).sum()
        for original, updated in zip(
            parameters_before, agent.online.parameters(), strict=True
        )
    ).sqrt()
    assert agent.last_train_metrics["economic_boundary_count"] == 12.0
    assert (
        agent.last_train_metrics["economic_boundary_active_constraint_count"]
        == 12.0
    )
    assert agent.last_train_metrics["economic_boundary_backtracks"] < 12.0
    assert (
        agent.last_train_metrics["economic_boundary_hard_constraint_count"]
        <= agent.last_train_metrics[
            "economic_boundary_active_constraint_count"
        ]
    )
    assert before.keys() == after.keys()
    assert float(parameter_delta) > 0.0
    assert all(after[name] >= 0.25 - 1e-7 for name in satisfied)


def test_satisfied_economic_boundaries_stay_in_every_optimizer_projection(
) -> None:
    """Reproduce r5: one satisfied side must not disappear from protection."""
    contexts = (
        (Action.ENTER_LONG_1, (0.90, 0.85, 0.10, 0.10, 0.10, 0.70, 0.20)),
        (Action.ENTER_SHORT_1, (0.10, 0.10, 0.90, 0.85, 0.10, 0.70, 0.20)),
        (Action.ENTER_LONG_1, (0.62, 0.51, 0.42, 0.39, 0.15, 0.55, 0.30)),
        (Action.ENTER_SHORT_1, (0.35, 0.25, 0.65, 0.58, 0.20, 0.50, 0.30)),
    )
    replay = _replay(seed=91)
    for index, (side, context) in enumerate(contexts):
        replay.add(_economic_episode(
            episode_id=f"winner-{index}",
            side=side,
            economic_win=True,
            context=context,
            offset=float(index * 2),
        ))
        replay.add(_economic_episode(
            episode_id=f"failure-{index}",
            side=side,
            economic_win=False,
            context=context,
            offset=float(index * 2 + 1),
        ))
    sequences = replay.sample(8)
    agent = _agent()

    for _ in range(3):
        before = _grouped_boundary_margins(agent, sequences)
        satisfied = {
            name for name, margin in before.items() if margin >= 0.25
        }
        agent.train_batch(sequences)
        after = _grouped_boundary_margins(agent, sequences)

        assert agent.last_train_metrics["economic_boundary_count"] == 12.0
        assert (
            agent.last_train_metrics[
                "economic_boundary_active_constraint_count"
            ]
            == 12.0
        )
        assert agent.last_train_metrics["economic_boundary_backtracks"] < 12.0
        assert (
            agent.last_train_metrics[
                "economic_boundary_hard_constraint_count"
            ]
            <= agent.last_train_metrics[
                "economic_boundary_active_constraint_count"
            ]
        )
        assert all(
            after[name] >= 0.25 - 1e-7 for name in satisfied
        ), {"before": before, "after": after}


def test_unsatisfied_boundary_cannot_rollback_the_whole_optimizer_step_e2e(
) -> None:
    """Reproduce v31: learning must continue while correct margins stay safe."""
    contexts = (
        (Action.ENTER_LONG_1, (0.90, 0.85, 0.10, 0.10, 0.10, 0.70, 0.20)),
        (Action.ENTER_SHORT_1, (0.10, 0.10, 0.90, 0.85, 0.10, 0.70, 0.20)),
        (Action.ENTER_LONG_1, (0.62, 0.51, 0.42, 0.39, 0.15, 0.55, 0.30)),
        (Action.ENTER_SHORT_1, (0.35, 0.25, 0.65, 0.58, 0.20, 0.50, 0.30)),
    )
    replay = _replay(seed=91)
    for index, (side, context) in enumerate(contexts):
        for economic_win, kind in ((True, "winner"), (False, "failure")):
            replay.add(_economic_episode(
                episode_id=f"{kind}-{index}",
                side=side,
                economic_win=economic_win,
                context=context,
                offset=float(index * 2 + int(not economic_win)),
            ))
    sequences = replay.sample(8)
    agent = _agent()
    agent.regime_selectivity_paired_a_plus_winner_loss_weight = 2.0
    for _ in range(41):
        agent.train_batch(sequences)

    margins_before = _grouped_boundary_margins(agent, sequences)
    satisfied = {
        name for name, margin in margins_before.items() if margin >= 0.25
    }
    unsatisfied = set(margins_before) - satisfied
    assert satisfied
    if not unsatisfied:
        assert all(margin >= 0.25 for margin in margins_before.values())
        return
    parameters_before = tuple(
        parameter.detach().clone() for parameter in agent.online.parameters()
    )

    agent.train_batch(sequences)

    margins_after = _grouped_boundary_margins(agent, sequences)
    parameter_delta = sum(
        ((after.detach() - before) ** 2).sum()
        for before, after in zip(
            parameters_before, agent.online.parameters(), strict=True
        )
    ).sqrt()
    assert agent.last_train_metrics["economic_boundary_backtracks"] < 12.0
    assert float(parameter_delta) > 0.0
    assert all(margins_after[name] >= 0.25 - 1e-7 for name in satisfied)


@pytest.mark.parametrize(
    ("side", "side_name", "contexts"),
    (
        (
            Action.ENTER_LONG_1,
            "long",
            (
                (0.90, 0.85, 0.10, 0.10, 0.10, 0.70, 0.20),
                (0.65, 0.58, 0.35, 0.25, 0.20, 0.50, 0.30),
            ),
        ),
        (
            Action.ENTER_SHORT_1,
            "short",
            (
                (0.10, 0.10, 0.90, 0.85, 0.10, 0.70, 0.20),
                (0.35, 0.25, 0.65, 0.58, 0.20, 0.50, 0.30),
            ),
        ),
    ),
)
def test_repeated_pairs_learn_entry_opposite_and_wait_boundaries_e2e(
    side: Action,
    side_name: str,
    contexts: tuple[tuple[float, ...], ...],
) -> None:
    """Each authenticated side must be learnable, not merely preserved."""
    replay = _replay(seed=91)
    for index, context in enumerate(contexts):
        for economic_win, kind in ((True, "winner"), (False, "failure")):
            replay.add(_economic_episode(
                episode_id=f"{side_name}-{kind}-{index}",
                side=side,
                economic_win=economic_win,
                context=context,
                offset=float(index * 2 + int(not economic_win)),
            ))
    sequences = replay.sample(4)
    agent = _agent()
    agent.regime_selectivity_paired_a_plus_winner_loss_weight = 2.0
    for group in agent.optimizer.param_groups:
        group["lr"] = 0.003

    for _ in range(128):
        agent.train_batch(sequences)

    agent.discard_teacher()
    agent.assert_teacher_free()
    margins = _grouped_boundary_margins(agent, sequences)
    assert margins[f"{side_name}_winner_vs_wait"] >= 0.25
    assert margins[f"{side_name}_winner_vs_opposite"] >= 0.25
    assert margins[f"{side_name}_wait_vs_failure"] >= 0.25


def test_directional_tie_replay_learns_wait_over_both_entries_e2e() -> None:
    """Equal Long and Short evidence must become a learned WAIT boundary."""
    tied_context = (0.90, 0.85, 0.90, 0.85, 0.10, 0.70, 0.20)
    replay = _replay(seed=92)
    for index, side in enumerate(
        (Action.ENTER_LONG_1, Action.ENTER_SHORT_1)
    ):
        replay.add(_economic_episode(
            episode_id=f"tied-failure-{index}",
            side=side,
            economic_win=False,
            context=tied_context,
            offset=float(index),
        ))
    sequences = replay.sample(2)
    agent = _agent()
    for group in agent.optimizer.param_groups:
        group["lr"] = 0.003

    for _ in range(256):
        agent.train_batch(sequences)

    agent.discard_teacher()
    agent.assert_teacher_free()
    for sequence in sequences:
        hidden = None
        anchor_values = None
        for transition_index, transition in enumerate(sequence):
            _, hidden, action_values = agent.select_action(
                transition.observation,
                hidden=hidden,
                valid_actions=transition.valid_actions,
                epsilon=0.0,
                return_action_values=True,
            )
            if transition_index == agent.recurrent_burn_in:
                anchor_values = action_values
        assert anchor_values is not None
        assert (
            anchor_values[int(Action.WAIT)]
            >= anchor_values[int(Action.ENTER_LONG_1)] + 0.25
        )
        assert (
            anchor_values[int(Action.WAIT)]
            >= anchor_values[int(Action.ENTER_SHORT_1)] + 0.25
        )


def test_train_agent_reports_pass_replay_and_contrastive_boundaries_e2e(
) -> None:
    """Prove the campaign training seam retains every economic-flow receipt."""
    long_context = np.asarray(
        (0.90, 0.85, 0.10, 0.10, 0.10, 0.70, 0.20),
        dtype=np.float32,
    )
    short_context = (0.10, 0.10, 0.90, 0.85, 0.10, 0.70, 0.20)

    class PassingEnvironment:
        def __init__(self) -> None:
            self.index = 0

        def reset(self, *, options=None):
            self.index = 0
            assert options is not None
            start = options["challenge_start_state"]
            return np.asarray((0.0, 0.90, 0.10), np.float32), {
                "valid_actions": FLAT_ACTIONS,
                "ticker": "NQ",
                "start": 0,
                "end": 12,
                "realized_pnl": float(start.realized_pnl),
                "mll_headroom_fraction": 0.5,
            }

        def step(self, action):
            del action
            self.index += 1
            terminated = self.index == 12
            pnl = 6_100.0 if terminated else -1_500.0 + 650.0 * self.index
            return (
                np.asarray((self.index / 100.0, 0.90, 0.10), np.float32),
                1.0 if terminated else 0.0,
                terminated,
                False,
                {
                    "valid_actions": () if terminated else FLAT_ACTIONS,
                    "ticker": "NQ",
                    "fill_index": self.index,
                    "outcome": "pass" if terminated else None,
                    "primary_side": "long",
                    "trade_count": 1,
                    "win_count": 1,
                    "winning_r_sum": 2.0,
                    "win_rate": 1.0,
                    "avg_win_r": 2.0,
                    "realized_pnl": pnl,
                    "equity_pnl": pnl,
                    "mll_headroom_fraction": 0.5,
                },
            )

    ordinary = _replay(seed=604)
    for episode in (
        _economic_episode(
            episode_id="long-failure-seed",
            side=Action.ENTER_LONG_1,
            economic_win=False,
            context=tuple(long_context),
            offset=1.0,
        ),
        _economic_episode(
            episode_id="short-pass-seed",
            side=Action.ENTER_SHORT_1,
            economic_win=True,
            context=short_context,
            offset=2.0,
        ),
        _economic_episode(
            episode_id="short-failure-seed",
            side=Action.ENTER_SHORT_1,
            economic_win=False,
            context=short_context,
            offset=3.0,
        ),
    ):
        ordinary.add(episode)

    pass_replay = _replay(seed=605)
    persisted: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    metadata = EntryTargetMetadata(
        side="long",
        event_anchor_rows=(5,),
        candidate_decision_offset=0,
        fill_offset=1,
        continuation=True,
        economic_win=True,
        economic_good=True,
        available=True,
        censored=False,
        unavailable_reason=None,
    )

    result = train_agent(
        _agent(),
        PassingEnvironment(),
        episodes=1,
        minimum_environment_steps=12,
        budget_mode="episodes",
        replay=ordinary,
        warmup_episodes=1,
        updates_per_episode=1,
        batch_sequences=4,
        recurrent_horizon=6,
        epsilon_start=0.0,
        epsilon_end=0.0,
        episode_tickers=("NQ",),
        ticker_seed=5,
        teacher_lookup=(
            lambda ticker, index: long_context.copy() if index == 5 else None
        ),
        teacher_channels=TEACHER_CHANNELS,
        entry_action_lookup=(
            lambda ticker, index: Action.ENTER_LONG_1 if index == 5 else None
        ),
        entry_action_metadata_lookup=(
            lambda ticker, index: metadata if index == 5 else None
        ),
        balance_curriculum=BalanceCurriculumSettings(
            schedule_seed=37,
            start_pnls=(-1_500.0,),
            mll_floor_pnl=-3_000.0,
            pass_replay_update_period=1,
            pass_replay_max_examples=8,
            pass_replay_path="runs/e2e-pass-replay.pt",
            pass_replay_sha256="a" * 64,
            pass_replay_output="e2e-pass-replay.pt",
        ),
        balance_pass_replay=pass_replay,
        balance_pass_replay_callback=persisted.append,
        episode_diagnostic_callback=diagnostics.append,
    )

    assert result.passes == 1
    assert result.blows == result.timeouts == 0
    assert len(persisted) == 1
    assert {
        episode["episode_id"] for episode in persisted[0]["episodes"]
    } == {"short-pass-seed", next(
        episode["episode_id"]
        for episode in persisted[0]["episodes"]
        if episode["episode_id"].startswith("historical-")
    )}
    diagnostic = diagnostics[0]
    assert diagnostic["balance_pass_replay_promoted_passes"] == 1
    assert diagnostic["balance_pass_replay_sequences"] == 1
    assert diagnostic["updates"] == 1
    assert diagnostic["mean_regime_selectivity_paired_a_plus_pair_count"] == 2.0
    assert diagnostic[
        "mean_regime_selectivity_paired_a_plus_long_pair_count"
    ] == 1.0
    assert diagnostic[
        "mean_regime_selectivity_paired_a_plus_short_pair_count"
    ] == 1.0
    assert diagnostic["mean_economic_boundary_count"] == 12.0
    assert diagnostic[
        "mean_economic_boundary_active_constraint_count"
    ] == 12.0
    assert diagnostic["mean_economic_boundary_backtracks"] == 0.0
    assert diagnostic["mean_economic_boundary_final_min_margin_delta"] >= -1e-7
    assert diagnostic[
        "mean_economic_boundary_final_min_required_headroom"
    ] >= -1e-7
    assert diagnostic[
        "mean_economic_boundary_long_winner_min_margin_delta"
    ] >= -1e-7
    assert diagnostic[
        "mean_economic_boundary_short_winner_min_margin_delta"
    ] >= -1e-7
    assert diagnostic[
        "mean_economic_boundary_failed_long_min_margin_delta"
    ] >= -1e-7
    assert diagnostic[
        "mean_economic_boundary_failed_short_min_margin_delta"
    ] >= -1e-7


@pytest.mark.parametrize(
    ("side", "context", "exit_reason", "authenticated_target", "expected"),
    (
        (
            Action.ENTER_LONG_1,
            (0.90, 0.85, 0.10, 0.10, 0.10, 0.20, 0.70),
            "initial_stop",
            False,
            Action.WAIT,
        ),
        (
            Action.ENTER_SHORT_1,
            (0.10, 0.10, 0.90, 0.85, 0.10, 0.20, 0.70),
            "initial_stop",
            False,
            Action.WAIT,
        ),
        (
            Action.ENTER_LONG_1,
            (0.90, 0.85, 0.10, 0.10, 0.10, 0.20, 0.70),
            "voluntary_close",
            False,
            None,
        ),
        (
            Action.ENTER_SHORT_1,
            (0.10, 0.10, 0.90, 0.85, 0.10, 0.20, 0.70),
            "initial_stop",
            True,
            Action.ENTER_SHORT_1,
        ),
    ),
)
def test_executed_entry_economic_supervision_boundary_e2e(
    side: Action,
    context: tuple[float, ...],
    exit_reason: str,
    authenticated_target: bool,
    expected: Action | None,
) -> None:
    """Fill only unlabeled initial-stop failures; preserve every other row."""
    teacher_context = np.asarray(context, dtype=np.float32)
    side_name = "long" if side == Action.ENTER_LONG_1 else "short"

    class FailedEntryEnvironment:
        def __init__(self) -> None:
            self.index = 0

        def reset(self, *, options=None):
            self.index = 0
            return np.asarray((0.0, 0.10, 0.90), np.float32), {
                "valid_actions": FLAT_ACTIONS,
                "ticker": "CL",
                "start": 0,
                "end": 2,
                "realized_pnl": -1_500.0,
                "mll_headroom_fraction": 0.5,
            }

        def step(self, action):
            self.index += 1
            if self.index == 1:
                assert action == side
                return (
                    np.asarray((0.1, 0.10, 0.90), np.float32),
                    0.0,
                    False,
                    False,
                    {
                        "valid_actions": (Action.HOLD, Action.CLOSE),
                        "ticker": "CL",
                        "fill_index": 1,
                        "outcome": None,
                        "primary_side": side_name,
                        "realized_pnl": -1_500.0,
                        "equity_pnl": -1_500.0,
                        "mll_headroom_fraction": 0.5,
                    },
                )
            assert action in {Action.HOLD, Action.CLOSE}
            return (
                np.asarray((0.2, 0.10, 0.90), np.float32),
                -1.0,
                True,
                False,
                {
                    "valid_actions": (),
                    "ticker": "CL",
                    "fill_index": 2,
                    "outcome": "timeout",
                    "primary_side": side_name,
                    "trade_count": 1,
                    "win_count": 0,
                    "winning_r_sum": 0.0,
                    "win_rate": 0.0,
                    "avg_win_r": 0.0,
                    "realized_pnl": -1_800.0,
                    "equity_pnl": -1_800.0,
                    "mll_headroom_fraction": 0.4,
                },
            )

        def closed_trade_receipts(self):
            return ({
                "trade_index": 0,
                "ticker": "CL",
                "side": side_name,
                "source_decision_index": 0,
                "entry_index": 1,
                "exit_index": 2,
                "entry_timestamp": "2024-01-01T00:01",
                "exit_timestamp": "2024-01-01T00:02",
                "entry_realized_pnl": -1_500.0,
                "entry_mll_floor_pnl": -3_000.0,
                "entry_mll_headroom": 1_500.0,
                "pnl": -300.0,
                "realized_r": -1.0,
                "mfe_r": 0.2,
                "mae_r": 1.0,
                "ratchet_activated": False,
                "exit_reason": exit_reason,
                "hold_bars": 1,
            },)

    replay = _replay(seed=606)
    replay.add(_economic_episode(
        episode_id=f"{side_name}-winner-seed",
        side=side,
        economic_win=True,
        context=tuple(teacher_context),
        offset=9.0,
    ))
    diagnostics: list[dict[str, object]] = []
    agent = _agent()
    original_select_action = agent.select_action

    def select_action(
        observation,
        *,
        hidden,
        valid_actions,
        epsilon,
        return_action_values=False,
    ):
        selected, next_hidden, values = original_select_action(
            observation,
            hidden=hidden,
            valid_actions=valid_actions,
            epsilon=epsilon,
            return_action_values=return_action_values,
        )
        if set(valid_actions) == set(FLAT_ACTIONS):
            selected = side
        return selected, next_hidden, values

    agent.select_action = select_action
    metadata = EntryTargetMetadata(
        side=side_name,
        event_anchor_rows=(0,),
        candidate_decision_offset=0,
        fill_offset=1,
        continuation=True,
        economic_win=True,
        economic_good=True,
        available=True,
        censored=False,
        unavailable_reason=None,
    )
    train_agent(
        agent,
        FailedEntryEnvironment(),
        episodes=1,
        minimum_environment_steps=2,
        budget_mode="episodes",
        replay=replay,
        warmup_episodes=2,
        updates_per_episode=1,
        batch_sequences=2,
        recurrent_horizon=6,
        epsilon_start=0.0,
        epsilon_end=0.0,
        episode_tickers=("CL",),
        ticker_seed=7,
        teacher_lookup=(
            lambda ticker, index: teacher_context.copy() if index == 0 else None
        ),
        teacher_channels=TEACHER_CHANNELS,
        entry_action_lookup=(
            lambda ticker, index: side
            if authenticated_target and index == 0
            else None
        ),
        entry_action_metadata_lookup=(
            lambda ticker, index: metadata
            if authenticated_target and index == 0
            else None
        ),
        episode_diagnostic_callback=diagnostics.append,
    )

    payload = next(
        episode
        for episode in replay.state_dict()["episodes"]
        if episode["episode_id"].startswith("historical-")
    )
    assert payload["actions"][0] == int(side)
    assert payload["entry_action_targets"][0] == (
        -1 if expected is None else int(expected)
    )
    if expected is None:
        assert payload["paired_a_plus_sides"][0] == -1
        assert payload["paired_a_plus_economic_wins"][0] == -1
    else:
        assert payload["paired_a_plus_sides"][0] == int(side)
        assert payload["paired_a_plus_economic_wins"][0] == int(
            expected == side
        )
        np.testing.assert_allclose(
            payload["paired_a_plus_contexts"][0],
            teacher_context,
            rtol=0,
            atol=0,
        )
    promoted = int(exit_reason == "initial_stop" and not authenticated_target)
    assert diagnostics[0]["executed_failure_supervision_promoted_rows"] == promoted
    assert diagnostics[0][
        f"executed_failure_supervision_promoted_{side_name}_rows"
    ] == promoted
    if promoted:
        anchors = [
            sequence[2]
            for sequence in replay.sample(2)
            if sequence[2].paired_a_plus_pair_side == side
        ]
        assert len(anchors) == 2
        assert {anchor.paired_a_plus_economic_win for anchor in anchors} == {
            True,
            False,
        }
        assert {anchor.entry_action_target for anchor in anchors} == {
            side,
            Action.WAIT,
        }
