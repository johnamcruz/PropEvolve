from __future__ import annotations
import argparse, hashlib, json, pickle, random
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from propevolve.agent import RecurrentC51Agent, exact_action_margin_losses
from propevolve.causal_separability import (
    canonical_side_context,
    temporal_ridge_separability,
)
from propevolve.decision import Action
from propevolve.replay import BalancedSequenceReplay
from propevolve.training import _prioritize_paired_a_plus_violations

SIDES = (Action.ENTER_LONG_1, Action.ENTER_SHORT_1)
SN = {SIDES[0]: "long", SIDES[1]: "short"}


def causal_feature_families(
    context: np.ndarray,
    teacher_targets: np.ndarray,
    anchor_index: int,
    side: Action,
) -> dict[str, np.ndarray]:
    """Expose the causal feature families already available at one entry row."""
    canonical = canonical_side_context(context, side)
    families = {
        "static_expansion": canonical[:4],
        "regime": canonical[4:7],
        "static_expansion_regime": canonical,
    }
    targets = np.asarray(teacher_targets, dtype=np.float32)
    if (
        targets.ndim != 2
        or anchor_index < 5
        or anchor_index >= len(targets)
        or targets.shape[1] < 4
    ):
        return families
    history = targets[anchor_index - 5:anchor_index + 1]
    if not np.isfinite(history[:, :4]).all():
        return families
    expansion = history[:, :4]
    if side == Action.ENTER_SHORT_1:
        expansion = expansion[:, [2, 3, 0, 1]]
    expansion_lifecycle = np.concatenate((
        canonical[:4],
        expansion[-1] - expansion[-2],
        expansion[-1] - expansion[0],
    )).astype(np.float32, copy=False)
    families["expansion_lifecycle"] = expansion_lifecycle
    combined = [expansion_lifecycle, canonical[4:7]]
    if targets.shape[1] >= 11 and np.isfinite(history[:, 7:11]).all():
        trend = history[:, 7:11]
        if side == Action.ENTER_SHORT_1:
            trend = trend[:, [1, 0, 3, 2]]
        trend_lifecycle = np.concatenate((
            trend[-1],
            trend[-1] - trend[-2],
            trend[-1] - trend[0],
        )).astype(np.float32, copy=False)
        families["trend_lifecycle"] = trend_lifecycle
        combined.append(trend_lifecycle)
    families["combined_causal_lifecycle"] = np.concatenate(combined)
    return families


def sha(path):
    d = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            d.update(chunk)
    return d.hexdigest()


def discounted_returns_to_go(rewards: np.ndarray, *, discount: float) -> np.ndarray:
    if (
        rewards.ndim != 1
        or not np.isfinite(rewards).all()
        or isinstance(discount, bool)
        or not np.isfinite(discount)
        or not 0.0 < discount <= 1.0
    ):
        raise ValueError("challenge return audit contract is invalid")
    result = np.empty(len(rewards), dtype=np.float64)
    value = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        value = float(rewards[index]) + discount * value
        result[index] = value
    return result


def challenge_outcome_cohort(
    outcome: str,
    terminal_pnl: float,
    near_blow_pnl: float,
) -> str:
    if outcome == "pass":
        return "pass"
    if outcome == "blow":
        return "blow"
    if terminal_pnl >= 0.0:
        return "nonnegative_timeout"
    if terminal_pnl <= near_blow_pnl:
        return "near_blow_timeout"
    return "negative_timeout"


def challenge_return_credit_coverage(
    sequences,
    *,
    recurrent_burn_in: int,
    n_step_return: int,
) -> dict[str, object]:
    if not sequences:
        return {
            "sequences": 0,
            "credited_rows": 0,
            "learnable_credited_rows": 0,
            "discarded_credited_rows": 0,
            "actions": {},
        }
    sequence_length = len(sequences[0])
    learning_end = sequence_length - n_step_return
    counts = Counter()
    credited_rows = 0
    learnable_rows = 0
    for sequence in sequences:
        if len(sequence) != sequence_length:
            raise ValueError("challenge-return credit coverage is ragged")
        for index, transition in enumerate(sequence):
            if transition.challenge_return_to_go is None:
                continue
            credited_rows += 1
            counts[transition.action.name] += 1
            if recurrent_burn_in <= index <= learning_end:
                learnable_rows += 1
    return {
        "sequences": len(sequences),
        "credited_rows": credited_rows,
        "learnable_credited_rows": learnable_rows,
        "discarded_credited_rows": credited_rows - learnable_rows,
        "actions": dict(counts),
    }


def make_replay(
    contract,
    schema,
    payloads,
    *,
    paired_a_plus_context_matching=None,
):
    effective_contract = dict(contract)
    if paired_a_plus_context_matching is not None:
        effective_contract["paired_a_plus_context_matching"] = (
            paired_a_plus_context_matching
        )
        effective_contract.setdefault("paired_a_plus_control_candidates", 8)
        schema = 15
    r = BalancedSequenceReplay(
        capacity_episodes=effective_contract["capacity_episodes"],
        capacity_transitions=effective_contract["capacity_transitions"],
        sequence_length=effective_contract["sequence_length"],
        terminal_sequence_fraction=effective_contract["terminal_sequence_fraction"],
        safety_sequence_fraction=effective_contract["safety_sequence_fraction"],
        entry_opportunity_sequence_fraction=effective_contract[
            "entry_opportunity_sequence_fraction"
        ],
        regime_wait_sequence_fraction=effective_contract[
            "regime_wait_sequence_fraction"
        ],
        regime_wait_sequence_update_period=effective_contract[
            "regime_wait_sequence_update_period"
        ],
        entry_opportunity_side_balance=effective_contract[
            "entry_opportunity_side_balance"
        ],
        paired_a_plus_population_weighting=effective_contract.get(
            "paired_a_plus_population_weighting",
            "population_proportional_v1",
        ),
        paired_a_plus_context_matching=effective_contract.get(
            "paired_a_plus_context_matching",
            "static_expansion_regime_v1",
        ),
        paired_a_plus_control_candidates=effective_contract.get(
            "paired_a_plus_control_candidates",
            8,
        ),
        recurrent_burn_in=effective_contract["recurrent_burn_in"],
        n_step_return=effective_contract["n_step_return"],
        seed=314159,
    )
    r.load_state_dict(
        {
            "schema_version": schema,
            "contract": effective_contract,
            "random_state": random.Random(314159).getstate(),
            "sample_calls": 0,
            "episodes": payloads,
        }
    )
    return r


def qtrace(agent, seqs):
    obs = torch.as_tensor(
        np.stack([[t.observation for t in s] for s in seqs]),
        dtype=torch.float32,
        device=agent.device,
    )
    nxt = torch.as_tensor(
        np.stack([[t.next_observation for t in s] for s in seqs]),
        dtype=torch.float32,
        device=agent.device,
    )
    causal = torch.cat((obs[:, :1], nxt), 1)
    resets = tuple(
        (s[0].recurrent_reset, *(t.next_recurrent_reset for t in s)) for s in seqs
    )
    b = agent.recurrent_burn_in
    hidden = None
    if b:
        with torch.no_grad():
            _, hidden = agent._recurrent_features_with_resets(
                agent.online, causal[:, :b], tuple(r[:b] for r in resets)
            )
        hidden = hidden.detach()
    rec, _ = agent._recurrent_features_with_resets(
        agent.online, causal[:, b:], tuple(r[b:] for r in resets), hidden
    )
    logits = agent.online.distribution_logits(rec[:, :-1]).float()
    return (logits.softmax(-1) * agent.support).sum(-1), causal, resets


def flatgrad(agent):
    return torch.cat(
        [
            (torch.zeros_like(p) if p.grad is None else p.grad)
            .detach()
            .reshape(-1)
            .cpu()
            for p in agent.online.parameters()
        ]
    )


def gradloss(agent, fn):
    agent.online.zero_grad(set_to_none=True)
    loss = fn(agent)
    loss.backward()
    return flatgrad(agent), float(loss.detach())


def tdgrad(checkpoint, seqs):
    a, _ = RecurrentC51Agent.load(
        checkpoint,
        device="cpu",
        learner_backend_override="pytorch",
    )
    a.policy_retention_loss_weight = 0.0
    a.trend_start_confluence_loss_weight = 0.0
    a.gradient_clip = 1e12
    a.optimizer.step = lambda *x, **y: None
    a.train_batch(
        seqs,
        teacher_weight_scale=0.0,
        entry_action_weight_scale=0.0,
        recovery_target=None,
        recovery_value_loss_weight=0.0,
        retain_nonnegative_entry_policy=False,
    )
    return flatgrad(a), float(a.last_train_metrics["rl_loss"])


def learnmask(seqs, burn, nstep):
    steps = len(seqs[0]) - burn - nstep + 1
    out = np.zeros((len(seqs), steps), bool)
    for bi, s in enumerate(seqs):
        for ti in range(steps):
            for off in range(nstep):
                t = s[burn + ti + off]
                if not t.training_valid:
                    break
                if t.terminated or off == nstep - 1:
                    out[bi, ti] = True
                    break
    return out


def exactloss(a, seqs):
    q = qtrace(a, seqs)[0]
    steps = len(seqs[0]) - a.recurrent_burn_in - a.n_step_return + 1
    q = q[:, :steps, :3]
    mask = learnmask(seqs, a.recurrent_burn_in, a.n_step_return)
    targets = np.full(mask.shape, -1, np.int64)
    for bi, s in enumerate(seqs):
        for ti in range(steps):
            x = s[a.recurrent_burn_in + ti].entry_action_target
            if x in {Action.WAIT, *SIDES}:
                targets[bi, ti] = int(x)
    rows = torch.as_tensor(mask & (targets >= 0))
    tar = torch.as_tensor(targets, dtype=torch.long)
    sq = q[rows]
    st = tar[rows]
    ce = F.cross_entropy(sq, st, reduction="none")
    mg = exact_action_margin_losses(sq, st, margin=a.entry_action_margin)
    counts = torch.bincount(st, minlength=3)
    vals = [(ce[st == i] + mg[st == i]).mean() for i in range(3) if counts[i] > 0]
    return a.entry_action_loss_weight * torch.stack(vals).mean()


def waitgrad(checkpoint, seqs, td):
    ws = tuple(
        tuple(
            replace(
                t,
                entry_action_target=t.entry_action_target
                if t.entry_action_target == Action.WAIT
                else None,
                paired_a_plus_pair_id=None,
                paired_a_plus_pair_side=None,
                paired_a_plus_population_weight=None,
            )
            for t in s
        )
        for s in seqs
    )
    a, _ = RecurrentC51Agent.load(
        checkpoint,
        device="cpu",
        learner_backend_override="pytorch",
    )
    a.policy_retention_loss_weight = 0.0
    a.entry_action_loss_weight = 0.0
    a.trend_start_confluence_loss_weight = 0.0
    a.gradient_clip = 1e12
    a.optimizer.step = lambda *x, **y: None
    a.train_batch(
        ws,
        teacher_weight_scale=0.0,
        entry_action_weight_scale=1.0,
        recovery_target=None,
        recovery_value_loss_weight=0.0,
        retain_nonnegative_entry_policy=False,
    )
    keys = (
        "regime_selectivity_exact_wait_rows",
        "regime_selectivity_dominant_chop_rows",
        "regime_selectivity_failed_long_confluence_rows",
        "regime_selectivity_failed_short_confluence_rows",
    )
    return (
        flatgrad(a) - td,
        float(a.last_train_metrics["regime_selectivity_loss"]),
        {k: float(a.last_train_metrics.get(k, 0)) for k in keys},
    )


def pairlosses(a, seqs):
    q = qtrace(a, seqs)[0][:, 0, :3]
    pairs = defaultdict(dict)
    for i, s in enumerate(seqs):
        t = s[a.recurrent_burn_in]
        pairs[int(t.paired_a_plus_pair_id)][bool(t.paired_a_plus_economic_win)] = (i, t)
    sidevals = defaultdict(lambda: defaultdict(list))
    for rows in pairs.values():
        wi, wt = rows[True]
        fi, ft = rows[False]
        side = Action(wt.paired_a_plus_pair_side)
        good = q[wi, int(side)] - q[wi, 0]
        bad = q[fi, int(side)] - q[fi, 0]
        sidevals[side]["paired_relative"].append(
            F.softplus(a.regime_selectivity_paired_a_plus_margin + bad - good) / 3
        )
        sidevals[side]["paired_winner"].append(
            a.regime_selectivity_paired_a_plus_winner_loss_weight
            * float(wt.paired_a_plus_population_weight)
            * F.softplus(a.entry_action_margin - good)
            / 3
        )
        sidevals[side]["paired_failure"].append(
            float(ft.paired_a_plus_population_weight)
            * F.softplus(a.entry_action_margin + bad)
            / 3
        )
    out = {}
    for n in ("paired_relative", "paired_winner", "paired_failure"):
        out[n] = (
            a.regime_selectivity_loss_weight
            * torch.stack(
                [torch.stack(sidevals[s][n]).mean() for s in SIDES if sidevals[s][n]]
            ).mean()
        )
    out["balance_outcome_total"] = sum(out.values())
    return out


def cosine(a, b):
    d = float(a.norm() * b.norm())
    return 0.0 if d == 0 else max(-1.0, min(1.0, float(torch.dot(a, b) / d)))


def applygrad(a, g):
    off = 0
    with torch.no_grad():
        for p in a.online.parameters():
            n = p.numel()
            p.add_(g[off : off + n].view_as(p), alpha=-a.learning_rate)
            off += n


def grouped_economic_margins(q_values, seqs, *, recurrent_burn_in):
    values = defaultdict(list)
    for row, sequence in zip(q_values, seqs, strict=True):
        anchor = sequence[recurrent_burn_in]
        side = Action(anchor.paired_a_plus_pair_side)
        side_name = SN[side]
        if anchor.paired_a_plus_economic_win:
            opposite = SIDES[1] if side == SIDES[0] else SIDES[0]
            values[f"{side_name}_winner_over_wait"].append(
                float(row[int(side)] - row[int(Action.WAIT)])
            )
            values[f"{side_name}_winner_over_opposite"].append(
                float(row[int(side)] - row[int(opposite)])
            )
        else:
            values[f"{side_name}_wait_over_failure"].append(
                float(row[int(Action.WAIT)] - row[int(side)])
            )
    return {name: float(np.mean(rows)) for name, rows in values.items()}


def actual_configured_update(checkpoint, seqs, before):
    """Measure the real optimizer path on the frozen recurrent batch."""
    agent, _ = RecurrentC51Agent.load(
        checkpoint,
        device="cpu",
        learner_backend_override="pytorch",
    )
    parameters_before = tuple(
        parameter.detach().clone() for parameter in agent.online.parameters()
    )
    margins_before = grouped_economic_margins(
        before,
        seqs,
        recurrent_burn_in=agent.recurrent_burn_in,
    )
    agent.train_batch(seqs)
    after = qtrace(agent, seqs)[0][:, 0, :3].detach().cpu()
    margins_after = grouped_economic_margins(
        after,
        seqs,
        recurrent_burn_in=agent.recurrent_burn_in,
    )
    parameter_delta_norm = sum(
        ((after_parameter.detach() - before_parameter) ** 2).sum()
        for before_parameter, after_parameter in zip(
            parameters_before, agent.online.parameters(), strict=True
        )
    ).sqrt()
    metrics = agent.last_train_metrics
    return {
        "parameter_delta_norm": float(parameter_delta_norm),
        "mean_abs_q_change": float((after - before).abs().mean()),
        "max_abs_q_change": float((after - before).abs().max()),
        "greedy_changed_rows": int(
            (before.argmax(-1) != after.argmax(-1)).sum()
        ),
        "economic_boundary_count": float(
            metrics["economic_boundary_count"]
        ),
        "economic_boundary_active_constraint_count": float(
            metrics["economic_boundary_active_constraint_count"]
        ),
        "economic_boundary_hard_constraint_count": float(
            metrics["economic_boundary_hard_constraint_count"]
        ),
        "economic_boundary_backtracks": float(
            metrics["economic_boundary_backtracks"]
        ),
        "economic_boundary_initial_min_margin_delta": float(
            metrics["economic_boundary_initial_min_margin_delta"]
        ),
        "economic_boundary_final_min_margin_delta": float(
            metrics["economic_boundary_final_min_margin_delta"]
        ),
        "economic_boundary_final_min_required_headroom": float(
            metrics["economic_boundary_final_min_required_headroom"]
        ),
        "economic_boundary_long_winner_min_margin_delta": float(
            metrics["economic_boundary_long_winner_min_margin_delta"]
        ),
        "economic_boundary_short_winner_min_margin_delta": float(
            metrics["economic_boundary_short_winner_min_margin_delta"]
        ),
        "economic_boundary_failed_long_min_margin_delta": float(
            metrics["economic_boundary_failed_long_min_margin_delta"]
        ),
        "economic_boundary_failed_short_min_margin_delta": float(
            metrics["economic_boundary_failed_short_min_margin_delta"]
        ),
        "grouped_margins_before": margins_before,
        "grouped_margins_after": margins_after,
        "grouped_margin_deltas": {
            name: margins_after[name] - value
            for name, value in margins_before.items()
        },
        "gradient_norm": float(metrics["gradient_norm"]),
        "gradient_conflict_primary_norm": float(
            metrics["gradient_conflict_primary_norm"]
        ),
        "gradient_conflict_safety_norm": float(
            metrics["gradient_conflict_safety_norm"]
        ),
        "gradient_conflict_opportunity_norm": float(
            metrics["gradient_conflict_opportunity_norm"]
        ),
    }


def configured_optimizer_overfit_probe(checkpoint, seqs, *, updates):
    """Prove whether the production learner can acquire one frozen pair batch."""
    if isinstance(updates, bool) or not isinstance(updates, int) or updates < 0:
        raise ValueError("optimizer overfit updates must be a nonnegative integer")
    agent, _ = RecurrentC51Agent.load(
        checkpoint,
        device="cpu",
        learner_backend_override="pytorch",
    )

    def margins():
        values = qtrace(agent, seqs)[0][:, 0, :3].detach().cpu()
        return grouped_economic_margins(
            values,
            seqs,
            recurrent_burn_in=agent.recurrent_burn_in,
        )

    before = margins()
    first_all_boundaries_satisfied = 0 if all(
        value >= agent.entry_action_margin for value in before.values()
    ) else None
    backtrack_updates = 0
    rejected_updates = 0
    for update in range(1, updates + 1):
        agent.train_batch(seqs)
        metrics = agent.last_train_metrics
        backtrack_updates += int(metrics["economic_boundary_backtracks"] > 0)
        rejected_updates += int(metrics["economic_boundary_backtracks"] >= 12)
        current = margins()
        if (
            first_all_boundaries_satisfied is None
            and all(
                value >= agent.entry_action_margin
                for value in current.values()
            )
        ):
            first_all_boundaries_satisfied = update
    after = margins()
    return {
        "updates": updates,
        "target_margin": agent.entry_action_margin,
        "margins_before": before,
        "margins_after": after,
        "first_all_boundaries_satisfied_update": (
            first_all_boundaries_satisfied
        ),
        "all_boundaries_satisfied": all(
            value >= agent.entry_action_margin for value in after.values()
        ),
        "backtrack_updates": backtrack_updates,
        "rejected_updates": rejected_updates,
    }


def parity(a, seqs):
    q, causal, resets = qtrace(a, seqs)
    with torch.no_grad():
        rec, _ = a._recurrent_features_with_resets(a.online, causal, resets)
        logits = a.online.distribution_logits(rec[:, a.recurrent_burn_in : -1]).float()
        full = (logits.softmax(-1) * a.support).sum(-1)
    obsok = []
    idxok = []
    for s in seqs:
        for x, y in zip(s, s[1:]):
            if x.training_valid and y.training_valid:
                obsok.append(np.array_equal(x.next_observation, y.observation))
                if (
                    x.source_decision_index is not None
                    and y.source_decision_index is not None
                ):
                    idxok.append(y.source_decision_index == x.source_decision_index + 1)
    return {
        "full_vs_split_max_abs_q": float((full - q).abs().max().detach()),
        "observation_contiguity_all": all(obsok),
        "decision_index_contiguity_all": all(idxok),
        "pair_anchor_indices": [
            [i for i, t in enumerate(s) if t.paired_a_plus_pair_id is not None]
            for s in seqs
        ],
    }


def main():
    ap = argparse.ArgumentParser(
        description="Audit one frozen recurrent checkpoint and authenticated replay batch."
    )
    ap.add_argument(
        "--attempt-dir",
        type=Path,
        help=(
            "Campaign attempt directory containing training-recovery.pt and "
            "training-replay; writes frozen-checkpoint-batch-audit.json there"
        ),
    )
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--replay-root", type=Path)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--near-blow-pnl", type=float, default=-2250.0)
    ap.add_argument("--pair-count", type=int, default=4)
    ap.add_argument(
        "--challenge-return-discount",
        type=float,
        default=0.5 ** (1.0 / (30 * 480)),
    )
    ap.add_argument("--challenge-return-weight", type=float, default=0.05)
    ap.add_argument("--optimizer-overfit-updates", type=int, default=0)
    ap.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    ap.add_argument(
        "--learner-backend-override",
        choices=("pytorch", "mlx"),
    )
    ap.add_argument(
        "--paired-context-matching",
        choices=(
            "static_expansion_regime_v1",
            "regime_control_expansion_lifecycle_v1",
            "expansion_trend_lifecycle_control_v2",
        ),
    )
    ap.add_argument(
        "--violation-prioritized-pairs-per-side",
        type=int,
        default=0,
    )
    args = ap.parse_args()
    if (
        not np.isfinite(args.near_blow_pnl)
        or args.near_blow_pnl >= 0
        or args.pair_count < 2
        or not np.isfinite(args.challenge_return_discount)
        or not 0.0 < args.challenge_return_discount <= 1.0
        or not np.isfinite(args.challenge_return_weight)
        or args.challenge_return_weight < 0.0
        or args.optimizer_overfit_updates < 0
        or args.violation_prioritized_pairs_per_side < 0
        or args.violation_prioritized_pairs_per_side > 0
        and args.paired_context_matching not in {
            "regime_control_expansion_lifecycle_v1",
            "expansion_trend_lifecycle_control_v2",
        }
    ):
        ap.error(
            "--near-blow-pnl must be negative and --pair-count must be at least two"
        )
    explicit_paths = (args.checkpoint, args.replay_root, args.output)
    if args.attempt_dir is not None:
        if any(path is not None for path in explicit_paths):
            ap.error(
                "--attempt-dir cannot be combined with --checkpoint, "
                "--replay-root, or --output"
            )
        checkpoint = args.attempt_dir / "training-recovery.pt"
        replay_root = args.attempt_dir / "training-replay"
        output = args.attempt_dir / "frozen-checkpoint-batch-audit.json"
    else:
        if any(path is None for path in explicit_paths):
            ap.error(
                "provide --attempt-dir or all of --checkpoint, --replay-root, "
                "and --output"
            )
        checkpoint = args.checkpoint
        replay_root = args.replay_root
        output = args.output

    agent, manifest = RecurrentC51Agent.load(
        checkpoint,
        device=args.device,
        learner_backend_override=args.learner_backend_override,
    )
    desc = manifest["replay_checkpoint"]
    contract = desc["contract"]
    population = Counter()
    violations = Counter()
    candidates = defaultdict(list)
    index = {}
    authenticated = 0
    total_bytes = 0
    challenge_cohort_returns = defaultdict(list)
    labeled_return_count = 0
    labeled_returns_above_support = 0
    labeled_returns_below_support = 0
    winner_dominant_chop_rows = 0
    winner_dominant_chop_mass = 0.0
    failure_dominant_chop_rows = 0
    failure_dominant_chop_mass = 0.0
    separability_rows = defaultdict(list)
    for number, entry in enumerate(desc["episodes"], 1):
        path = replay_root / entry["path"]
        digest = sha(path)
        authenticated += int(
            digest == entry["sha256"] and path.stat().st_size == entry["size_bytes"]
        )
        total_bytes += entry["size_bytes"]
        with path.open("rb") as f:
            p = pickle.load(f)
        index[p["episode_id"]] = entry
        targets = np.asarray(p["entry_action_targets"])
        sides = np.asarray(p["paired_a_plus_sides"])
        wins = np.asarray(p["paired_a_plus_economic_wins"])
        contexts = np.asarray(p["paired_a_plus_contexts"])
        rewards = np.asarray(p["rewards"], dtype=np.float64)
        returns_to_go = discounted_returns_to_go(
            rewards,
            discount=args.challenge_return_discount,
        )
        terminal_pnl = float(p["terminal_pnl"])
        cohort = challenge_outcome_cohort(
            str(p["outcome"]), terminal_pnl, args.near_blow_pnl
        )
        challenge_cohort_returns[cohort].append(float(returns_to_go[0]))
        labeled_rows = wins >= 0
        labeled_return_count += int(labeled_rows.sum())
        labeled_returns_above_support += int(
            (returns_to_go[labeled_rows] > agent.value_max).sum()
        )
        labeled_returns_below_support += int(
            (returns_to_go[labeled_rows] < agent.value_min).sum()
        )
        teacher_targets = np.asarray(p["teacher_targets"])
        if teacher_targets.shape[1] >= 7:
            chop_mass = np.maximum(
                0.0,
                teacher_targets[:, 4]
                - np.maximum(teacher_targets[:, 5], teacher_targets[:, 6]),
            )
            winner_rows = wins == 1
            failure_rows = wins == 0
            winner_dominant_chop_rows += int((winner_rows & (chop_mass > 0)).sum())
            winner_dominant_chop_mass += float(chop_mass[winner_rows].sum())
            failure_dominant_chop_rows += int((failure_rows & (chop_mass > 0)).sum())
            failure_dominant_chop_mass += float(chop_mass[failure_rows].sum())
        rows = (sides >= 0) | (wins >= 0)
        violations["partial_pair_evidence"] += int(np.sum((sides >= 0) != (wins >= 0)))
        violations["invalid_context"] += int(
            np.sum(
                rows
                & (
                    (~np.isfinite(contexts).all(1))
                    | (contexts < 0).any(1)
                    | (contexts > 1).any(1)
                )
            )
        )
        for side in SIDES:
            for win in (0, 1):
                mask = (sides == int(side)) & (wins == win)
                label = "winner" if win else "failure"
                population[f"{SN[side]}_{label}"] += int(mask.sum())
                expected = int(side) if win else 0
                violations[f"{SN[side]}_{label}_target_mismatch"] += int(
                    np.sum(mask & (targets != expected))
                )
                for ix in np.flatnonzero(mask & (targets == expected)):
                    try:
                        feature_families = causal_feature_families(
                            contexts[ix],
                            teacher_targets,
                            int(ix),
                            side,
                        )
                    except ValueError:
                        violations["invalid_causal_separability_row"] += 1
                        continue
                    for family, features in feature_families.items():
                        row = (
                            int(p["ended_at_ns"]),
                            int(win),
                            np.asarray(features, dtype=np.float64),
                        )
                        separability_rows[(SN[side], family)].append(row)
                        separability_rows[("pooled", family)].append(row)
                eligible = (
                    p["outcome"] == "pass"
                    if win
                    else (
                        p["outcome"] == "timeout"
                        and p["terminal_pnl"] is not None
                        and p["terminal_pnl"] <= args.near_blow_pnl
                    )
                )
                if eligible:
                    score = float(np.sum(p["rewards"], dtype=np.float64))
                    for ix in np.flatnonzero(mask):
                        candidates[(side, bool(win))].append(
                            (
                                int(p["ended_at_ns"]),
                                score,
                                p["episode_id"],
                                int(ix),
                                contexts[ix].copy(),
                                float(returns_to_go[ix]),
                            )
                        )
        del p
        print(
            f'[audit] authenticated_and_scanned={number}/{len(desc["episodes"])}',
            flush=True,
        )
    causal_separability = {}
    for (scope, family), rows in sorted(separability_rows.items()):
        try:
            result = temporal_ridge_separability(
                np.stack([row[2] for row in rows]),
                np.asarray([row[1] for row in rows], dtype=np.int8),
                np.asarray([row[0] for row in rows], dtype=np.int64),
            )
        except ValueError as error:
            causal_separability[f"{scope}.{family}"] = {
                "status": "invalid",
                "error": str(error),
                "rows": len(rows),
            }
        else:
            causal_separability[f"{scope}.{family}"] = result.to_dict()
    for key, v in candidates.items():
        win = key[1]
        v.sort(
            key=(lambda x: (-x[0], -x[1], x[2], x[3]))
            if win
            else (lambda x: (-x[0], x[1], x[2], x[3]))
        )
        candidates[key] = v[:8]
    specs = []
    selected = []
    for pid in range(args.pair_count):
        side = SIDES[pid % 2]
        ww = candidates[(side, True)]
        ff = candidates[(side, False)]
        if not ww or not ff:
            raise ValueError(f"missing frozen economic pool for {SN[side]}")
        w = ww[(pid // 2) % len(ww)]
        wc = w[4] if side == SIDES[0] else w[4][[2, 3, 0, 1, 4, 5, 6]]

        def dist(x):
            c = x[4] if side == SIDES[0] else x[4][[2, 3, 0, 1, 4, 5, 6]]
            return float(np.square(wc - c).sum()), x[2], x[3]

        f = min(ff, key=dist)
        specs.append((pid, side, w, f, dist(f)[0], len(ww), len(ff)))
        selected.extend((w[2], f[2]))
    payloads = []
    for eid in dict.fromkeys(selected):
        with (replay_root / index[eid]["path"]).open("rb") as f:
            payloads.append(pickle.load(f))
    replay = make_replay(
        contract,
        desc["replay_schema_version"],
        payloads,
        paired_a_plus_context_matching=args.paired_context_matching,
    )
    challenge_credit_sequences = replay.sample_balance_outcome_contrast_pairs(
        args.pair_count,
        near_blow_pnl=args.near_blow_pnl,
        max_examples=8,
        pair_id_start=1_000_000,
        challenge_return_discount=args.challenge_return_discount,
    )
    challenge_credit_coverage = challenge_return_credit_coverage(
        challenge_credit_sequences,
        recurrent_burn_in=contract["recurrent_burn_in"],
        n_step_return=contract["n_step_return"],
    )
    seqs = []
    pair_report = []
    for pid, side, w, f, d, nw, nf in specs:
        anchors = []
        weights = (2 * nw / (nw + nf), 2 * nf / (nw + nf))
        for item, win, weight in ((w, True, weights[0]), (f, False, weights[1])):
            ep = replay._episodes[item[2]]
            seq = list(
                ep.target_anchored_sequence(
                    anchor_index=item[3],
                    length=contract["sequence_length"],
                    recurrent_burn_in=contract["recurrent_burn_in"],
                    n_step_return=contract["n_step_return"],
                )
            )
            anchor = seq[contract["recurrent_burn_in"]]
            seq[contract["recurrent_burn_in"]] = replace(
                anchor,
                paired_a_plus_pair_id=pid,
                paired_a_plus_pair_side=side,
                paired_a_plus_population_weight=weight,
            )
            seqs.append(tuple(seq))
            anchors.append(
                {
                    "episode": item[2],
                    "source_index": item[3],
                    "economic_win": win,
                    "exact_target": Action(anchor.entry_action_target).name,
                    "population_weight": weight,
                    "challenge_return_to_go": item[5],
                }
            )
        pair_report.append(
            {
                "pair_id": pid,
                "side": SN[side],
                "context_squared_distance": d,
                "anchors": anchors,
            }
        )
    violation_diagnostic = None
    if args.violation_prioritized_pairs_per_side > 0:
        candidate_sequences = replay.sample_paired_a_plus_candidate_pairs(
            max(args.pair_count, args.violation_prioritized_pairs_per_side),
        )
        prioritized, violation_diagnostic = (
            _prioritize_paired_a_plus_violations(
                agent,
                candidate_sequences,
                pairs_per_side=args.violation_prioritized_pairs_per_side,
            )
        )
        seqs = list(prioritized)
        def source_payload(anchor):
            for payload in payloads:
                source_indices = np.asarray(payload["source_decision_indices"])
                side_rows = np.asarray(payload["paired_a_plus_sides"])
                win_rows = np.asarray(payload["paired_a_plus_economic_wins"])
                for source_index in np.flatnonzero(
                    (source_indices == anchor.source_decision_index)
                    & (side_rows == int(anchor.paired_a_plus_pair_side))
                    & (
                        win_rows
                        == int(bool(anchor.paired_a_plus_economic_win))
                    )
                ):
                    if np.array_equal(
                        payload["observations"][source_index],
                        anchor.observation,
                    ):
                        return payload, int(source_index)
            raise ValueError("prioritized pair source is unavailable")

        pair_report = []
        for pair_offset in range(0, len(seqs), 2):
            pair_sequences = seqs[pair_offset:pair_offset + 2]
            pair_anchors = [
                sequence[contract["recurrent_burn_in"]]
                for sequence in pair_sequences
            ]
            winner_anchor = next(
                anchor
                for anchor in pair_anchors
                if anchor.paired_a_plus_economic_win
            )
            failure_anchor = next(
                anchor
                for anchor in pair_anchors
                if not anchor.paired_a_plus_economic_win
            )
            side = Action(winner_anchor.paired_a_plus_pair_side)
            anchor_report = []
            matched = []
            for anchor in (winner_anchor, failure_anchor):
                payload, source_index = source_payload(anchor)
                episode = replay._episodes[payload["episode_id"]]
                matched.append(replay._paired_match_context(
                    episode,
                    source_index,
                    side,
                ))
                anchor_report.append({
                    "episode": payload["episode_id"],
                    "source_index": source_index,
                    "economic_win": bool(anchor.paired_a_plus_economic_win),
                    "exact_target": Action(anchor.entry_action_target).name,
                    "population_weight": float(
                        anchor.paired_a_plus_population_weight
                    ),
                    "challenge_return_to_go": float(
                        discounted_returns_to_go(
                            np.asarray(payload["rewards"], dtype=np.float64),
                            discount=args.challenge_return_discount,
                        )[source_index]
                    ),
                })
            control_distance = float(
                np.square(matched[0][0] - matched[1][0]).sum()
            )
            regime_slice = (
                slice(4, 7)
                if args.paired_context_matching
                == "expansion_trend_lifecycle_control_v2"
                else slice(None)
            )
            regime_distance = float(np.square(
                matched[0][0][regime_slice] - matched[1][0][regime_slice]
            ).sum())
            lifecycle = (
                matched[0][1] if len(matched[0]) > 1 else matched[0][0]
            )
            failure_lifecycle = (
                matched[1][1] if len(matched[1]) > 1 else matched[1][0]
            )
            pair_report.append({
                "pair_id": int(winner_anchor.paired_a_plus_pair_id),
                "side": SN[side],
                "context_squared_distance": float(
                    np.square(lifecycle - failure_lifecycle).sum()
                ),
                "control_squared_distance": control_distance,
                "regime_squared_distance": regime_distance,
                "anchors": anchor_report,
            })
    seqs = tuple(seqs)
    before = qtrace(agent, seqs)[0][:, 0, :3].detach().cpu()
    boundary_margins_before = []
    for pair_index, pair in enumerate(pair_report):
        side = SIDES[0] if pair["side"] == "long" else SIDES[1]
        opposite = SIDES[1] if side == SIDES[0] else SIDES[0]
        winner_q = before[pair_index * 2]
        failure_q = before[pair_index * 2 + 1]
        boundary_margins_before.extend((
            {
                "name": f"paired_{pair_index}_winner_over_wait",
                "margin": float(winner_q[int(side)] - winner_q[0]),
            },
            {
                "name": f"paired_{pair_index}_winner_over_opposite",
                "margin": float(
                    winner_q[int(side)] - winner_q[int(opposite)]
                ),
            },
            {
                "name": f"paired_{pair_index}_wait_over_failure",
                "margin": float(failure_q[0] - failure_q[int(side)]),
            },
        ))
    td, tdloss = tdgrad(checkpoint, seqs)
    ea, _ = RecurrentC51Agent.load(
        checkpoint,
        device="cpu",
        learner_backend_override="pytorch",
    )
    eg, el = gradloss(ea, lambda a: exactloss(a, seqs))
    wg, wl, wrows = waitgrad(checkpoint, seqs, td)
    grads = {"c51_td": td, "exact_action": eg, "chop_wait": wg}
    losses = {"c51_td": tdloss, "exact_action": el, "chop_wait": wl}
    for name in (
        "paired_relative",
        "paired_winner",
        "paired_failure",
        "balance_outcome_total",
    ):
        pa, _ = RecurrentC51Agent.load(
            checkpoint,
            device="cpu",
            learner_backend_override="pytorch",
        )
        g, l = gradloss(pa, lambda a, n=name: pairlosses(a, seqs)[n])
        grads[name] = g
        losses[name] = l
    norms = {n: {"norm": float(g.norm()), "loss": losses[n]} for n, g in grads.items()}
    cos = {n: {m: cosine(g, h) for m, h in grads.items()} for n, g in grads.items()}
    changes = {}
    for name, g in grads.items():
        ca, _ = RecurrentC51Agent.load(
            checkpoint,
            device="cpu",
            learner_backend_override="pytorch",
        )
        applygrad(ca, g)
        after = qtrace(ca, seqs)[0][:, 0, :3].detach().cpu()
        ba = before.argmax(-1)
        aa = after.argmax(-1)
        changes[name] = {
            "changed_rows": int((ba != aa).sum()),
            "total_rows": len(seqs),
            "changes": dict(
                Counter(
                    f"{Action(int(x)).name}->{Action(int(y)).name}"
                    for x, y in zip(ba, aa)
                    if x != y
                )
            ),
            "mean_abs_q_change": float((after - before).abs().mean()),
            "max_abs_q_change": float((after - before).abs().max()),
        }
    par = parity(agent, seqs)
    rt, _ = RecurrentC51Agent.load(
        checkpoint,
        device="cpu",
        learner_backend_override="pytorch",
    )
    rtq = qtrace(rt, seqs)[0][:, 0, :3].detach().cpu()
    par["checkpoint_roundtrip_max_abs_q"] = float((rtq - before).abs().max())
    configured_update = actual_configured_update(checkpoint, seqs, before)
    optimizer_overfit_probe = configured_optimizer_overfit_probe(
        checkpoint,
        seqs,
        updates=args.optimizer_overfit_updates,
    )
    challenge_bonuses = []
    for row_index, pair in enumerate(pair_report):
        winner = pair["anchors"][0]
        side = SIDES[0] if pair["side"] == "long" else SIDES[1]
        values = before[row_index * 2]
        improvement = max(
            float(winner["challenge_return_to_go"]),
            float(values[int(side)]),
        ) - float(values.max())
        challenge_bonuses.append(
            args.challenge_return_weight * max(0.0, improvement)
        )
    report = {
        "checkpoint": str(checkpoint),
        "completed_episodes": manifest["progress"]["completed_episodes"],
        "resume_identity": manifest["resume_identity"],
        "audit_contract": {
            "near_blow_pnl": args.near_blow_pnl,
            "pair_count": args.pair_count,
            "paired_context_matching": (
                args.paired_context_matching
                or contract.get(
                    "paired_a_plus_context_matching",
                    "static_expansion_regime_v1",
                )
            ),
            "violation_prioritized_pairs_per_side": (
                args.violation_prioritized_pairs_per_side
            ),
        },
        "replay_authentication": {
            "verified_shards": authenticated,
            "declared_shards": len(desc["episodes"]),
            "total_bytes": total_bytes,
        },
        "economic_label_population": dict(population),
        "economic_label_violations": dict(violations),
        "causal_winner_failure_separability": causal_separability,
        "challenge_return_audit": {
            "discount": args.challenge_return_discount,
            "weight": args.challenge_return_weight,
            "c51_support": [agent.value_min, agent.value_max],
            "cohorts": {
                name: {
                    "episodes": len(values),
                    "mean_episode_return": float(np.mean(values)),
                    "minimum_episode_return": float(np.min(values)),
                    "maximum_episode_return": float(np.max(values)),
                }
                for name, values in sorted(challenge_cohort_returns.items())
            },
            "labeled_anchor_count": labeled_return_count,
            "labeled_returns_above_support": labeled_returns_above_support,
            "labeled_returns_below_support": labeled_returns_below_support,
            "audited_pass_winner_bonus_count": len(challenge_bonuses),
            "audited_pass_winner_bonus_mean": float(np.mean(challenge_bonuses)),
            "audited_pass_winner_bonus_max": float(np.max(challenge_bonuses)),
            "production_replay_credit_coverage": challenge_credit_coverage,
        },
        "economic_winner_chop_conflict": {
            "legacy_winner_rows": winner_dominant_chop_rows,
            "legacy_winner_membership_mass": winner_dominant_chop_mass,
            "corrected_winner_rows": 0,
            "corrected_winner_membership_mass": 0.0,
            "retained_failure_rows": failure_dominant_chop_rows,
            "retained_failure_membership_mass": failure_dominant_chop_mass,
        },
        "candidate_pool": {
            f'{SN[k[0]]}_{"winner" if k[1] else "failure"}': len(v)
            for k, v in candidates.items()
        },
        "audited_pairs": pair_report,
        "economic_boundary_margins_before": boundary_margins_before,
        "audited_pair_mass": dict(Counter(x["side"] for x in pair_report)),
        "violation_prioritization": violation_diagnostic,
        "gradient_components": norms,
        "gradient_cosine": cos,
        "chop_wait_rows": wrows,
        "greedy_action_changes_one_sgd_step_at_configured_lr": changes,
        "actual_configured_optimizer_update": configured_update,
        "configured_optimizer_overfit_probe": optimizer_overfit_probe,
        "recurrent_parity": par,
        "probe_greedy_before": dict(
            Counter(Action(int(x)).name for x in before.argmax(-1))
        ),
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
