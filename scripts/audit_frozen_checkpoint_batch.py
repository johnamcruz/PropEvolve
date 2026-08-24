from __future__ import annotations
import argparse, hashlib, json, pickle, random
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from propevolve.agent import RecurrentC51Agent, exact_action_margin_losses
from propevolve.decision import Action
from propevolve.replay import BalancedSequenceReplay

SIDES = (Action.ENTER_LONG_1, Action.ENTER_SHORT_1)
SN = {SIDES[0]: "long", SIDES[1]: "short"}


def sha(path):
    d = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            d.update(chunk)
    return d.hexdigest()


def make_replay(contract, schema, payloads):
    r = BalancedSequenceReplay(
        capacity_episodes=contract["capacity_episodes"],
        capacity_transitions=contract["capacity_transitions"],
        sequence_length=contract["sequence_length"],
        terminal_sequence_fraction=contract["terminal_sequence_fraction"],
        safety_sequence_fraction=contract["safety_sequence_fraction"],
        entry_opportunity_sequence_fraction=contract[
            "entry_opportunity_sequence_fraction"
        ],
        regime_wait_sequence_fraction=contract["regime_wait_sequence_fraction"],
        regime_wait_sequence_update_period=contract[
            "regime_wait_sequence_update_period"
        ],
        entry_opportunity_side_balance=contract["entry_opportunity_side_balance"],
        recurrent_burn_in=contract["recurrent_burn_in"],
        n_step_return=contract["n_step_return"],
        seed=314159,
    )
    r.load_state_dict(
        {
            "schema_version": schema,
            "contract": contract,
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
    a, _ = RecurrentC51Agent.load(checkpoint, device="cpu")
    a.policy_retention_loss_weight = 0.0
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
    a, _ = RecurrentC51Agent.load(checkpoint, device="cpu")
    a.policy_retention_loss_weight = 0.0
    a.entry_action_loss_weight = 0.0
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
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--replay-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--near-blow-pnl", type=float, default=-2250.0)
    ap.add_argument("--pair-count", type=int, default=4)
    args = ap.parse_args()
    if (
        not np.isfinite(args.near_blow_pnl)
        or args.near_blow_pnl >= 0
        or args.pair_count < 2
    ):
        ap.error(
            "--near-blow-pnl must be negative and --pair-count must be at least two"
        )
    agent, manifest = RecurrentC51Agent.load(args.checkpoint, device="cpu")
    desc = manifest["replay_checkpoint"]
    contract = desc["contract"]
    population = Counter()
    violations = Counter()
    candidates = defaultdict(list)
    index = {}
    authenticated = 0
    total_bytes = 0
    for number, entry in enumerate(desc["episodes"], 1):
        path = args.replay_root / entry["path"]
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
                            )
                        )
        del p
        print(
            f'[audit] authenticated_and_scanned={number}/{len(desc["episodes"])}',
            flush=True,
        )
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
        with (args.replay_root / index[eid]["path"]).open("rb") as f:
            payloads.append(pickle.load(f))
    replay = make_replay(contract, desc["replay_schema_version"], payloads)
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
    seqs = tuple(seqs)
    before = qtrace(agent, seqs)[0][:, 0, :3].detach().cpu()
    td, tdloss = tdgrad(args.checkpoint, seqs)
    ea, _ = RecurrentC51Agent.load(args.checkpoint, device="cpu")
    eg, el = gradloss(ea, lambda a: exactloss(a, seqs))
    wg, wl, wrows = waitgrad(args.checkpoint, seqs, td)
    grads = {"c51_td": td, "exact_action": eg, "chop_wait": wg}
    losses = {"c51_td": tdloss, "exact_action": el, "chop_wait": wl}
    for name in (
        "paired_relative",
        "paired_winner",
        "paired_failure",
        "balance_outcome_total",
    ):
        pa, _ = RecurrentC51Agent.load(args.checkpoint, device="cpu")
        g, l = gradloss(pa, lambda a, n=name: pairlosses(a, seqs)[n])
        grads[name] = g
        losses[name] = l
    norms = {n: {"norm": float(g.norm()), "loss": losses[n]} for n, g in grads.items()}
    cos = {n: {m: cosine(g, h) for m, h in grads.items()} for n, g in grads.items()}
    changes = {}
    for name, g in grads.items():
        ca, _ = RecurrentC51Agent.load(args.checkpoint, device="cpu")
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
    rt, _ = RecurrentC51Agent.load(args.checkpoint, device="cpu")
    rtq = qtrace(rt, seqs)[0][:, 0, :3].detach().cpu()
    par["checkpoint_roundtrip_max_abs_q"] = float((rtq - before).abs().max())
    report = {
        "checkpoint": str(args.checkpoint),
        "completed_episodes": manifest["progress"]["completed_episodes"],
        "resume_identity": manifest["resume_identity"],
        "audit_contract": {
            "near_blow_pnl": args.near_blow_pnl,
            "pair_count": args.pair_count,
        },
        "replay_authentication": {
            "verified_shards": authenticated,
            "declared_shards": len(desc["episodes"]),
            "total_bytes": total_bytes,
        },
        "economic_label_population": dict(population),
        "economic_label_violations": dict(violations),
        "candidate_pool": {
            f'{SN[k[0]]}_{"winner" if k[1] else "failure"}': len(v)
            for k, v in candidates.items()
        },
        "audited_pairs": pair_report,
        "audited_pair_mass": dict(Counter(x["side"] for x in pair_report)),
        "gradient_components": norms,
        "gradient_cosine": cos,
        "chop_wait_rows": wrows,
        "greedy_action_changes_one_sgd_step_at_configured_lr": changes,
        "recurrent_parity": par,
        "probe_greedy_before": dict(
            Counter(Action(int(x)).name for x in before.argmax(-1))
        ),
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
