"""PPO search for a policy that passes the $6,000 challenge without touching -$3,000.

The reasoning model is the product, but it cannot be its own searcher: at ~0.3s a
decision it gets ~3,200 learning samples where algoTraderAI's PPO had ~2,000,000, and
the first direct run diverged on a 4-episode leave-one-out baseline. This policy runs an
episode in ~2.4s over the same evidence, in the same environment, so it can do the
search. What it finds is then taught to the reasoning model, which has strictly more to
reason with -- Chronos embeddings and distilled teacher knowledge the 54 floats do not
carry -- and so has room to generalise BEYOND this teacher rather than merely clone it.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch

from propevolve.reasoning_policy.job import read_job, load_source_contract, load_role
from propevolve.reasoning_policy.compact_rl import (
    ACTION_ORDER, ActorCritic, challenge_objective, compute_gae, masked_categorical,
    ppo_update, rollout_episode)
from propevolve.reasoning_policy.evidence import EVIDENCE_FIELDS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/reasoning/expansion_flow_job.json")
    ap.add_argument("--iterations", type=int, default=60)
    ap.add_argument("--episodes", type=int, default=48, help="episodes per iteration")
    ap.add_argument("--greedy-episodes", type=int, default=40)
    ap.add_argument("--greedy-every", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--entropy", type=float, default=0.02)
    ap.add_argument("--hold-bias", type=float, default=2.0)
    ap.add_argument("--enter-bias", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", default="runs/reasoning-challenger/expansion-flow/compact-rl")
    args = ap.parse_args()

    cfg, root = read_job(args.config)
    src, _, _, _, identity = load_source_contract(cfg, root)
    env, sources = load_role(cfg, root, src, "train", include_specialists=True)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    net = ActorCritic(len(EVIDENCE_FIELDS), len(ACTION_ORDER), seed=args.seed,
                      action_bias={"HOLD": args.hold_bias,
                                   "ENTER_LONG_1": args.enter_bias,
                                   "ENTER_SHORT_1": args.enter_bias})
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)
    best = -9e9
    history = []

    for it in range(args.iterations):
        t0 = time.time()
        S, M, A, LP, ADV, RET = [], [], [], [], [], []
        outcomes, pnls = [], []
        for _ in range(args.episodes):
            ep = rollout_episode(net, env, sources, ticker="NQ",
                                 options={"ticker": "NQ"}, rng=rng)
            if not ep["actions"]:
                continue
            adv, ret = compute_gae(ep["rewards"], ep["values"], last_value=0.0)
            S += ep["states"]; M += ep["masks"]; A += ep["actions"]
            LP += ep["log_probs"]; ADV += adv; RET += ret
            outcomes.append(ep["outcome"]); pnls.append(ep["realized_pnl"])
        n = len(A)
        if n == 0:
            print(json.dumps({"iteration": it, "error": "no decisions"})); continue
        S = torch.tensor(np.array(S), dtype=torch.float32)
        M = torch.tensor(np.array(M), dtype=torch.bool)
        A = torch.tensor(A, dtype=torch.long)
        LP = torch.tensor(LP, dtype=torch.float32)
        ADV = torch.tensor(ADV, dtype=torch.float32)
        RET = torch.tensor(RET, dtype=torch.float32)
        stats = {}
        for _ in range(args.epochs):
            perm = torch.randperm(n)
            for i in range(0, n, args.minibatch):
                b = perm[i:i + args.minibatch]
                stats = ppo_update(net, opt, S[b], M[b], A[b], LP[b], ADV[b], RET[b],
                                   entropy_coef=args.entropy)
        # Evaluate GREEDILY as well. The sampled rate hid a policy whose argmax was
        # "always WAIT": it scored 12.5% pass by exploration while its deterministic
        # behaviour made zero trades and zero dollars. Only the greedy number says what
        # was actually learned, and it is the one that would ship.
        # Selecting on 8 greedy episodes optimised NOISE: a checkpoint saved at 62.5%
        # measured 32.5% over 40 episodes. Evaluate properly, less often.
        do_greedy = (it % args.greedy_every == 0)
        g_out = []; g_head = []
        for _ in range(args.greedy_episodes if do_greedy else 0):
            g = rollout_episode(net, env, sources, ticker="NQ",
                                options={"ticker": "NQ"}, rng=rng, greedy=True)
            g_out.append((g["outcome"] or "timeout", g["realized_pnl"]))
            g_head.append(g["min_headroom"])
        greedy_pass = (sum(o == "pass" for o, _ in g_out) / len(g_out)) if g_out else None
        greedy_pnl = float(np.mean([p for _, p in g_out])) if g_out else None
        greedy_near = (float(np.mean([h < 0.10 * env.spec.max_loss for h in g_head]))
                       if g_head else None)
        rates = {k: outcomes.count(k) / len(outcomes) for k in ("pass", "blow", "timeout")}
        # Only a real greedy measurement can move the checkpoint, and a fragile
        # policy cannot win: near-blow above the 10% gate is subtracted outright.
        if greedy_pass is None:
            score = None
        else:
            score = greedy_pass - max(0.0, greedy_near - 0.10) * 2.0
        row = {"iteration": it, "episodes": len(outcomes), "decisions": n,
               "pass": round(rates["pass"], 4), "blow": round(rates["blow"], 4),
               "timeout": round(rates["timeout"], 4),
               "score": (round(score, 4) if score is not None else None),
               "mean_pnl": round(float(np.mean(pnls)), 1),
               "max_pnl": round(float(np.max(pnls)), 1),
               "greedy_pass": greedy_pass, "greedy_pnl": greedy_pnl,
               "greedy_near": greedy_near,
               "secs": round(time.time() - t0, 1), **{k: round(v, 4) for k, v in stats.items()}}
        history.append(row)
        print(json.dumps(row), flush=True)
        if score is not None and score > best:
            best = score
            torch.save(net.state_dict(), out / "best.pt")
            (out / "best.json").write_text(json.dumps({**row, "source_identity": identity}, indent=2))
        (out / "history.jsonl").write_text("\n".join(json.dumps(r) for r in history) + "\n")
    print(json.dumps({"done": True, "best_score": round(best, 4)}))


if __name__ == "__main__":
    raise SystemExit(main())
