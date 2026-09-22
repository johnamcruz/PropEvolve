"""Search for a configuration that hits 60% pass with ZERO blow.

The target is explicit: blow rate 0, pass rate >= 60%, and balances that finish near the
$6,000 line rather than scraping the -$3,000 floor. Everything measured so far falls
short on pass (32.5-42.5%) while already achieving zero blow, so this searches the three
levers that move pass rate together rather than one at a time:

  per_trade_risk_dollars  -- at $300 against a $6,000 target you need ~20 net winning R,
                             which is why mean P&L is $1,280 and only ~40% get there.
                             Larger risk reaches the target in fewer trades but shrinks
                             the consecutive-loss budget (10 losses at $300, 5 at $600).
  ratchet                 -- 2R/0.5R cut winners to $55/trade; loose lets trends run.
  headroom skip           -- one rule took near-blow from 45% to 0.0% for 5 points of pass.

Scored lexicographically: any blow is disqualifying, then pass rate, with near-blow above
the 10% acceptance gate subtracted. A config that passes by grazing the floor does not win.
"""
import argparse, dataclasses, json, warnings
from pathlib import Path
import numpy as np
import optuna
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

from propevolve.reasoning_policy.job import read_job, load_source_contract, load_role
from propevolve.reasoning_policy.rl import decision_is_forced
from propevolve.decision import Action
from propevolve.environment import HistoricalChallengeEnv


def evaluate(base, over, skip_headroom, episodes, seed=7):
    spec = dataclasses.replace(base.spec, **over)
    env = HistoricalChallengeEnv(base.markets, tick_values=base.tick_values,
        round_trip_fees=base.round_trip_fees, spec=spec,
        observation_spec=base._assembler.trade_management,
        setup_signals=base.setup_signals, seed=seed)
    outs = {"pass": 0, "blow": 0, "timeout": 0}; pnls = []; heads = []
    for _ in range(episodes):
        _, info = env.reset()
        for _ in range(20000):
            if decision_is_forced(env):
                a = Action.WAIT
            else:
                legal = [Action(x) for x in info["valid_actions"]]
                if Action.ENTER_LONG_1 in legal:
                    st = env._account_state()
                    a = Action.WAIT if st.mll_headroom < skip_headroom else Action.ENTER_LONG_1
                elif Action.HOLD in legal:
                    a = Action.HOLD
                else:
                    a = legal[0]
            _, _, t, u, info = env.step(a)
            if t or u:
                break
        o = info.get("outcome", "timeout"); outs[o] = outs.get(o, 0) + 1
        pnls.append(env._account.realized_pnl); heads.append(env._minimum_mll_headroom)
    n = sum(outs.values())
    near = float(np.mean([h < 0.10 * spec.max_loss for h in heads]))
    return {k: v / n for k, v in outs.items()}, float(np.mean(pnls)), near


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=45)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--out", default="runs/reasoning-challenger/expansion-flow/passrate-scan")
    args = ap.parse_args()
    cfg, root = read_job("config/reasoning/expansion_flow_job.json")
    src, _, _, _, _ = load_source_contract(cfg, root)
    base, _ = load_role(cfg, root, src, "train", include_specialists=False)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows = []

    def objective(trial):
        risk = trial.suggest_float("per_trade_risk_dollars", 300.0, 900.0, step=50.0)
        act = trial.suggest_float("ratchet_activation_r", 4.0, 16.0, step=0.5)
        give = trial.suggest_float("ratchet_giveback_r", 1.0, act - 0.5, step=0.5)
        floor = trial.suggest_float("ratchet_lock_floor_r", 0.0, 3.0, step=0.5)
        skip = trial.suggest_float("skip_headroom", 0.0, 1500.0, step=150.0)
        daily = trial.suggest_categorical("daily_loss_limit_dollars", [None, 750.0, 1000.0, 1500.0])
        over = {"per_trade_risk_dollars": risk, "ratchet_activation_r": act,
                "ratchet_giveback_r": give, "ratchet_lock_floor_r": floor,
                "daily_loss_limit_dollars": daily}
        rates, pnl, near = evaluate(base, over, skip, args.episodes)
        if rates["blow"] > 0:
            score = -1.0 - rates["blow"]           # any blow is disqualifying
        else:
            score = rates["pass"] - max(0.0, near - 0.10) * 2.0
        row = {"trial": trial.number, **over, "skip_headroom": skip,
               "pass": round(rates["pass"], 4), "blow": round(rates["blow"], 4),
               "near": round(near, 4), "mean_pnl": round(pnl, 1), "score": round(score, 4)}
        rows.append(row); print(json.dumps(row), flush=True)
        (out / "trials.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        return score

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=23))
    study.optimize(objective, n_trials=args.trials)
    best = max(rows, key=lambda r: r["score"])
    (out / "best.json").write_text(json.dumps(best, indent=2))
    print(json.dumps({"BEST": best}))


if __name__ == "__main__":
    raise SystemExit(main())
