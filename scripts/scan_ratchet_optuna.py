"""Optuna scan of the execution-risk settings, scored zero-blow-first.

Measured on a trivial always-enter/always-hold policy, the shipped 2R/0.5R ratchet cut
winners to $55 a trade and 8.3% pass while 10R/9R gave $484 and 58.3% -- but 10R/9R also
dropped from 19.8 trades an episode to 5.5, which means the ratchet essentially never
engages and there is no trailing protection at all. A 12-episode read on 5.5 trades is
thin, so this scans the space with enough episodes to tell a real setting from variance.

The policy is deliberately trivial: this measures the ENVIRONMENT's execution settings,
not a policy. Whatever wins here is the ground the RL search then stands on.
"""
import argparse, dataclasses, json, warnings
from pathlib import Path
import numpy as np
import optuna
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

from propevolve.reasoning_policy.job import read_job, load_source_contract, load_role
from propevolve.reasoning_policy.rl import decision_is_forced
from propevolve.reasoning_policy.compact_rl import challenge_objective
from propevolve.decision import Action
from propevolve.environment import HistoricalChallengeEnv

BASE = {}

def evaluate(base_env, over, episodes, seed, session_exit=0.0):
    spec = dataclasses.replace(base_env.spec, **over)
    env = HistoricalChallengeEnv(base_env.markets, tick_values=base_env.tick_values,
        round_trip_fees=base_env.round_trip_fees, spec=spec,
        observation_spec=base_env._assembler.trade_management,
        setup_signals=base_env.setup_signals, seed=seed)
    outs = {"pass": 0, "blow": 0, "timeout": 0}; pnls = []; trades = []; headrooms = []
    for _ in range(episodes):
        _, info = env.reset(); tr = 0; was = None
        for _ in range(20000):
            if decision_is_forced(env):
                a = Action.WAIT
            else:
                legal = [Action(x) for x in info["valid_actions"]]
                if Action.ENTER_LONG_1 in legal:
                    a = Action.ENTER_LONG_1
                elif Action.HOLD in legal:
                    closing = (session_exit > 0
                               and env._account_state().session_remaining < session_exit)
                    a = Action.CLOSE if closing else Action.HOLD
                else:
                    a = legal[0]
            _, _, t, u, info = env.step(a)
            now = env._position
            if was is None and now is not None: tr += 1
            was = now
            if t or u: break
        o = info.get("outcome", "timeout"); outs[o] = outs.get(o, 0) + 1
        pnls.append(env._account.realized_pnl); trades.append(tr)
        headrooms.append(float(env._minimum_mll_headroom))
    n = sum(outs.values())
    rates = {k: v / n for k, v in outs.items()}
    # A config that passes 30% by nearly blowing the other 70% must not outrank a
    # robust one: trial 8 scored 30% pass with mean P&L of exactly $0, i.e. losers
    # ending near -$2,571 against a $3,000 floor. evaluation_metrics.json already
    # demands maximum_near_blow_rate 0.1; the scan has to honour it too.
    near = float(np.mean([h < 0.10 * base_env.spec.max_loss for h in headrooms]))
    return rates, float(np.mean(pnls)), float(np.mean(trades)), near


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--out", default="runs/reasoning-challenger/expansion-flow/ratchet-scan")
    args = ap.parse_args()
    cfg, root = read_job("config/reasoning/expansion_flow_job.json")
    src, _, _, _, _ = load_source_contract(cfg, root)
    base_env, _ = load_role(cfg, root, src, "train", include_specialists=False)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows = []

    def objective(trial):
        # Widened: the first scan converged on activation 12.0, the ceiling of the old
        # box, so the optimum was outside it. Session exit is a joint parameter because
        # closing daily restored frequency (6.5 -> 9.7 trades) at the same $/trade and
        # lifted mean P&L 48%, even though it cost pass rate -- the two objectives
        # genuinely diverge and the scan should weigh them together.
        act = trial.suggest_float("ratchet_activation_r", 2.0, 16.0, step=0.5)
        give = trial.suggest_float("ratchet_giveback_r", 0.5, act - 0.5, step=0.5)
        floor = trial.suggest_float("ratchet_lock_floor_r", 0.0, min(act, 3.0), step=0.5)
        daily = trial.suggest_categorical("daily_loss_limit_dollars", [None, 600.0, 750.0, 1000.0, 1500.0])
        session_exit = trial.suggest_categorical("session_exit_fraction", [0.0, 0.10, 0.25])
        over = {"ratchet_activation_r": act, "ratchet_giveback_r": give,
                "ratchet_lock_floor_r": floor, "daily_loss_limit_dollars": daily}
        rates, pnl, trades, near = evaluate(base_env, over, args.episodes, seed=7,
                                           session_exit=session_exit)
        score = challenge_objective(rates)
        if near > 0.10:
            score -= (near - 0.10)   # fragile configs rank below robust ones
        row = {"trial": trial.number, **over, "session_exit_fraction": session_exit,
               "pass": round(rates["pass"], 4),
               "blow": round(rates["blow"], 4), "score": round(score, 4),
               "mean_pnl": round(pnl, 1), "trades": round(trades, 1),
               "near_blow": round(near, 3)}
        rows.append(row); print(json.dumps(row), flush=True)
        (out / "trials.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        return score

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=17))
    study.optimize(objective, n_trials=args.trials)
    best = max(rows, key=lambda r: (r["score"], r["mean_pnl"]))
    (out / "best.json").write_text(json.dumps(best, indent=2))
    print(json.dumps({"best": best}))


if __name__ == "__main__":
    raise SystemExit(main())
