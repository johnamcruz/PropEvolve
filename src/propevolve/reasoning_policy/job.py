"""Explicit JSON-driven reasoning collection, QLoRA and simulator evaluation.

No campaign promotion, training on import, fabricated audits, or implicit model
downloads. Each command is an explicit stage. A readiness check is metadata-only.
"""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from .integrity import file_digest


def read_job(path):
    from .model_config import read_recipe
    config = read_recipe(path)
    root = Path(config["workspace_root"]).resolve()
    return config, root


def resolve(root, value):
    return root / value


def readiness(path):
    """Report missing resources without importing MLX or loading source arrays."""
    config, root = read_job(path)
    blockers = []
    for key in ("source_recipe", "temporal_split_audit", "context_config", "sft_config", "rl_config",
                "evaluation_policy_config", "evaluation_metrics_config"):
        if not config.get(key) or not resolve(root, config[key]).is_file():
            blockers.append(key)
    policy = config.get("collection_policy", {})
    if not policy.get("checkpoint") or not resolve(root, policy["checkpoint"]).is_file():
        blockers.append("collection_policy.checkpoint")
    if not policy.get("sha256"):
        blockers.append("collection_policy.sha256")
    if importlib.util.find_spec("mlx_lm") is None:
        blockers.append("mlx_lm")
    for role in ("train", "valid"):
        if not config.get("episodes", {}).get(role):
            blockers.append(f"episodes.{role}")
    if not config.get("evaluation_episodes"):
        blockers.append("evaluation_episodes")
    if config.get("volume_source") is not None:
        for key in ("manifest", "audit"):
            value = config["volume_source"].get(key)
            if not value or not resolve(root, value).is_file():
                blockers.append(f"volume_source.{key}")
    return {"ready": not blockers, "blockers": blockers,
            "scope": "resource presence only; causal and runtime checks still required",
            "training_started": False}


def load_source_contract(config, root):
    """Reuse source recipe defaults, not the C51 campaign runner or its gates."""
    from ..config import materialize_effective_config
    source_path = resolve(root, config["source_recipe"])
    source = materialize_effective_config(json.loads(source_path.read_text()))
    audit = json.loads(resolve(root, config["temporal_split_audit"]).read_text())
    if (audit.get("status") != "PASS"
            or audit.get("source_recipe_sha256") != file_digest(source_path)
            or audit.get("specialist_score_mode") not in {"out_of_fold", "post_fit"}
            or audit.get("sealed_touched") is not False):
        raise ValueError("source needs a matching passed fold-safe temporal audit")
    temporal = source["temporal"]
    bounds = {role: (temporal[f"{prefix}_start"], temporal[f"{prefix}_end"])
              for role, prefix in (("train", "train"), ("valid", "validation"))}
    ns = lambda value: int(np.datetime64(value, "ns").astype(np.int64))
    splits = {role: [ns(start), ns(end)] for role, (start, end) in bounds.items()}
    sealed = ns(config["sealed_start"])
    if (ns(temporal["sealed_start"]) != sealed
            or any(start >= end or end > sealed for start, end in splits.values())
            or splits["train"][1] > splits["valid"][0]
            or audit.get("splits") != splits):
        raise ValueError("source temporal roles conflict with reviewed/sealed boundaries")
    # Bind effective values too: editing inherited defaults must invalidate audit.
    import hashlib
    identity = hashlib.sha256(json.dumps(source, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if audit.get("effective_source_sha256") != identity:
        raise ValueError("effective source settings changed after audit")
    return source, audit, splits, sealed, identity


def load_role(config, root, source, role, *, include_specialists=True):
    from ..assets import AssetContract
    from ..cache import load_market_series
    from ..environment import ChallengeSpec, HistoricalChallengeEnv
    from ..observation import TradeManagementObservationSpec
    from ..teachers.composition import load_teacher_targets

    assets = AssetContract.load(resolve(root, source["assets"]))
    assets.verify()
    prefix = "train" if role == "train" else "validation"
    temporal = source["temporal"]
    markets = {ticker: load_market_series(
        Path(assets.market_data) / f"{ticker}_{source['timeframe_minutes']}min.csv",
        resolve(root, source["cache_root"]) / ticker, ticker=ticker,
        start=temporal[f"{prefix}_start"], end=temporal[f"{prefix}_end"],
    ) for ticker in config["tickers"][role]}
    sources = (load_teacher_targets(tuple(source["teachers"]), root=root, markets=markets).sources
               if include_specialists else ())
    if include_specialists and config.get("volume_source") is not None:
        from .specialist_cache import load_specialist_cache
        from .context import ContextConfig
        volume_config = config["volume_source"]
        volume = load_specialist_cache(resolve(root, volume_config["manifest"]),
            audit_path=resolve(root, volume_config["audit"]), markets=markets)
        if volume.kind != "volume":
            raise ValueError("volume_source must identify Volume evidence")
        fields = ContextConfig.load(resolve(root, config["context_config"])).fields
        if not any(field.startswith("volume.") for field in fields):
            raise ValueError("configured Volume source is not selected in context fields")
        sources = (*sources, volume)
    env = HistoricalChallengeEnv(
        markets, tick_values=source["point_values"], round_trip_fees=source["round_trip_fees"],
        spec=ChallengeSpec(**source["challenge"]),
        observation_spec=TradeManagementObservationSpec.from_config(source["observation"]),
        seed=config["seed"],
    )
    return env, sources


def collection_factory(config, root):
    """Load one frozen baseline, sharing weights but never recurrent state."""
    policy = config["collection_policy"]
    if policy["kind"] != "frozen_c51":
        raise ValueError("collection currently requires a declared frozen C51 continuation")
    path = resolve(root, policy["checkpoint"])
    if file_digest(path) != policy["sha256"]:
        raise ValueError("collection checkpoint identity mismatch")
    horizon = policy["recurrent_horizon"]
    if type(horizon) is not int or horizon < 1:
        raise ValueError("continuation recurrent horizon must be positive")
    from ..agent import RecurrentC51Agent
    agent, _ = RecurrentC51Agent.load(path, device=policy["device"],
                                    learner_backend_override=policy["learner_backend"])

    def factory():
        hidden, steps = None, 0
        def decide(observation, info):
            nonlocal hidden, steps
            if steps % horizon == 0:
                hidden = None
            action, hidden, _ = agent.select_action(
                observation, hidden=hidden, valid_actions=tuple(info["valid_actions"]),
                epsilon=0.0,
            )
            steps += 1
            return action
        return decide
    return factory


def collect_job(path):
    from .collector import collect_examples
    from .context import ContextConfig
    from .dataset import write_supervised_dataset
    config, root = read_job(path)
    source, audit, splits, sealed, identity = load_source_contract(config, root)
    continuation = audit.get("continuation", {})
    if (continuation.get("checkpoint_sha256") != config["collection_policy"]["sha256"]
            or type(continuation.get("fit_end_ns")) is not int
            or continuation["fit_end_ns"] >= splits["train"][0]):
        raise ValueError("continuation checkpoint needs a pre-training-period fit audit")
    context = ContextConfig.load(resolve(root, config["context_config"]))
    factory = collection_factory(config, root)
    kind = config["dataset_kind"]
    if kind not in {"action", "market"}:
        raise ValueError("dataset_kind must be action or market")
    output = resolve(root, config["dataset_output"])
    lineage = {"source_identity": identity, "specialist_identities": source["teachers"],
               "economic_contract": source["challenge"], "split_audit": audit,
               "context_config_sha256": file_digest(resolve(root, config["context_config"])),
               "job_config_sha256": file_digest(path)}
    if config.get("volume_source") is not None:
        lineage["volume_source"] = {key: file_digest(resolve(root, value))
                                    for key, value in config["volume_source"].items()}

    def records():
        for role in ("train", "valid"):
            env, sources = load_role(config, root, source, role)
            for episode in config["episodes"][role]:
                for pair in collect_examples(
                    env, reset_options=episode, context_config=context, sources=sources,
                    behavior_factory=factory, continuation_factory=factory,
                    source_id=identity + ":" + json.dumps(episode, sort_keys=True),
                    continuation_id=config["collection_policy"]["sha256"],
                    maximum_examples=config["maximum_examples_per_episode"],
                    sample_stride=config["sample_stride"], rollout_max_steps=config["rollout_max_steps"],
                    target_temperature=config["target_temperature"],
                    opportunity_contract=config["opportunity_contract"],
                ):
                    if pair[kind] is not None:
                        yield pair[kind]
    manifest = write_supervised_dataset(records(), output, splits=splits, lineage=lineage,
                                        sealed_start_ns=sealed)
    # No fabricated PASS: dataset audit is explicitly required after generation.
    return {"dataset": str(output), "counts": manifest["counts"],
            "next": "review dataset and supply matching audit.json before SFT"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("stage", choices=("check", "collect", "audit", "prepare", "train", "rl", "evaluate"))
    args = parser.parse_args(argv)
    config, root = read_job(args.config)
    if args.stage == "check":
        result = readiness(args.config)
    elif args.stage == "collect":
        result = collect_job(args.config)
    elif args.stage == "audit":
        from .mlx_sft import verify_dataset
        manifest = verify_dataset(resolve(root, config["dataset_output"]))
        result = {"dataset_audit_verified": True, "counts": manifest["counts"]}
    elif args.stage in {"prepare", "train"}:
        from .mlx_sft import main as sft_main
        command = ["--config", str(resolve(root, config["sft_config"])),
                   "--view", str(resolve(root, config["mlx_view"]))]
        if args.stage == "train":
            command.append("--train")
        sft_main(command)
        result = {"stage": args.stage, "completed": True}
    elif args.stage == "rl":
        from .context import ContextConfig
        from .policy import MLXActionPolicy
        from .rl import read_rl_config, MLXAdapterLearner, train_rl
        source, _, _, _, identity = load_source_contract(config, root)
        rl_config = read_rl_config(resolve(root, config["rl_config"]))
        output = resolve(root, rl_config["output_adapter"])
        if output.exists():
            raise FileExistsError("RL output exists; choose a new configured path")
        from .model_config import read_model_settings
        policy_config = resolve(root, rl_config["input_policy_config"])
        model_settings = read_model_settings(policy_config)
        if model_settings["adapter_path"] is None:
            raise ValueError("RL requires the supervised adapter as its parent")
        context = ContextConfig.load(resolve(root, config["context_config"]))
        contract = {
            "source_identity": identity, "policy": model_settings,
            "context": {"context_steps": context.context_steps, "fields": list(context.fields)},
            "episodes": config["episodes"]["train"],
            "learning": {key: value for key, value in rl_config.items() if key not in {
                "resume_checkpoint", "checkpoint_root", "checkpoint_keep", "output_adapter", "groups"}},
            "volume": None if config.get("volume_source") is None else {
                key: file_digest(resolve(root, value)) for key, value in config["volume_source"].items()},
        }
        resume = rl_config["resume_checkpoint"]
        if resume is not None:
            from .checkpoints import verify_checkpoint
            receipt = verify_checkpoint(resolve(root, resume))
            if receipt["contract"] != contract:
                raise ValueError("RL resume contract differs from saved training")
        env, sources = load_role(config, root, source, "train")
        policy = MLXActionPolicy.load(model_settings["model"],
            adapter_path=model_settings["adapter_path"] if resume is None else str(resolve(root, resume)),
            max_seq_length=model_settings["max_seq_length"],
            chat_template_kwargs=model_settings["chat_template_kwargs"])
        learner = MLXAdapterLearner(policy, rl_config)
        resume_state = None
        if resume is not None:
            from .checkpoints import restore_training_state
            resume_state = restore_training_state(resolve(root, resume), optimizer=learner.optimizer)
        metadata = {"source_identity": identity, "config": rl_config, "contract": contract,
                    "parent_adapter": model_settings["adapter_path"], "economic_validation": "not_yet_run",
                    "resume_boundary": "complete_rollout_group", "exact_resume_verified": False}
        def checkpoint(runtime):
            destination = resolve(root, rl_config["checkpoint_root"]) / f"group-{runtime['next_group']:06d}"
            learner.save(destination, model_settings["adapter_path"], metadata, runtime=runtime)
            from .checkpoints import prune_checkpoints
            removed = prune_checkpoints(destination.parent, keep=rl_config["checkpoint_keep"],
                contract=contract, protected=() if resume is None else (resolve(root, resume),))
            if removed:
                print(json.dumps({"pruned_intermediate_checkpoints": removed}), flush=True)
        metrics = train_rl(policy, env, learner=learner, episodes=config["episodes"]["train"],
                           context_config=context, sources=sources, config=rl_config,
                           resume_state=resume_state, checkpoint=checkpoint)
        learner.save(output, model_settings["adapter_path"], {**metadata, "groups": metrics})
        result = {"adapter": str(output), "groups": metrics}
    else:
        from .context import ContextConfig
        from .evaluation import evaluate_policy
        from .policy import MLXActionPolicy
        source, _, splits, _, identity = load_source_contract(config, root)
        destination = resolve(root, config["evaluation_output"])
        if destination.exists():
            raise FileExistsError("evaluation output exists; choose a new configured path")
        decision_path = resolve(root, config["evaluation_decisions"])
        if decision_path.exists():
            raise FileExistsError("evaluation decision log exists; choose a new configured path")
        if config.get("policy_config") is not None:
            from ..policy import load_policy
            policy = load_policy(resolve(root, config["policy_config"]), root=root)
        else:
            # Existing challenger recipes remain valid.
            policy = MLXActionPolicy.from_config(resolve(root, config["evaluation_policy_config"]))
        env, sources = load_role(config, root, source, "valid",
            include_specialists=policy.requires_specialists)
        criteria = json.loads(resolve(root, config["evaluation_metrics_config"]).read_text())
        decision_path.parent.mkdir(parents=True, exist_ok=True)
        with decision_path.open("x") as decisions:
            def log_decision(row):
                decisions.write(json.dumps(row, allow_nan=False) + "\n")
            result = evaluate_policy(policy, env, episodes=config["evaluation_episodes"],
                context_config=ContextConfig.load(resolve(root, config["context_config"])),
                sources=sources, max_steps=config["rollout_max_steps"],
                near_blow_headroom_fraction=criteria["near_blow_headroom_fraction"], on_decision=log_decision)
        from .selection import assess_candidate
        result.update(source_identity=identity, temporal_splits=splits, criteria=criteria,
                      selection=assess_candidate(result, criteria),
                      context_config_sha256=file_digest(resolve(root, config["context_config"])),
                      policy_config_sha256=file_digest(resolve(root,
                          config.get("policy_config") or config["evaluation_policy_config"])))
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x") as stream:
            json.dump(result, stream, indent=2, default=str, allow_nan=False)
    print(json.dumps(result, indent=2, default=str, allow_nan=False))
    return 1 if args.stage == "check" and not result["ready"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
