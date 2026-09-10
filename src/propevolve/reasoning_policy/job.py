"""Explicit JSON-driven reasoning collection, QLoRA and simulator evaluation.

No campaign promotion, training on import, fabricated audits, or implicit model
downloads. Each command is an explicit stage. A readiness check is metadata-only.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from ..decision import Action
from .integrity import file_digest


def read_job(path):
    from .model_config import read_recipe
    config = read_recipe(path)
    root = Path(config["workspace_root"]).resolve()
    return config, root


def resolve(root, value):
    return root / value


def sample_episode_specs(
    environment, *, tickers, count, seed, explicit=None, minimum_start_separation=0,
):
    """Materialize reproducible challenge windows without implicit run state."""
    if explicit:
        return [dict(item) for item in explicit]
    if (type(count) is not int or count < 1 or type(seed) is not int
            or type(minimum_start_separation) is not int or minimum_start_separation < 0
            or not tickers or len(set(tickers)) != len(tickers)):
        raise ValueError("invalid episode sampling contract")
    from ..environment import HistoricalChallengeEnv
    sampler = HistoricalChallengeEnv(
        environment.markets, tick_values=environment.tick_values,
        round_trip_fees=environment.round_trip_fees, spec=environment.spec,
        observation_spec=environment._assembler.trade_management, seed=seed,
    )
    schedule_rng = np.random.default_rng(seed)
    schedule = []
    while len(schedule) < count:
        cycle = list(tickers)
        schedule_rng.shuffle(cycle)
        schedule.extend(cycle)
    result, seen = [], set()
    starts_by_ticker = {ticker: [] for ticker in tickers}
    attempts = 0
    while len(result) < count and attempts < count * 100:
        ticker = schedule[len(result)]
        _, info = sampler.reset(options={"ticker": ticker})
        identity = (ticker, int(info["start"]))
        attempts += 1
        if (identity in seen or any(
                abs(identity[1] - prior) < minimum_start_separation
                for prior in starts_by_ticker[ticker])):
            continue
        seen.add(identity)
        starts_by_ticker[ticker].append(identity[1])
        result.append({"ticker": ticker, "start": identity[1]})
    if len(result) != count:
        raise ValueError("unable to sample unique challenge episodes")
    return result


def stratified_action_rows(candidates, *, per_action, seed):
    """Select balanced samples, or all eligible rows when the cap is null."""
    if ((per_action is not None and (type(per_action) is not int or per_action < 1))
            or type(seed) is not int):
        raise ValueError("invalid economic action sampling contract")
    if per_action is None:
        selected = []
        for ticker, payload in sorted(candidates.items()):
            labels = np.asarray(payload["labels"])
            eligible = np.asarray(payload["eligible"], dtype=bool)
            if labels.ndim != 1 or labels.shape != eligible.shape:
                raise ValueError("economic candidates are not aligned")
            if not np.isin(labels[eligible], [0, 1, 2]).all():
                raise ValueError("eligible economic row has invalid action")
            selected.extend((ticker, int(row), int(labels[row]))
                            for row in np.flatnonzero(eligible))
        return selected
    rng = np.random.default_rng(seed)
    selected = []
    for action in (int(Action.WAIT), int(Action.ENTER_LONG_1), int(Action.ENTER_SHORT_1)):
        groups = {}
        for ticker, payload in sorted(candidates.items()):
            labels = np.asarray(payload["labels"])
            eligible = np.asarray(payload["eligible"], dtype=bool)
            years = np.asarray(payload["years"])
            if labels.shape != eligible.shape or labels.shape != years.shape:
                raise ValueError("economic candidates are not aligned")
            for year in sorted(set(years[eligible & (labels == action)])):
                rows = np.flatnonzero(eligible & (labels == action) & (years == year))
                if len(rows):
                    groups.setdefault(ticker, []).append(list(rng.permutation(rows)))
        if sum(len(rows) for years in groups.values() for rows in years) < per_action:
            raise ValueError("insufficient natural economic labels for requested sample")
        action_rows = []
        tickers = sorted(groups)
        ticker_cursor = 0
        year_cursors = {ticker: 0 for ticker in tickers}
        while len(action_rows) < per_action:
            ticker = tickers[ticker_cursor % len(tickers)]
            year_groups = groups[ticker]
            picked = False
            for _ in range(len(year_groups)):
                year_index = year_cursors[ticker] % len(year_groups)
                year_cursors[ticker] += 1
                if year_groups[year_index]:
                    action_rows.append((ticker, int(year_groups[year_index].pop()), action))
                    picked = True
                    break
            if not picked:
                tickers.remove(ticker)
                if not tickers:
                    raise ValueError("economic action sample exhausted unexpectedly")
                ticker_cursor %= len(tickers)
                continue
            ticker_cursor += 1
        selected.extend(action_rows)
    # Collection walks large frozen memmaps. Emit cache-local rows here and
    # randomize the lightweight prepared row indices during SFT instead.
    selected.sort(key=lambda item: (item[0], item[1]))
    return selected


def economic_episode_specs(config, environment, sources, role):
    """Turn the exhaustive economic census into reproducible SFT anchors."""
    sampling = config["economic_action_sampling"][role]
    if (not isinstance(sampling, dict) or set(sampling) != {"per_action", "seed"}):
        raise ValueError("economic action sampling requires per_action and seed")
    from .labels import classify_market_action_rows
    contract = config["opportunity_contract"]
    warmup = config.get("collection_warmup_steps", 0)
    candidates = {}
    for ticker in config["tickers"][role]:
        market = environment.markets[ticker]
        labels = classify_market_action_rows(
            market, role_end=len(market.close),
            risk_dollars=environment.spec.per_trade_risk_dollars,
            point_value=environment.tick_values[ticker],
            round_trip_fee=environment.round_trip_fees[ticker],
            horizon=contract["horizon"], target_rs=contract["target_rs"],
            stop_r=contract["stop_r"],
            chunk_size=config.get("label_census_chunk_size", 16384),
        )
        eligible = labels >= 0
        eligible[:warmup] = False
        session_keys = environment._session_keys[ticker]
        unique_sessions = np.unique(session_keys)
        last_start_session = unique_sessions[-environment.spec.episode_days]
        maximum_start = min(
            len(market.close) - 2,
            int(np.searchsorted(session_keys, last_start_session, side="right") - 1),
        )
        eligible[maximum_start + warmup + 1:] = False
        for specialist in sources:
            eligible &= np.asarray(specialist.targets.availability[ticker], dtype=bool)
        candidates[ticker] = {
            "labels": labels, "eligible": eligible,
            "years": np.asarray(market.timestamps, dtype="datetime64[Y]").astype(str),
        }
    return [{"ticker": ticker, "start": row - warmup, "expected_action": action}
            for ticker, row, action in stratified_action_rows(
                candidates, per_action=sampling["per_action"], seed=sampling["seed"])]


def configured_episodes(config, environment, role, *, evaluation=False):
    if evaluation:
        explicit = config.get("evaluation_episodes")
        spec = config.get("evaluation_episode_sampling")
    else:
        explicit = config.get("episodes", {}).get(role)
        spec = config.get("episode_sampling", {}).get(role)
    if explicit:
        return sample_episode_specs(environment, tickers=tuple(config["tickers"][role]),
                                    count=len(explicit), seed=config["seed"], explicit=explicit)
    if (not isinstance(spec, dict)
            or set(spec) - {"count", "seed", "minimum_start_separation"}
            or not {"count", "seed"}.issubset(spec)):
        raise ValueError(f"missing episode sampling for {role}")
    return sample_episode_specs(environment, tickers=tuple(config["tickers"][role]),
                                count=spec["count"], seed=spec["seed"],
                                minimum_start_separation=spec.get("minimum_start_separation", 0))


def collection_source_for_role(config, source, role, source_splits):
    """Optionally carve an inner distillation split from the training reserve."""
    declared = config.get("dataset_temporal")
    if declared is None:
        return source, source_splits[role]
    if set(declared) != {"train_start", "train_end", "validation_start", "validation_end"}:
        raise ValueError("dataset_temporal requires exact train/validation bounds")
    ns = lambda value: int(np.datetime64(value, "ns").astype(np.int64))
    train = [ns(declared["train_start"]), ns(declared["train_end"])]
    valid = [ns(declared["validation_start"]), ns(declared["validation_end"])]
    source_train = source_splits["train"]
    if (train[0] >= train[1] or valid[0] >= valid[1] or train[1] > valid[0]
            or train[0] < source_train[0] or valid[1] > source_train[1]):
        raise ValueError("inner distillation roles must be disjoint inside source training")
    import copy
    bounded = copy.deepcopy(source)
    if role == "train":
        bounded["temporal"].update(train_start=declared["train_start"],
                                   train_end=declared["train_end"])
        return bounded, train
    bounded["temporal"].update(validation_start=declared["validation_start"],
                               validation_end=declared["validation_end"])
    return bounded, valid


def readiness(path):
    """Report missing resources without importing MLX or loading source arrays."""
    config, root = read_job(path)
    blockers = []
    for key in ("source_recipe", "temporal_split_audit", "context_config", "sft_config", "rl_config",
                "evaluation_policy_config", "evaluation_metrics_config"):
        if not config.get(key) or not resolve(root, config[key]).is_file():
            blockers.append(key)
    policy = config.get("collection_policy", {})
    if policy.get("kind") in {None, "frozen_c51"}:
        if not policy.get("checkpoint") or not resolve(root, policy["checkpoint"]).is_file():
            blockers.append("collection_policy.checkpoint")
        if not policy.get("sha256"):
            blockers.append("collection_policy.sha256")
    elif policy.get("kind") != "reset_states":
        blockers.append("collection_policy.kind")
    if importlib.util.find_spec("mlx_lm") is None:
        blockers.append("mlx_lm")
    for role in ("train", "valid"):
        if (not config.get("episodes", {}).get(role)
                and not config.get("episode_sampling", {}).get(role)):
            blockers.append(f"episodes.{role}")
    if not config.get("evaluation_episodes") and not config.get("evaluation_episode_sampling"):
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


def publish_source_audit(path):
    """Authenticate the exact local source, teachers and continuation policy.

    Specialist caches are inspected only for the declared supervision roles.
    The unseen action-validation role needs embeddings and prices, never teacher
    scores.  The receipt is atomic and cannot overwrite reviewed evidence.
    """
    from ..config import materialize_effective_config
    from .context import ContextConfig
    import hashlib

    config, root = read_job(path)
    destination = resolve(root, config["temporal_split_audit"])
    if destination.exists():
        raise FileExistsError(f"source audit already exists: {destination}")
    source_path = resolve(root, config["source_recipe"])
    source = materialize_effective_config(json.loads(source_path.read_text()))
    temporal = source["temporal"]
    ns = lambda value: int(np.datetime64(value, "ns").astype(np.int64))
    splits = {role: [ns(temporal[f"{prefix}_start"]), ns(temporal[f"{prefix}_end"])]
              for role, prefix in (("train", "train"), ("valid", "validation"))}
    sealed = ns(config["sealed_start"])
    if (ns(temporal["sealed_start"]) != sealed
            or splits["train"][1] > splits["valid"][0]
            or any(start >= end or end > sealed for start, end in splits.values())):
        raise ValueError("source roles are overlapping or cross sealed 2026")
    context = ContextConfig.load(resolve(root, config["context_config"]))
    if context.input_mode != "embeddings":
        raise ValueError("teacher-free challenger source requires embedding context")
    specialist_roles = tuple(config.get("specialist_supervision_roles", ("train",)))
    if not specialist_roles or set(specialist_roles) - {"train", "valid"}:
        raise ValueError("invalid specialist supervision roles")
    role_receipts = {}
    for role in ("train", "valid"):
        role_source, dataset_bounds = collection_source_for_role(config, source, role, splits)
        env, sources = load_role(config, root, role_source, role,
                                 include_specialists=role in specialist_roles)
        dimensions = {market.embeddings.shape[1] for market in env.markets.values()}
        if dimensions != {config["embedding_dim"]}:
            raise ValueError("source embedding dimension differs from reasoning projector")
        kinds = tuple(item.kind for item in sources)
        if role in specialist_roles and kinds != tuple(item["kind"] for item in source["teachers"]):
            raise ValueError("loaded specialist ordering differs from source recipe")
        role_receipts[role] = {
            "tickers": sorted(env.markets), "embedding_dim": dimensions.pop(),
            "rows": {ticker: len(market.close) for ticker, market in env.markets.items()},
            "specialists": list(kinds),
            "dataset_bounds": dataset_bounds,
        }
    policy = config["collection_policy"]
    if policy.get("kind") == "frozen_c51":
        checkpoint = resolve(root, policy["checkpoint"])
        if file_digest(checkpoint) != policy["sha256"]:
            raise ValueError("collection checkpoint identity mismatch")
        recipe_path = resolve(root, policy["recipe"])
        if file_digest(recipe_path) != policy["recipe_sha256"]:
            raise ValueError("collection policy recipe identity mismatch")
        continuation_recipe = json.loads(recipe_path.read_text())
        fit_end = ns(continuation_recipe["temporal"]["train_end"])
        if fit_end > splits["valid"][0]:
            raise ValueError("collection policy was fitted through unseen validation")
        continuation = {"kind": "frozen_c51", "checkpoint_sha256": policy["sha256"],
                        "recipe_sha256": policy["recipe_sha256"], "fit_end_ns": fit_end}
    elif policy.get("kind") == "reset_states" and set(policy) == {"kind"}:
        continuation = {"kind": "reset_states"}
    else:
        raise ValueError("invalid collection policy contract")
    effective_identity = hashlib.sha256(
        json.dumps(source, sort_keys=True, allow_nan=False).encode()).hexdigest()
    audit = {
        "schema": "propevolve_reasoning_source_audit_v1", "status": "PASS",
        "source_recipe_sha256": file_digest(source_path),
        "effective_source_sha256": effective_identity,
        "specialist_score_mode": config["specialist_score_mode"],
        "specialist_supervision_roles": list(specialist_roles),
        "sealed_touched": False, "splits": splits, "roles": role_receipts,
        "continuation": continuation,
    }
    if audit["specialist_score_mode"] not in {"out_of_fold", "post_fit"}:
        raise ValueError("unknown specialist score mode")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".source-audit-", suffix=".json",
                                              dir=destination.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(audit, stream, indent=2, allow_nan=False)
        os.rename(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return audit


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
    if policy.get("kind") == "reset_states":
        if set(policy) != {"kind"}:
            raise ValueError("reset-state collection accepts no prior-policy settings")
        def factory():
            def decide(observation, info):
                legal = tuple(Action(value) for value in info["valid_actions"])
                for candidate in (Action.WAIT, Action.HOLD):
                    if candidate in legal:
                        return candidate
                raise ValueError("reset-state collector has no passive legal action")
            return decide
        return factory
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


def action_collection_plan(config, expected_action):
    """Resolve one config-declared flat or full-trade supervision path."""
    action = Action(expected_action)
    scope = config.get("action_supervision_scope", "entry")
    if scope not in {"entry", "trade_mastery"}:
        raise ValueError("unknown action supervision scope")
    maximum = config["maximum_examples_per_episode"]
    if scope == "trade_mastery" and action in {
            Action.ENTER_LONG_1, Action.ENTER_SHORT_1}:
        return {
            "mode": "trade_mastery_grid",
            "maximum_examples": maximum,
            "initial_entry_action": action,
        }
    return {
        "mode": "market_barrier_grid",
        "maximum_examples": 1 if scope == "trade_mastery" else maximum,
        "initial_entry_action": None,
    }


def collect_job(path):
    from .collector import collect_examples
    from .context import ContextConfig
    from .dataset import write_supervised_dataset
    config, root = read_job(path)
    source, audit, splits, sealed, identity = load_source_contract(config, root)
    context = ContextConfig.load(resolve(root, config["context_config"]))
    factory = collection_factory(config, root)
    kind = config["dataset_kind"]
    if kind not in {"action", "market"}:
        raise ValueError("dataset_kind must be action or market")
    continuation = audit.get("continuation", {})
    policy = config["collection_policy"]
    if policy["kind"] == "frozen_c51":
        if (continuation.get("kind") not in {None, "frozen_c51"}
                or continuation.get("checkpoint_sha256") != policy["sha256"]
                or type(continuation.get("fit_end_ns")) is not int
                or (kind == "action" and continuation["fit_end_ns"] > splits["valid"][0])):
            raise ValueError("continuation checkpoint must predate unseen action validation")
        continuation_id = policy["sha256"]
        action_label_mode = "continuation"
    elif policy["kind"] == "reset_states" and continuation == {"kind": "reset_states"}:
        continuation_id = "market-barrier-grid"
        action_label_mode = "market_barrier_grid"
    else:
        raise ValueError("source audit does not match collection policy")
    role_sources = {role: collection_source_for_role(config, source, role, splits)
                    for role in ("train", "valid")}
    dataset_splits = {role: role_sources[role][1] for role in role_sources}
    output = resolve(root, config["dataset_output"])
    lineage = {"source_identity": identity, "specialist_identities": source["teachers"],
               "economic_contract": config.get("opportunity_contract", source["challenge"]),
               "supervision_scope": config.get("action_supervision_scope", kind),
               "split_audit": audit,
               "context_config_sha256": file_digest(resolve(root, config["context_config"])),
               "job_config_sha256": file_digest(path)}
    if config.get("volume_source") is not None:
        lineage["volume_source"] = {key: file_digest(resolve(root, value))
                                    for key, value in config["volume_source"].items()}

    def records():
        for role in ("train", "valid"):
            augment_action = kind == "action" and config.get("augment_action_targets", False)
            specialist_roles = tuple(config.get("specialist_supervision_roles", ("train",)))
            include_specialists = (kind == "market" or
                                   (augment_action and role in specialist_roles))
            env, sources = load_role(config, root, role_sources[role][0], role,
                                     include_specialists=include_specialists)
            episodes = (economic_episode_specs(config, env, sources, role)
                        if config.get("economic_action_sampling") is not None else
                        configured_episodes(config, env, role))
            if (config.get("economic_action_sampling") is not None
                    and config.get("embedding_storage") == "source_embedding_reference_v1"):
                # The schedule is now lightweight. Release the all-market selector
                # before traversing one cache-local ticker at a time.
                del env, sources
                import gc
                gc.collect()
                groups = {}
                for selected in episodes:
                    groups.setdefault(selected["ticker"], []).append(selected)
                for ticker in sorted(groups):
                    single = dict(config)
                    single["tickers"] = {**config["tickers"], role: [ticker]}
                    ticker_env, ticker_sources = load_role(
                        single, root, role_sources[role][0], role,
                        include_specialists=include_specialists,
                    )
                    # Consume each ticker completely before loading the next.
                    for selected in groups[ticker]:
                        episode = {key: selected[key] for key in ("ticker", "start")}
                        plan = (action_collection_plan(config, selected["expected_action"])
                                if kind == "action" else None)
                        for pair in collect_examples(
                            ticker_env, reset_options=episode, context_config=context,
                            sources=ticker_sources, behavior_factory=factory,
                            continuation_factory=factory,
                            source_id=identity + ":" + json.dumps(episode, sort_keys=True),
                            continuation_id=continuation_id,
                            maximum_examples=(config["maximum_examples_per_episode"]
                                              if plan is None else plan["maximum_examples"]),
                            sample_stride=config["sample_stride"],
                            rollout_max_steps=config["rollout_max_steps"],
                            target_temperature=config["target_temperature"],
                            opportunity_contract=config["opportunity_contract"],
                            action_label_mode=(action_label_mode if plan is None else plan["mode"]),
                            initial_entry_action=(None if plan is None else
                                                  plan["initial_entry_action"]),
                            collection_warmup_steps=config.get("collection_warmup_steps", 0),
                            collect_action_targets=kind == "action",
                            collect_market_targets=kind == "market",
                            augment_action_targets=augment_action,
                        ):
                            if pair[kind] is not None:
                                expected = Action(selected["expected_action"]).name
                                legal = pair[kind]["targets"].get("action_order", ())
                                if (expected in legal
                                        and pair[kind]["messages"][-1]["content"] != expected):
                                    raise ValueError("selected economic action changed during collection")
                                yield pair[kind]
                    del ticker_env, ticker_sources
                    gc.collect()
            else:
                for selected in episodes:
                    episode = {key: selected[key] for key in ("ticker", "start")}
                    for pair in collect_examples(
                        env, reset_options=episode, context_config=context, sources=sources,
                        behavior_factory=factory, continuation_factory=factory,
                        source_id=identity + ":" + json.dumps(episode, sort_keys=True),
                        continuation_id=continuation_id,
                        maximum_examples=config["maximum_examples_per_episode"],
                        sample_stride=config["sample_stride"],
                        rollout_max_steps=config["rollout_max_steps"],
                        target_temperature=config["target_temperature"],
                        opportunity_contract=config["opportunity_contract"],
                        action_label_mode=action_label_mode,
                        collection_warmup_steps=config.get("collection_warmup_steps", 0),
                        collect_action_targets=kind == "action",
                        collect_market_targets=kind == "market",
                        augment_action_targets=augment_action,
                    ):
                        if pair[kind] is not None:
                            if kind == "action" and "expected_action" in selected:
                                expected = Action(selected["expected_action"]).name
                                if pair[kind]["messages"][-1]["content"] != expected:
                                    raise ValueError("selected economic action changed during collection")
                            yield pair[kind]
    manifest = write_supervised_dataset(
        records(), output, splits=dataset_splits, lineage=lineage,
        sealed_start_ns=sealed,
        embedding_storage=config.get("embedding_storage", "json"),
        embedding_source_cache_root=(
            resolve(root, source["cache_root"])
            if config.get("embedding_storage") == "source_embedding_reference_v1"
            else None
        ),
    )
    # No fabricated PASS: dataset audit is explicitly required after generation.
    return {"dataset": str(output), "counts": manifest["counts"],
            "next": "review dataset and supply matching audit.json before SFT"}


def label_census_job(path):
    """Count every causal economic action label before selecting an SFT corpus."""
    config, root = read_job(path)
    source, _, splits, sealed, identity = load_source_contract(config, root)
    destination = resolve(root, config["label_census_output"])
    if destination.exists():
        raise FileExistsError(f"label census already exists: {destination}")
    from .context import ContextConfig
    from .labels import classify_market_action_rows
    context = ContextConfig.load(resolve(root, config["context_config"]))
    contract = config["opportunity_contract"]
    report = {
        "schema": "propevolve_reasoning_action_label_census_v1",
        "status": "PASS", "source_identity": identity,
        "sealed_start_ns": sealed, "sealed_touched": False,
        "economic_contract": contract, "roles": {},
    }
    action_names = {int(action): action.name for action in (
        Action.WAIT, Action.ENTER_LONG_1, Action.ENTER_SHORT_1)}
    for role in ("train", "valid"):
        role_counts = {name: 0 for name in action_names.values()}
        role_years = {}
        ticker_reports = {}
        for ticker in config["tickers"][role]:
            single = dict(config)
            single["tickers"] = {**config["tickers"], role: [ticker]}
            include_specialists = role in tuple(config.get("specialist_supervision_roles", ("train",)))
            env, sources = load_role(single, root, source, role,
                                     include_specialists=include_specialists)
            market = env.markets[ticker]
            labels = classify_market_action_rows(
                market, role_end=len(market.close),
                risk_dollars=env.spec.per_trade_risk_dollars,
                point_value=env.tick_values[ticker],
                round_trip_fee=env.round_trip_fees[ticker],
                horizon=contract["horizon"], target_rs=contract["target_rs"],
                stop_r=contract["stop_r"],
                chunk_size=config.get("label_census_chunk_size", 16384),
            )
            eligible = labels >= 0
            eligible[:context.context_steps - 1] = False
            for specialist in sources:
                eligible &= np.asarray(specialist.targets.availability[ticker], dtype=bool)
            counts = {name: int(np.count_nonzero(eligible & (labels == value)))
                      for value, name in action_names.items()}
            years = np.asarray(market.timestamps, dtype="datetime64[Y]").astype(str)
            by_year = {}
            for year in sorted(set(years[eligible])):
                mask = eligible & (years == year)
                by_year[year] = {name: int(np.count_nonzero(mask & (labels == value)))
                                 for value, name in action_names.items()}
                aggregate = role_years.setdefault(year, {name: 0 for name in action_names.values()})
                for name, value in by_year[year].items():
                    aggregate[name] += value
            ticker_reports[ticker] = {
                "eligible_rows": int(np.count_nonzero(eligible)),
                "actions": counts, "years": by_year,
            }
            for name, value in counts.items():
                role_counts[name] += value
        report["roles"][role] = {
            "bounds_ns": splits[role], "eligible_rows": sum(role_counts.values()),
            "actions": role_counts, "years": role_years, "tickers": ticker_reports,
        }
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".reasoning-label-census-", suffix=".json", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
        os.rename(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("stage", choices=("check", "publish-source-audit", "census", "collect", "publish-audit", "audit", "prepare", "train", "rl", "evaluate"))
    args = parser.parse_args(argv)
    config, root = read_job(args.config)
    if args.stage == "check":
        result = readiness(args.config)
    elif args.stage == "publish-source-audit":
        result = publish_source_audit(args.config)
    elif args.stage == "collect":
        result = collect_job(args.config)
    elif args.stage == "census":
        result = label_census_job(args.config)
    elif args.stage == "publish-audit":
        from .dataset import audit_supervised_dataset
        _, source_audit, _, _, _ = load_source_contract(config, root)
        audit = audit_supervised_dataset(
            resolve(root, config["dataset_output"]),
            specialist_score_mode=source_audit["specialist_score_mode"],
        )
        result = {"dataset_audit_published": True, "audit": audit}
    elif args.stage == "audit":
        from .mlx_sft import verify_dataset
        manifest = verify_dataset(resolve(root, config["dataset_output"]))
        result = {"dataset_audit_verified": True, "counts": manifest["counts"]}
    elif args.stage in {"prepare", "train"}:
        from .mlx_sft import main as sft_main
        command = ["--config", str(resolve(root, config["sft_config"])),
                   "--root", str(root),
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
        from .model_config import read_model_settings, validate_trade_mastery_parent
        policy_config = resolve(root, rl_config["input_policy_config"])
        model_settings = read_model_settings(policy_config, root=root)
        validate_trade_mastery_parent(model_settings)
        if model_settings["adapter_path"] is None:
            raise ValueError("RL requires the supervised adapter as its parent")
        from .rl import require_challenge_mastery_context
        context = require_challenge_mastery_context(ContextConfig.load(resolve(
            root, config.get("rl_context_config", config["context_config"])
        )))
        if context.input_mode != model_settings["input_mode"]:
            raise ValueError("RL context and policy input modes differ")
        env, sources = load_role(config, root, source, "train",
                                 include_specialists=context.input_mode == "specialists")
        training_episodes = configured_episodes(config, env, "train")
        contract = {
            "source_identity": identity, "policy": model_settings,
            "context": {"context_steps": context.context_steps, "text_steps": context.text_steps,
                        "fields": list(context.fields), "input_mode": context.input_mode},
            "episodes": training_episodes,
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
        policy = MLXActionPolicy.load(model_settings["model"],
            adapter_path=model_settings["adapter_path"] if resume is None else str(resolve(root, resume)),
            max_seq_length=model_settings["max_seq_length"],
            chat_template_kwargs=model_settings["chat_template_kwargs"],
            input_mode=model_settings["input_mode"], projector=model_settings["projector"])
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
        diagnostic_records = None
        if rl_config.get("frozen_audit") is not None:
            from .frozen_audit import load_frozen_records
            diagnostic_records = load_frozen_records(rl_config["frozen_audit"], root=root)
        metrics = train_rl(policy, env, learner=learner, episodes=training_episodes,
                           context_config=context, sources=sources, config=rl_config,
                           resume_state=resume_state, checkpoint=checkpoint,
                           diagnostic_records=diagnostic_records)
        learner.save(output, model_settings["adapter_path"], {**metadata, "groups": metrics})
        result = {"adapter": str(output), "groups": metrics}
    else:
        from .context import ContextConfig
        from .evaluation import evaluate_responsibilities
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
            policy = MLXActionPolicy.from_config(resolve(root, config["evaluation_policy_config"]), root=root)
        env, sources = load_role(config, root, source, "valid",
            include_specialists=policy.requires_specialists)
        criteria = json.loads(resolve(root, config["evaluation_metrics_config"]).read_text())
        trade_criteria = json.loads(resolve(
            root, config["trade_mastery_metrics_config"]).read_text())
        trade_audit = config.get("trade_mastery_audit")
        if not isinstance(trade_audit, dict):
            raise ValueError(
                "evaluation requires a configured frozen trade-mastery audit")
        from .frozen_audit import load_frozen_records
        trade_records = load_frozen_records(trade_audit, root=root)
        decision_path.parent.mkdir(parents=True, exist_ok=True)
        from .rl import require_challenge_mastery_context
        evaluation_context_path = config.get(
            "evaluation_context_config",
            config.get("rl_context_config", config["context_config"]),
        )
        evaluation_context = require_challenge_mastery_context(
            ContextConfig.load(resolve(root, evaluation_context_path))
        )
        with decision_path.open("x") as decisions:
            def log_decision(row):
                decisions.write(json.dumps(row, allow_nan=False) + "\n")
            result = evaluate_responsibilities(policy, env,
                trade_mastery_records=trade_records, episodes=configured_episodes(
                config, env, "valid", evaluation=True),
                context_config=evaluation_context,
                sources=sources, max_steps=config["rollout_max_steps"],
                near_blow_headroom_fraction=criteria["near_blow_headroom_fraction"], on_decision=log_decision)
        from .selection import assess_candidate, assess_trade_mastery
        challenge = result["challenge_mastery"]
        result.update(source_identity=identity, temporal_splits=splits,
                      criteria={"trade_mastery": trade_criteria,
                                "challenge_mastery": criteria},
                      selection={"trade_mastery": assess_trade_mastery(
                                     result["trade_mastery"], trade_criteria),
                                 "challenge_mastery": assess_candidate(
                                     challenge, criteria)},
                      context_config_sha256=file_digest(resolve(root, evaluation_context_path)),
                      policy_config_sha256=file_digest(resolve(root,
                          config.get("policy_config") or config["evaluation_policy_config"])))
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x") as stream:
            json.dump(result, stream, indent=2, default=str, allow_nan=False)
    print(json.dumps(result, indent=2, default=str, allow_nan=False))
    return 1 if args.stage == "check" and not result["ready"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
