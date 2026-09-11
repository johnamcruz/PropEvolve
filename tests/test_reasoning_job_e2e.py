"""Public job boundary: real files/simulator, external model runtime optional."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from propevolve.decision import Action
from propevolve.reasoning_policy import job
from propevolve.reasoning_policy.job import (
    action_collection_plan,
    collection_factory,
    configured_episodes,
    economic_episode_specs,
    collect_job,
    label_census_job,
    load_role,
    load_source_contract,
    publish_source_audit,
    readiness,
)
from propevolve.reasoning_policy.integrity import file_digest
from test_reasoning_source_config import source_payload


def test_readiness_reports_missing_sources_without_loading_a_model(tmp_path):
    config = tmp_path / "arbitrary-name.json"
    config.write_text(json.dumps({
        "workspace_root": str(tmp_path), "source_recipe": None,
        "temporal_split_audit": None, "context_config": "absent-context.json",
        "sft_config": "absent-model.json", "collection_policy": {"kind": "typo"},
    }))
    result = readiness(config)
    assert result["ready"] is False
    assert {"source_recipe", "temporal_split_audit", "context_config", "sft_config",
            "collection_policy.kind"}.issubset(result["blockers"])


def test_reset_state_collection_needs_no_prior_policy_artifact(tmp_path):
    required = {}
    for name in ("source.json", "audit.json", "context.json", "sft.json", "rl.json",
                 "policy.json", "metrics.json"):
        (tmp_path / name).write_text("{}")
    config = tmp_path / "scratch.json"
    config.write_text(json.dumps({
        "workspace_root": str(tmp_path), "source_recipe": "source.json",
        "temporal_split_audit": "audit.json", "context_config": "context.json",
        "sft_config": "sft.json", "rl_config": "rl.json",
        "evaluation_policy_config": "policy.json",
        "evaluation_metrics_config": "metrics.json",
        "collection_policy": {"kind": "reset_states"},
        "episode_sampling": {"train": {"count": 1, "seed": 1},
                             "valid": {"count": 1, "seed": 2}},
        "evaluation_episode_sampling": {"count": 1, "seed": 3},
    }))
    result = readiness(config)
    assert "collection_policy.checkpoint" not in result["blockers"]
    decide = collection_factory(json.loads(config.read_text()), tmp_path)()
    assert decide(None, {"valid_actions": [Action.WAIT]}) is Action.WAIT


def test_trade_mastery_job_config_expands_only_winning_entries_into_position_labels():
    config = {
        "action_supervision_scope": "trade_mastery",
        "maximum_examples_per_episode": 3,
    }
    assert action_collection_plan(config, Action.ENTER_LONG_1) == {
        "mode": "trade_mastery_grid",
        "maximum_examples": 3,
        "initial_entry_action": Action.ENTER_LONG_1,
    }
    assert action_collection_plan(config, Action.ENTER_SHORT_1) == {
        "mode": "trade_mastery_grid",
        "maximum_examples": 3,
        "initial_entry_action": Action.ENTER_SHORT_1,
    }
    assert action_collection_plan(config, Action.WAIT) == {
        "mode": "market_barrier_grid",
        "maximum_examples": 1,
        "initial_entry_action": None,
    }


def test_passive_collection_uses_wait_flat_and_hold_positioned(tmp_path):
    decide = collection_factory({"collection_policy": {"kind": "reset_states"}}, tmp_path)()

    assert decide(None, {"valid_actions": [Action.WAIT, Action.ENTER_LONG_1]}) is Action.WAIT
    assert decide(None, {"valid_actions": [Action.HOLD, Action.CLOSE]}) is Action.HOLD
    with pytest.raises(ValueError, match="no passive legal action"):
        decide(None, {"valid_actions": [Action.ENTER_LONG_1, Action.ENTER_SHORT_1]})
    with pytest.raises(ValueError, match="requires reset states"):
        collection_factory({"collection_policy": {"kind": "policy"}}, tmp_path)


def test_job_rejects_unknown_trade_mastery_scope():
    with pytest.raises(ValueError, match="unknown action supervision scope"):
        action_collection_plan({
            "action_supervision_scope": "challenge_mastery",
            "maximum_examples_per_episode": 2,
        }, Action.WAIT)


def test_configured_episodes_respects_explicit_evaluation_role():
    config = {
        "seed": 7,
        "tickers": {"valid": ["NQ"]},
        "evaluation_episodes": [{"ticker": "NQ", "start": 11}],
    }

    assert configured_episodes(config, object(), "valid", evaluation=True) == [
        {"ticker": "NQ", "start": 11}
    ]
    with pytest.raises(ValueError, match="missing episode sampling"):
        configured_episodes({"tickers": {"train": ["NQ"]}}, object(), "train")


def test_economic_episode_specs_selects_only_causal_available_rows(monkeypatch):
    market = SimpleNamespace(
        close=np.arange(12, dtype=float),
        timestamps=np.asarray(
            ["2021-01-01"] * 6 + ["2022-01-01"] * 6,
            dtype="datetime64[ns]",
        ),
    )
    environment = SimpleNamespace(
        markets={"NQ": market},
        spec=SimpleNamespace(per_trade_risk_dollars=300.0, episode_days=2),
        tick_values={"NQ": 20.0},
        round_trip_fees={"NQ": 3.84},
        _session_keys={"NQ": np.asarray([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6])},
    )
    labels = np.asarray([-1, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1], dtype=np.int8)
    monkeypatch.setattr(
        "propevolve.reasoning_policy.labels.classify_market_action_rows",
        lambda *args, **kwargs: labels.copy(),
    )
    sources = [SimpleNamespace(targets=SimpleNamespace(
        availability={"NQ": np.asarray(
            [True, True, True, True, True, True, True, True, True, False, True, True]
        )}
    ))]
    config = {
        "tickers": {"train": ["NQ"]},
        "economic_action_sampling": {"train": {"per_action": 1, "seed": 3}},
        "opportunity_contract": {"horizon": 5, "target_rs": [2.0], "stop_r": -1.0},
        "collection_warmup_steps": 1,
    }

    selected = economic_episode_specs(config, environment, sources, "train")

    assert {row["expected_action"] for row in selected} == {0, 1, 2}
    assert all(row["start"] >= 0 for row in selected)


def test_readiness_accepts_complete_resource_contract_and_volume(tmp_path, monkeypatch):
    names = [
        "source.json", "audit.json", "context.json", "sft.json", "rl.json",
        "policy.json", "metrics.json", "volume-manifest.json", "volume-audit.json",
    ]
    for name in names:
        (tmp_path / name).write_text("{}")
    config = tmp_path / "job.json"
    config.write_text(json.dumps({
        "workspace_root": str(tmp_path),
        "source_recipe": "source.json",
        "temporal_split_audit": "audit.json",
        "context_config": "context.json",
        "sft_config": "sft.json",
        "rl_config": "rl.json",
        "evaluation_policy_config": "policy.json",
        "evaluation_metrics_config": "metrics.json",
        "collection_policy": {"kind": "reset_states"},
        "episode_sampling": {
            "train": {"count": 1, "seed": 1},
            "valid": {"count": 1, "seed": 2},
        },
        "evaluation_episode_sampling": {"count": 1, "seed": 3},
        "volume_source": {"manifest": "volume-manifest.json", "audit": "volume-audit.json"},
    }))
    monkeypatch.setattr(job.importlib.util, "find_spec", lambda name: object())

    assert readiness(config)["ready"] is True


def _authenticated_source_contract(tmp_path):
    source = source_payload()
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps(source))
    ns = lambda value: int(np.datetime64(value, "ns").astype(np.int64))
    splits = {
        "train": [ns("2021-01-01"), ns("2025-01-01")],
        "valid": [ns("2025-01-01"), ns("2026-01-01")],
    }
    audit = {
        "status": "PASS",
        "source_recipe_sha256": file_digest(source_path),
        "effective_source_sha256": hashlib.sha256(
            json.dumps(source, sort_keys=True, allow_nan=False).encode()
        ).hexdigest(),
        "specialist_score_mode": "out_of_fold",
        "sealed_touched": False,
        "splits": splits,
    }
    audit_path = tmp_path / "source-audit.json"
    audit_path.write_text(json.dumps(audit))
    config = {
        "source_recipe": source_path.name,
        "temporal_split_audit": audit_path.name,
        "sealed_start": "2026-01-01",
    }
    return config, source, audit, splits


def test_source_contract_authenticates_recipe_effective_values_and_temporal_roles(tmp_path):
    config, source, audit, splits = _authenticated_source_contract(tmp_path)

    loaded = load_source_contract(config, tmp_path)

    assert loaded[:3] == (source, audit, splits)
    assert loaded[3] == int(np.datetime64("2026-01-01", "ns").astype(np.int64))
    assert loaded[4] == audit["effective_source_sha256"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda config, audit: audit.update(status="BLOCKED"), "matching passed"),
        (lambda config, audit: audit.update(sealed_touched=True), "matching passed"),
        (lambda config, audit: config.update(sealed_start="2025-12-01"), "temporal roles"),
        (lambda config, audit: audit.update(effective_source_sha256="0" * 64), "effective source"),
    ],
)
def test_source_contract_fails_closed_when_reviewed_evidence_drifts(
        tmp_path, mutation, message):
    config, _, audit, _ = _authenticated_source_contract(tmp_path)
    mutation(config, audit)
    (tmp_path / "source-audit.json").write_text(json.dumps(audit))

    with pytest.raises(ValueError, match=message):
        load_source_contract(config, tmp_path)


def test_label_census_reports_every_role_action_by_ticker_and_year(tmp_path, monkeypatch):
    output = tmp_path / "census.json"
    config = {
        "label_census_output": output.name,
        "context_config": "context.json",
        "tickers": {"train": ["NQ"], "valid": ["NQ"]},
        "opportunity_contract": {"horizon": 5, "target_rs": [2.0], "stop_r": -1.0},
    }
    market = SimpleNamespace(
        close=np.arange(7, dtype=float),
        timestamps=np.asarray([
            "2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04",
            "2022-01-01", "2022-01-02", "2022-01-03",
        ], dtype="datetime64[ns]"),
    )
    environment = SimpleNamespace(
        markets={"NQ": market}, spec=SimpleNamespace(per_trade_risk_dollars=300.0),
        tick_values={"NQ": 20.0}, round_trip_fees={"NQ": 3.84},
    )
    labels = np.asarray([-1, 0, 1, 2, 0, 1, 2], dtype=np.int8)
    monkeypatch.setattr(job, "read_job", lambda path: (config, tmp_path))
    monkeypatch.setattr(job, "load_source_contract", lambda *args: (
        {"temporal": {}}, {}, {"train": [0, 10], "valid": [10, 20]}, 30, "source-id"
    ))
    monkeypatch.setattr(job, "collection_source_for_role",
                        lambda config, source, role, splits: (source, splits[role]))
    monkeypatch.setattr(job, "load_role",
                        lambda *args, **kwargs: (environment, ()))
    monkeypatch.setattr(
        "propevolve.reasoning_policy.context.ContextConfig.load",
        lambda path: SimpleNamespace(context_steps=2),
    )
    monkeypatch.setattr(
        "propevolve.reasoning_policy.labels.classify_market_action_rows",
        lambda *args, **kwargs: labels.copy(),
    )

    report = label_census_job(tmp_path / "job.json")

    assert output.is_file()
    assert report["sealed_touched"] is False
    assert report["roles"]["train"]["actions"] == {
        "WAIT": 2, "ENTER_LONG_1": 2, "ENTER_SHORT_1": 2,
    }
    assert report["roles"]["valid"]["years"]["2022"]["ENTER_SHORT_1"] == 1
    with pytest.raises(FileExistsError, match="already exists"):
        label_census_job(tmp_path / "job.json")


def test_collect_job_preserves_economic_action_and_lineage_across_roles(
        tmp_path, monkeypatch):
    job_path = tmp_path / "job.json"
    context_path = tmp_path / "context.json"
    job_path.write_text("{}")
    context_path.write_text("{}")
    config = {
        "context_config": context_path.name,
        "collection_policy": {"kind": "reset_states"},
        "dataset_kind": "action",
        "dataset_output": "dataset",
        "tickers": {"train": ["NQ"], "valid": ["NQ"]},
        "maximum_examples_per_episode": 1,
        "sample_stride": 1,
        "rollout_max_steps": 2,
        "target_temperature": 1.0,
        "opportunity_contract": {"horizon": 5, "target_rs": [2.0], "stop_r": -1.0},
    }
    source = {"teachers": [], "challenge": {}, "cache_root": "cache"}
    source_audit = {"continuation": {"kind": "reset_states"}}
    monkeypatch.setattr(job, "read_job", lambda path: (config, tmp_path))
    monkeypatch.setattr(job, "load_source_contract", lambda *args: (
        source, source_audit, {"train": [0, 10], "valid": [10, 20]}, 30, "source-id"
    ))
    monkeypatch.setattr(job, "collection_source_for_role",
                        lambda config, source, role, splits: (source, splits[role]))
    monkeypatch.setattr(job, "load_role", lambda *args, **kwargs: (object(), ()))
    monkeypatch.setattr(job, "configured_episodes", lambda config, env, role: [
        {"ticker": "NQ", "start": 1 if role == "train" else 11,
         "expected_action": int(Action.ENTER_LONG_1)}
    ])
    monkeypatch.setattr(
        "propevolve.reasoning_policy.context.ContextConfig.load",
        lambda path: SimpleNamespace(context_steps=2),
    )
    def collect_examples(*args, **kwargs):
        yield {"action": {
            "source_id": kwargs["source_id"],
            "messages": [{"role": "assistant", "content": "ENTER_LONG_1"}],
        }}
    monkeypatch.setattr(
        "propevolve.reasoning_policy.collector.collect_examples", collect_examples
    )
    captured = {}
    def write(records, output, **kwargs):
        captured["records"] = list(records)
        captured.update(kwargs)
        return {"counts": {"train": 1, "valid": 1}}
    monkeypatch.setattr(
        "propevolve.reasoning_policy.dataset.write_supervised_dataset", write
    )

    result = collect_job(job_path)

    assert result["counts"] == {"train": 1, "valid": 1}
    assert len(captured["records"]) == 2
    assert captured["splits"] == {"train": [0, 10], "valid": [10, 20]}
    assert captured["lineage"]["source_identity"] == "source-id"
    assert captured["sealed_start_ns"] == 30


def test_publish_source_audit_authenticates_embedding_roles_and_teacher_order(
        tmp_path, monkeypatch):
    source = source_payload()
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps(source))
    context_path = tmp_path / "context.json"
    context_path.write_text("{}")
    config = {
        "source_recipe": source_path.name,
        "temporal_split_audit": "source-audit.json",
        "context_config": context_path.name,
        "sealed_start": "2026-01-01",
        "embedding_dim": 4,
        "specialist_score_mode": "out_of_fold",
        "specialist_supervision_roles": ["train"],
        "collection_policy": {"kind": "reset_states"},
    }
    monkeypatch.setattr(job, "read_job", lambda path: (config, tmp_path))
    monkeypatch.setattr(
        "propevolve.reasoning_policy.context.ContextConfig.load",
        lambda path: SimpleNamespace(input_mode="embeddings"),
    )
    market = SimpleNamespace(close=np.arange(3), embeddings=np.zeros((3, 4)))
    def load_role(config, root, source, role, *, include_specialists=True):
        sources = tuple(SimpleNamespace(kind=item["kind"]) for item in source["teachers"])
        return SimpleNamespace(markets={"NQ": market}), sources if include_specialists else ()
    monkeypatch.setattr(job, "load_role", load_role)

    audit = publish_source_audit(tmp_path / "job.json")

    assert audit["status"] == "PASS"
    assert audit["sealed_touched"] is False
    assert audit["roles"]["train"]["specialists"] == ["expansion", "regime", "trend"]
    assert audit["roles"]["valid"]["specialists"] == []
    assert json.loads((tmp_path / "source-audit.json").read_text()) == audit
    with pytest.raises(FileExistsError, match="already exists"):
        publish_source_audit(tmp_path / "job.json")


def test_collect_job_streams_source_embedding_rows_ticker_by_ticker(
        tmp_path, monkeypatch):
    job_path = tmp_path / "job.json"
    context_path = tmp_path / "context.json"
    job_path.write_text("{}")
    context_path.write_text("{}")
    config = {
        "context_config": context_path.name,
        "collection_policy": {"kind": "reset_states"},
        "dataset_kind": "action",
        "dataset_output": "dataset",
        "tickers": {"train": ["ES", "NQ"], "valid": ["ES", "NQ"]},
        "maximum_examples_per_episode": 3,
        "sample_stride": 1,
        "rollout_max_steps": 2,
        "target_temperature": 1.0,
        "opportunity_contract": {"horizon": 5, "target_rs": [2.0], "stop_r": -1.0},
        "economic_action_sampling": {
            "train": {"per_action": 1, "seed": 1},
            "valid": {"per_action": 1, "seed": 2},
        },
        "embedding_storage": "source_embedding_reference_v1",
        "action_supervision_scope": "trade_mastery",
    }
    source = {"teachers": [], "challenge": {}, "cache_root": "cache"}
    monkeypatch.setattr(job, "read_job", lambda path: (config, tmp_path))
    monkeypatch.setattr(job, "load_source_contract", lambda *args: (
        source, {"continuation": {"kind": "reset_states"}},
        {"train": [0, 10], "valid": [10, 20]}, 30, "source-id",
    ))
    monkeypatch.setattr(job, "collection_source_for_role",
                        lambda config, source, role, splits: (source, splits[role]))
    loaded_tickers = []
    # load_role receives role as the fourth positional argument.
    def role_loader(config, root, source, role, **kwargs):
        tickers = tuple(config["tickers"][role])
        loaded_tickers.append(tickers)
        return SimpleNamespace(name=tickers), ()
    monkeypatch.setattr(job, "load_role", role_loader)
    monkeypatch.setattr(job, "economic_episode_specs", lambda config, env, sources, role: [
        {"ticker": "ES", "start": 1, "expected_action": int(Action.WAIT)},
        {"ticker": "NQ", "start": 2, "expected_action": int(Action.ENTER_SHORT_1)},
    ])
    monkeypatch.setattr(
        "propevolve.reasoning_policy.context.ContextConfig.load",
        lambda path: SimpleNamespace(context_steps=2),
    )
    def collect_examples(*args, **kwargs):
        target = ("ENTER_SHORT_1" if kwargs["initial_entry_action"] is Action.ENTER_SHORT_1
                  else "WAIT")
        yield {"action": {"messages": [{"role": "assistant", "content": target}],
                          "targets": {"action_order": [target]}}}
    monkeypatch.setattr(
        "propevolve.reasoning_policy.collector.collect_examples", collect_examples
    )
    captured = {}
    def write(records, output, **kwargs):
        captured["records"] = list(records)
        captured.update(kwargs)
        return {"counts": {"train": 2, "valid": 2}}
    monkeypatch.setattr(
        "propevolve.reasoning_policy.dataset.write_supervised_dataset", write
    )

    assert collect_job(job_path)["counts"] == {"train": 2, "valid": 2}
    assert len(captured["records"]) == 4
    assert captured["embedding_source_cache_root"] == tmp_path / "cache"
    assert loaded_tickers.count(("ES",)) == 2
    assert loaded_tickers.count(("NQ",)) == 2


def test_job_cli_routes_metadata_and_dataset_stages(tmp_path, monkeypatch, capsys):
    config = {
        "dataset_output": "dataset",
        "sft_config": "sft.json",
        "mlx_view": "view",
    }
    monkeypatch.setattr(job, "read_job", lambda path: (config, tmp_path))
    monkeypatch.setattr(job, "readiness", lambda path: {
        "ready": True, "blockers": [], "training_started": False,
    })
    monkeypatch.setattr(job, "publish_source_audit", lambda path: {"status": "PASS"})
    monkeypatch.setattr(job, "collect_job", lambda path: {"counts": {"train": 2}})
    monkeypatch.setattr(job, "label_census_job", lambda path: {"eligible_rows": 3})
    monkeypatch.setattr(job, "load_source_contract",
                        lambda *args: ({}, {"specialist_score_mode": "out_of_fold"}, {}, 0, "id"))
    monkeypatch.setattr(
        "propevolve.reasoning_policy.dataset.audit_supervised_dataset",
        lambda path, **kwargs: {"status": "PASS", "mode": kwargs["specialist_score_mode"]},
    )
    monkeypatch.setattr(
        "propevolve.reasoning_policy.mlx_sft.verify_dataset",
        lambda path: {"counts": {"train": 2, "valid": 1}},
    )
    sft_commands = []
    monkeypatch.setattr(
        "propevolve.reasoning_policy.mlx_sft.main", lambda command: sft_commands.append(command)
    )

    for stage in ("check", "publish-source-audit", "collect", "census",
                  "publish-audit", "audit", "prepare", "train"):
        assert job.main(["--config", "job.json", stage]) == 0

    assert sft_commands == [
        ["--config", str(tmp_path / "sft.json"), "--root", str(tmp_path),
         "--view", str(tmp_path / "view")],
        ["--config", str(tmp_path / "sft.json"), "--root", str(tmp_path),
         "--view", str(tmp_path / "view"), "--train"],
    ]
    assert '"dataset_audit_verified": true' in capsys.readouterr().out


def test_job_check_returns_nonzero_when_resources_are_not_ready(tmp_path, monkeypatch):
    monkeypatch.setattr(job, "read_job", lambda path: ({}, tmp_path))
    monkeypatch.setattr(job, "readiness", lambda path: {
        "ready": False, "blockers": ["source_recipe"], "training_started": False,
    })

    assert job.main(["--config", "job.json", "check"]) == 1


def test_load_role_composes_market_teachers_volume_and_shared_environment(
        tmp_path, monkeypatch):
    verified = []
    asset = SimpleNamespace(market_data=tmp_path / "bars", verify=lambda: verified.append(True))
    market = SimpleNamespace(close=np.arange(3))
    base_sources = (SimpleNamespace(kind="expansion"),)
    volume = SimpleNamespace(kind="volume")
    captured = {}
    monkeypatch.setattr("propevolve.assets.AssetContract.load", lambda path: asset)
    monkeypatch.setattr("propevolve.cache.load_market_series",
                        lambda *args, **kwargs: market)
    monkeypatch.setattr(
        "propevolve.teachers.composition.load_teacher_targets",
        lambda *args, **kwargs: SimpleNamespace(sources=base_sources),
    )
    monkeypatch.setattr(
        "propevolve.reasoning_policy.specialist_cache.load_specialist_cache",
        lambda *args, **kwargs: volume,
    )
    monkeypatch.setattr(
        "propevolve.reasoning_policy.context.ContextConfig.load",
        lambda path: SimpleNamespace(fields=("volume.participation",)),
    )
    monkeypatch.setattr("propevolve.environment.ChallengeSpec",
                        lambda **values: SimpleNamespace(**values))
    monkeypatch.setattr(
        "propevolve.observation.TradeManagementObservationSpec.from_config",
        lambda values: SimpleNamespace(**values),
    )
    def environment(markets, **kwargs):
        captured.update(markets=markets, **kwargs)
        return SimpleNamespace(markets=markets)
    monkeypatch.setattr("propevolve.environment.HistoricalChallengeEnv", environment)
    config = {
        "tickers": {"train": ["NQ"]}, "seed": 7,
        "context_config": "context.json",
        "volume_source": {"manifest": "volume.json", "audit": "volume-audit.json"},
    }
    source = {
        "assets": "assets.json", "timeframe_minutes": 3, "cache_root": "cache",
        "teachers": [{"kind": "expansion"}],
        "temporal": {"train_start": "2021-01-01", "train_end": "2022-01-01"},
        "point_values": {"NQ": 20.0}, "round_trip_fees": {"NQ": 3.84},
        "challenge": {"profit_target": 6000.0}, "observation": {"management_state": "off"},
    }

    env, sources = load_role(config, tmp_path, source, "train")

    assert verified == [True]
    assert env.markets == {"NQ": market}
    assert [item.kind for item in sources] == ["expansion", "volume"]
    assert captured["tick_values"] == {"NQ": 20.0}
    assert captured["round_trip_fees"] == {"NQ": 3.84}


def test_evaluation_job_runs_teacher_free_trade_and_challenge_boundaries(
        tmp_path, monkeypatch):
    for name, payload in {
        "policy.json": {},
        "context.json": {},
        "challenge-metrics.json": {"near_blow_headroom_fraction": 0.1},
        "trade-metrics.json": {"minimum_macro_accuracy": 0.4},
    }.items():
        (tmp_path / name).write_text(json.dumps(payload))
    records = tmp_path / "trade-records.jsonl"
    records.write_text(json.dumps({
        "completed_at_ns": 110, "label_end_ns": 120, "messages": [],
    }) + "\n")
    config = {
        "policy_config": None,
        "evaluation_policy_config": "policy.json",
        "evaluation_metrics_config": "challenge-metrics.json",
        "trade_mastery_metrics_config": "trade-metrics.json",
        "trade_mastery_audit": {
            "records": records.name, "sha256": file_digest(records),
            "maximum_records": 1, "role_start_ns": 100, "role_end_ns": 200,
        },
        "evaluation_output": "evaluation.json",
        "evaluation_decisions": "decisions.jsonl",
        "context_config": "context.json",
        "rollout_max_steps": 8,
    }
    policy = SimpleNamespace(requires_specialists=False)
    monkeypatch.setattr(job, "read_job", lambda path: (config, tmp_path))
    monkeypatch.setattr(job, "load_source_contract", lambda *args: (
        {}, {}, {"train": [0, 100], "valid": [100, 200]}, 300, "source-id"
    ))
    monkeypatch.setattr(
        "propevolve.reasoning_policy.policy.MLXActionPolicy.from_config",
        lambda *args, **kwargs: policy,
    )
    monkeypatch.setattr(job, "load_role", lambda *args, **kwargs: (object(), ()))
    monkeypatch.setattr(job, "configured_episodes",
                        lambda *args, **kwargs: [{"ticker": "NQ", "start": 1}])
    monkeypatch.setattr(
        "propevolve.reasoning_policy.context.ContextConfig.load",
        lambda path: SimpleNamespace(input_mode="embeddings"),
    )
    monkeypatch.setattr(
        "propevolve.reasoning_policy.rl.require_challenge_mastery_context", lambda value: value
    )
    def evaluate(*args, **kwargs):
        kwargs["on_decision"]({"action": "WAIT"})
        return {
            "trade_mastery": {"macro_accuracy": 0.6},
            "challenge_mastery": {"pass_rate": 0.5, "blow_rate": 0.0},
        }
    monkeypatch.setattr(
        "propevolve.reasoning_policy.evaluation.evaluate_responsibilities", evaluate
    )
    monkeypatch.setattr(
        "propevolve.reasoning_policy.selection.assess_trade_mastery",
        lambda metrics, criteria: {"accepted": True},
    )
    monkeypatch.setattr(
        "propevolve.reasoning_policy.selection.assess_candidate",
        lambda metrics, criteria: {"accepted": False},
    )

    assert job.main(["--config", "job.json", "evaluate"]) == 0
    report = json.loads((tmp_path / "evaluation.json").read_text())
    assert report["source_identity"] == "source-id"
    assert report["selection"] == {
        "trade_mastery": {"accepted": True},
        "challenge_mastery": {"accepted": False},
    }
    assert json.loads((tmp_path / "decisions.jsonl").read_text()) == {"action": "WAIT"}
