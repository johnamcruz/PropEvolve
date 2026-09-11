import json

import pytest


def settings(**updates):
    value = dict(assessment_path="unused", scores_sha256="0" * 64,
                 summary_sha256="1" * 64, rows_per_group=2,
                 mistake_fraction=.5, seed=17)
    value.update(updates)
    return value


def test_subset_keeps_mistakes_and_mastered_examples_without_crossing_training_role():
    from propevolve.reasoning_policy.targeted_subset import select_training_indices
    rows = [dict(index=i, ticker="NQ", target="ENTER_LONG_1", completed_at_ns=100+i,
                 target_advantage=advantage) for i, advantage in enumerate([-2., -1., .5, 1.])]
    options = settings()
    result = select_training_indices(rows, options, train_bounds=(100, 200))
    assert len(result) == 2
    assert len(set(result) & {0, 1}) == 1
    assert len(set(result) & {2, 3}) == 1
    assert result == select_training_indices(rows, options, train_bounds=(100, 200))
    with pytest.raises(ValueError, match="training"):
        select_training_indices(rows, options, train_bounds=(101, 200))


def test_subset_preserves_rare_exit_and_failure_groups_and_rejects_duplicate_rows():
    from propevolve.reasoning_policy.targeted_subset import select_training_indices
    rows = [dict(index=i, ticker=ticker, target=action, completed_at_ns=100+i,
                 target_advantage=advantage) for i, (ticker, action, advantage) in enumerate([
                     ("NQ", "CLOSE", -1.), ("ES", "WAIT", .5),
                     ("NQ", "ENTER_SHORT_1", -2.), ("NQ", "ENTER_SHORT_1", .3)])]
    options = settings(mistake_fraction=.75, seed=1)
    assert select_training_indices(rows, options, train_bounds=(100, 200)) == [0, 1, 2, 3]
    with pytest.raises(ValueError):
        select_training_indices(rows + rows[:1], options, train_bounds=(100, 200))


def test_completed_assessment_rotates_mistakes_and_retention_without_changing_actions(tmp_path):
    from propevolve.reasoning_policy.integrity import file_digest
    from propevolve.reasoning_policy.targeted_subset import TargetedSampler
    view = tmp_path / "view_manifest.json"
    view.write_text("{}")
    assessment = tmp_path / "assessment"
    assessment.mkdir()
    rows = [dict(index=i, ticker="NQ", target=target, completed_at_ns=100+i,
                 target_advantage=advantage) for i, (target, advantage) in enumerate([
                     ("ENTER_LONG_1", -2.), ("ENTER_LONG_1", 1.),
                     ("ENTER_SHORT_1", -1.), ("ENTER_SHORT_1", .5)])]
    scores = assessment / "scores.jsonl"
    scores.write_text("".join(json.dumps(row) + "\n" for row in rows))
    summary = assessment / "summary.json"
    summary.write_text(json.dumps({"role": "train", "rows": 4,
        "weights_updated": False, "view_manifest_sha256": file_digest(view)}))
    options = settings(assessment_path=str(assessment),
        scores_sha256=file_digest(scores), summary_sha256=file_digest(summary))
    sampler = TargetedSampler.from_assessment(options,
        view_manifest_path=view, train_bounds=(100, 200), expected_rows=4)
    order = list(map(int, sampler.order(0)))
    assert len(order) == 4
    assert len(set(order) & {0, 1}) == 2
    assert len(set(order) & {2, 3}) == 2
    scores.write_text(scores.read_text() + "\n")
    with pytest.raises(ValueError, match="changed"):
        TargetedSampler.from_assessment(options,
            view_manifest_path=view, train_bounds=(100, 200), expected_rows=4)


def test_targeted_rounds_refresh_examples_and_reject_incomplete_assessment():
    from propevolve.reasoning_policy.targeted_subset import TargetedSampler
    rows = [dict(index=i, ticker="NQ", target=target, completed_at_ns=100+i,
                 target_advantage=(-1. if i % 2 == 0 else 1.))
            for i, target in enumerate(
                ["ENTER_LONG_1"] * 6 + ["ENTER_SHORT_1"] * 6)]
    sampler = TargetedSampler(rows, settings(), train_bounds=(100, 200),
                              expected_rows=12)
    first = set(map(int, sampler.order(0)))
    second = set(map(int, sampler.order(1)))
    assert first != second
    assert len(first) == len(second) == 4
    receipt = sampler.selection_receipt(0)
    assert receipt["draw_count"] == receipt["unique_rows"] == 4
    assert receipt["mistake_draws"] == receipt["anchor_draws"] == 2
    assert receipt["per_action"]["ENTER_LONG_1"] == {
        "mistake_draws": 1, "anchor_draws": 1}
    assert receipt["per_ticker"]["NQ"] == {
        "mistake_draws": 2, "anchor_draws": 2}
    assert {row["kind"] for row in receipt["draws"]} == {"mistake", "anchor"}
    assert {row["feedback"] for row in receipt["draws"]} == {
        "incorrect: predicted None; target is ENTER_LONG_1",
        "incorrect: predicted None; target is ENTER_SHORT_1",
        "correct: retain ENTER_LONG_1 above alternatives",
        "correct: retain ENTER_SHORT_1 above alternatives",
    }
    with pytest.raises(ValueError, match="exactly"):
        TargetedSampler(rows[:-1], settings(), train_bounds=(100, 200),
                        expected_rows=12)


def test_sft_json_enables_only_authenticated_targeted_action_sampling(tmp_path):
    from propevolve.reasoning_policy.mlx_sft import read_sft_config
    payload = {
        "model": "fixture-model", "data": "fixture-data",
        "adapter_path": "adapter", "train": True, "fine_tune_type": "lora",
        "mask_prompt": True, "num_layers": 1, "batch_size": 1, "iters": 5,
        "learning_rate": 1e-5, "max_seq_length": 64, "grad_checkpoint": True,
        "grad_accumulation_steps": 5, "trust_remote_code": False,
        "lora_parameters": {"rank": 2, "scale": 4., "dropout": 0.},
        "action_supervision": {"enabled": True, "soft_target_weight": 1.,
                               "ranking_weight": 1., "margin": .25},
        "batch_sampling": "balanced_actions",
        "early_stopping": {"enabled": True, "patience_evaluations": 2,
                           "min_delta": 0., "restore_best": True,
                           "monitor": "worst_action_advantage", "mode": "max"},
        "targeted_sampling": settings(assessment_path="assessment"),
    }
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(payload))
    selected = read_sft_config(path, root=tmp_path)
    assert selected["targeted_sampling"]["assessment_path"] == str(
        (tmp_path / "assessment").resolve())
    path.write_text(json.dumps({**payload, "action_supervision": {
        **payload["action_supervision"], "enabled": False}}))
    with pytest.raises(ValueError, match="targeted sampling"):
        read_sft_config(path, root=tmp_path)


def test_targeted_sampler_drives_real_mlx_training_batches():
    pytest.importorskip("mlx.core")
    from itertools import islice
    from propevolve.reasoning_policy.supervised_trainer import tensor_batches
    from propevolve.reasoning_policy.targeted_subset import TargetedSampler
    targets = ["ENTER_LONG_1", "ENTER_LONG_1", "ENTER_SHORT_1", "ENTER_SHORT_1"]
    scored = [dict(index=i, ticker="NQ", target=target, completed_at_ns=100+i,
                   target_advantage=(-1. if i % 2 == 0 else 1.))
              for i, target in enumerate(targets)]
    sampler = TargetedSampler(scored, settings(), train_bounds=(100, 200),
                              expected_rows=4)
    rows = []
    for index, target in enumerate(targets):
        rows.append({"tokens": [1, 2, 3], "offset": 1,
            "alternatives": [([1, 2, 3], 1), ([1, 4, 3], 1), ([1, 5, 3], 1)],
            "action_targets": {"names": ["WAIT", "ENTER_LONG_1", "ENTER_SHORT_1"],
                               "probabilities": [0., .5, .5], "values": [0., 1., -1.]},
            "target_name": target, "causal_state": [],
            "market_embeddings": [[float(index), 0.]], "market_available": [True]})
    batches = list(islice(tensor_batches(
        rows, 2, 8, loop=True, seed=17, targeted_sampler=sampler), 2))
    observed = [int(value) for batch in batches
                for value in batch[-2][:, 0, 0].tolist()]
    assert sorted(observed) == [0, 1, 2, 3]
