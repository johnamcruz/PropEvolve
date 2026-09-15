"""Native MLX campaign integration, not evidence of real-market learnability."""
import copy
import json
from pathlib import Path

import pytest


def test_native_staged_campaign_assesses_corrects_checkpoints_and_resumes(tmp_path):
    pytest.importorskip("mlx.core")
    from mlx_lm import load
    from test_reasoning_local_qlora_e2e import tiny_quantized_qwen
    from test_reasoning_prepared_full_action_e2e import prepared_action_view
    from test_staged_queries import settings
    from propevolve.reasoning_policy.dataset import write_supervised_dataset
    from propevolve.reasoning_policy.integrity import file_digest
    from propevolve.reasoning_policy.mlx_sft import prepare_mlx_view, train_prepared
    from propevolve.reasoning_policy.corrective_campaign import run_campaign

    model = tiny_quantized_qwen(tmp_path / "model", tied=True)
    _, tokenizer = load(model)
    _, original, config = prepared_action_view(tmp_path, embeddings=True,
        model=model, tokenizer=tokenizer)
    actions = ("WAIT", "ENTER_LONG_1", "ENTER_SHORT_1", "HOLD", "CLOSE")
    records = []
    start, day = original["completed_at_ns"], 86400 * 10**9
    for role in range(2):
        for i, action in enumerate(actions):
            row = copy.deepcopy(original)
            row.update(source_id=f"{role}-{action}", ticker="NQ",
                completed_at_ns=start + role * day + i * 100,
                label_end_ns=start + role * day + i * 100 + 10)
            names = list(actions[:3] if i < 3 else actions[3:])
            row["messages"][1]["content"] = json.dumps({
                "fields": ["trade.current_r"], "history_oldest_first": [[0.]],
                "legal_actions": names})
            row["messages"][-1]["content"] = action
            row["targets"] = {"action_order": names,
                "action_probabilities": [float(n == action) for n in names],
                "outcomes": {n: {"reward_to_go": 0. if n == "WAIT" else 2. if n == action else -1.}
                             for n in names},
                "specialist_targets": {f"{g}.signal": .5 for g in ("expansion", "trend", "regime", "volume")}}
            records.append(row)
    data = tmp_path / "staged-data"
    write_supervised_dataset(records, data,
        splits={"train": [start, start + day], "valid": [start + day, start + 2 * day]},
        sealed_start_ns=start + 2 * day,
        lineage={"source_identity": "fixture", "specialist_identities": "fixture",
                 "economic_contract": "fixture", "split_audit": "fixture"})
    (data / "audit.json").write_text(json.dumps({"status": "PASS",
        "manifest_sha256": file_digest(data / "manifest.json"),
        "specialist_score_mode": "post_fit", "sealed_touched": False,
        "actions_by_role": {role: {action: 1 for action in actions}
                            for role in ("train", "valid")}}))
    staged = settings()
    staged["state_fields"] = ["trade.current_r"]
    staged["market"]["channels"] = [{"name": f"{g}.signal", "query": f"{g}?", "weight": 1.}
                                    for g in ("expansion", "trend", "regime", "volume")]
    config.update(architecture="staged_reasoning_v1", staged_policy=staged,
        interpretation_loss_weight=.5, selection="hierarchical_greedy", data=str(data),
        adapter_path=str(tmp_path / "parent-adapter"), iters=1, batch_size=5,
        trainable_components=["lora", "projector"], learning_rate=1e-4,
        chat_template_kwargs={"enable_thinking": False}, steps_per_report=1,
        steps_per_eval=1, save_every=1, val_batches=5, seed=11,
        decision_objective="hierarchical_binary", stage_role="trade_mastery",
        dataset_requirements={"minimum_rows_per_action": {
            role: {action: 1 for action in actions} for role in ("train", "valid")},
            "expected_splits": {"train": [start, start + day],
                                "valid": [start + day, start + 2 * day]},
            "sealed_start_ns": start + 2 * day},
        early_stopping={"enabled": True, "patience_evaluations": 2, "min_delta": 0.,
                        "restore_best": True, "monitor": "worst_task_advantage", "mode": "max"})
    parent = tmp_path / "parent.json"
    parent.write_text(json.dumps(config))
    view = tmp_path / "staged-view"
    prepare_mlx_view(parent, view, tokenizer=tokenizer)
    train_prepared(parent, view)
    template = tmp_path / "template.json"
    template.write_text(json.dumps({**config, "batch_sampling": "balanced_actions"}))
    acceptance = {"primary_metric": "worst_task_advantage", "minimum_primary_improvement": .01,
        "minimum_mean_mistake_advantage_delta": .01, "minimum_retained_mastery_rate": .8,
        "maximum_per_action_mistake_regression": .05, "maximum_per_task_advantage_regression": .05}
    campaign = tmp_path / "campaign.json"
    campaign.write_text(json.dumps({
        "schema": "propevolve_reasoning_corrective_campaign_v2",
        "workspace_root": str(tmp_path), "state_file": "campaign/state.json",
        "output_root": "campaign", "initial_policy_config": str(parent),
        "sft_template_config": str(template), "prepared_view": str(view), "rounds": 1,
        "initial_assessments": {"train": None, "valid": None}, "preserved_assessments": [],
        "required_teacher_groups": ["expansion", "trend", "regime", "volume"],
        "subset": {"rows_per_group": 2, "mistake_fraction": .5, "seed": 11,
                   "balance_mode": "hierarchical_boundaries"},
        "mastered_anchor_retention": {"loss_weight": 1., "temperature": 1.},
        "rejection_adaptation": {"enabled": True}, "acceptance": acceptance,
        "timeouts": {"assessment_seconds": 120, "training_seconds": 120}}))

    class NativePhases:
        """In-process native engine; subprocess lifecycle is tested separately."""
        def assess(self, policy, prepared, role, output, log):
            from propevolve.reasoning_policy.learning_audit import assess_prepared
            assess_prepared(policy, prepared, role=role, output=output)

        def train(self, recipe, prepared, log):
            train_prepared(recipe, prepared)

    result = run_campaign(campaign, phases=NativePhases())
    assert result["status"] == "COMPLETE"
    round_state = result["rounds"][0]
    assert round_state["decision"] in {"ACCEPTED", "REJECTED"}
    assert round_state["candidate_train"] and round_state["candidate_valid"]
    assert set(round_state["per_boundary"]) == {
        "entry.ENTER", "entry.WAIT", "direction.LONG", "direction.SHORT",
        "management.HOLD", "management.CLOSE"}
    child = json.loads(Path(round_state["candidate_policy_config"]).read_text())
    assert child["resume_adapter_requirements"]["staged_policy"] == staged
    assert run_campaign(campaign, phases=NativePhases()) == result

    # The accepted SFT artifact (or safe parent fallback) starts challenge RL.
    # Complete causal account context must be added without changing the market
    # interpretation or discarding its trained weights.
    import numpy as np
    from propevolve.reasoning_policy.context import ContextConfig
    from propevolve.reasoning_policy.mlx_sft import read_sft_config
    from propevolve.reasoning_policy.rl import (
        CHALLENGE_MASTERY_FIELDS, load_challenge_policy, rollout, MLXAdapterLearner)
    from propevolve.reasoning_policy.checkpoints import restore_training_state
    from test_reasoning_challenger_e2e import environment
    selected = read_sft_config(result["current_policy_config"])
    original_fields = list(selected["staged_policy"]["state_fields"])
    context = ContextConfig(3, CHALLENGE_MASTERY_FIELDS, input_mode="embeddings")
    actor = load_challenge_policy(selected, context)
    assert selected["staged_policy"]["state_fields"] == original_fields
    assert actor.settings["stage_role"] == "challenge_mastery"
    assert actor.settings["staged_policy"]["market"] == staged["market"]
    decisions, _ = rollout(actor, environment(), options={"ticker": "NQ", "start": 0},
        context_config=context, sources=(), rng=np.random.default_rng(11), max_steps=8)
    rl_settings = {"seed": 11, "learning_rate": 1e-5, "weight_decay": 0.,
        "max_update_rows": 1, "epochs": 1, "minibatch_size": 1,
        "clip_epsilon": .2, "kl_weight": .01, "entropy_weight": 0., "max_grad_norm": 1.}
    learner = MLXAdapterLearner(actor, rl_settings)
    row = decisions[0]
    learner.update([(row, 1.)], np.random.default_rng(1))
    checkpoint = tmp_path / "rl-checkpoint"
    learner.save(checkpoint, selected["adapter_path"], {"contract": "fixture"},
        runtime={"next_group": 1, "metrics": [], "rng": np.random.default_rng(1).bit_generator.state})
    from propevolve.decision import Action
    legal = [Action[name] for name in row.actions]
    expected = actor.assess(row.staged_context, legal)
    resumed = load_challenge_policy(selected, context, resume_checkpoint=checkpoint)
    assert resumed.assess(row.staged_context, legal) == expected
    resumed_learner = MLXAdapterLearner(resumed, rl_settings)
    assert restore_training_state(checkpoint, optimizer=resumed_learner.optimizer)["next_group"] == 1
    learner.update([(row, 1.)], np.random.default_rng(2))
    resumed_learner.update([(row, 1.)], np.random.default_rng(2))
    actual = resumed.assess(row.staged_context, legal)
    expected = actor.assess(row.staged_context, legal)
    np.testing.assert_allclose(list(actual["log_probs"].values()),
        list(expected["log_probs"].values()), rtol=1e-6, atol=1e-6)

    # Resume must reject a different assessment contract even when tensor
    # dimensions and checkpoint files are still compatible.
    changed_context = ContextConfig(3, tuple(reversed(CHALLENGE_MASTERY_FIELDS)),
                                    input_mode="embeddings")
    with pytest.raises(ValueError, match="staged_policy"):
        load_challenge_policy(selected, changed_context, resume_checkpoint=checkpoint)
    with pytest.raises(ValueError, match="embedding history"):
        load_challenge_policy(selected,
            ContextConfig(4, CHALLENGE_MASTERY_FIELDS, input_mode="embeddings"))
