"""Staged epoch selection and final promotion use identical frozen evidence."""
import json
import numpy as np


def test_epoch_and_final_promotion_agree_for_staged_assessments(tmp_path):
    from test_reasoning_corrective_campaign import ACTIONS, assessment, gate
    from propevolve.decision import Action
    from propevolve.reasoning_policy.staged_policy import legal_action_log_probs
    from propevolve.reasoning_policy.corrective_campaign import (
        compare_frozen_assessments, evaluate_checkpoint_candidate)
    before, after = tmp_path / "parent", tmp_path / "candidate"
    for path, weak, strong in ((before, -.5, 1.), (after, .2, .8)):
        labels = [(action, advantage) for action in ACTIONS for advantage in (weak, strong)]
        assessment(path, labels, primary=weak)
        rows = [json.loads(line) for line in (path / "scores.jsonl").read_text().splitlines()]
        for row in rows:
            target, a = row["target"], row["target_advantage"]
            logits = ([-a, 0., 0.] if target == "WAIT" else
                      [a, a, 0.] if target == "ENTER_LONG_1" else
                      [a, -a, 0.] if target == "ENTER_SHORT_1" else
                      [0., 0., a] if target == "HOLD" else [0., 0., -a])
            row["assessment"] = dict(zip(("entry", "direction", "management"), logits))
            row["score_type"] = "log_probability"
            names = list(row["scores"])
            row["scores"] = dict(zip(names, legal_action_log_probs(np.array(logits),
                [Action[n] for n in names], xp=np).tolist()))
        (path / "scores.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        summary = json.loads((path / "summary.json").read_text())
        summary["metrics"]["decision_boundary_semantics"] = "staged_independent_binary_v1"
        (path / "summary.json").write_text(json.dumps(summary))
    candidates = [json.loads(line) for line in (after / "scores.jsonl").read_text().splitlines()]
    prepared = [{"target_name": r["target"], "action_targets": {"names": list(r["scores"])}}
                for r in candidates]
    scored = {r["index"]: {"assessment": list(r["assessment"].values()),
                          "log_probs": list(r["scores"].values())} for r in candidates}
    metrics = json.loads((after / "summary.json").read_text())["metrics"]
    during = evaluate_checkpoint_candidate(before, prepared, scored, metrics, gate())
    final = compare_frozen_assessments(before, after, gate())
    assert during == final
    assert final["decision"] == "ACCEPTED"
