"""Configuration-driven economic screening, never automatic model promotion."""
import math


def assess_candidate(report, criteria):
    failures = []
    episodes = report.get("episodes", [])
    if len(episodes) < criteria["minimum_episodes"]:
        failures.append("insufficient_episodes")
    if criteria["require_teacher_free"] and report.get("teacher_free") is not True:
        failures.append("teacher_dependence")
    if report.get("near_blow_headroom_fraction") != criteria["near_blow_headroom_fraction"]:
        failures.append("near_blow_definition_mismatch")
    for metric, bound, minimum in (
        ("pass_rate", criteria["minimum_pass_rate"], True),
        ("blow_rate", criteria["maximum_blow_rate"], False),
        ("near_blow_rate", criteria["maximum_near_blow_rate"], False),
    ):
        value = report.get(metric)
        if (type(value) not in (float, int) or not math.isfinite(value)
                or not 0 <= value <= 1 or (value < bound if minimum else value > bound)):
            failures.append(metric)
    for side in ("long", "short"):
        if report.get("action_counts", {}).get(f"ENTER_{side.upper()}_1", 0) < criteria[f"minimum_{side}_entries"]:
            failures.append(f"{side}_participation")
    return {"verdict": "REJECT" if failures else "REVIEW_CANDIDATE", "failures": failures,
            "promoted": False, "note": "Observed development screening only; frozen temporal confirmation is still required."}
