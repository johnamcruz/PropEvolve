"""Deprecated direct-action inference must not remain a runnable alternative."""
import ast
from pathlib import Path


def test_no_production_entrypoint_uses_retired_action_scorer():
    root = Path(__file__).resolve().parents[1]
    retired = root / "src/propevolve/reasoning_policy/policy.py"
    assert not retired.exists(), "delete the retired direct-action scorer"
    forbidden = {"MLXActionPolicy", "sequence_scores", "completion_scores", "tokenize_completions"}
    for folder in (root / "src", root / "scripts"):
        for path in folder.rglob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                name = (node.id if isinstance(node, ast.Name) else
                        node.attr if isinstance(node, ast.Attribute) else
                        node.name if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.alias)) else None)
                assert name not in forbidden, f"retired scorer reference: {path}:{node.lineno}"
