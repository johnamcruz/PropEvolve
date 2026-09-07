"""MLX optimizer/RNG trees in JSON + safetensors; never pickle executable state."""
import json
from pathlib import Path
from .integrity import file_digest


def seal_checkpoint(path):
    path = Path(path)
    files = {item.name: file_digest(item) for item in path.iterdir() if item.is_file()}
    (path / "checkpoint_manifest.json").write_text(json.dumps(files, indent=2))


def verify_checkpoint(path):
    path = Path(path)
    files = json.loads((path / "checkpoint_manifest.json").read_text())
    required = {"adapters.safetensors", "adapter_config.json", "rl_receipt.json", "training.json", "training.safetensors"}
    if not required.issubset(files):
        raise ValueError("incomplete RL training checkpoint")
    for name, digest in files.items():
        if Path(name).name != name or file_digest(path / name) != digest:
            raise ValueError("RL checkpoint content changed")
    return json.loads((path / "rl_receipt.json").read_text())


def prune_checkpoints(root, *, keep, contract, protected=()):
    """Remove only sealed checkpoints of this exact contract; no recursive delete."""
    if type(keep) is not int or keep < 1:
        raise ValueError("checkpoint_keep must be positive")
    candidates = []
    root = Path(root)
    protected = {Path(path).resolve() for path in protected}
    for path in root.iterdir():
        if not path.is_dir() or path.is_symlink() or path.resolve() in protected:
            continue
        if not (path / "checkpoint_manifest.json").is_file():
            continue
        receipt = verify_checkpoint(path)
        if receipt.get("contract") != contract:
            continue
        manifest = json.loads((path / "checkpoint_manifest.json").read_text())
        if {item.name for item in path.iterdir()} != set(manifest) | {"checkpoint_manifest.json"}:
            continue  # unrelated files make this directory ineligible for cleanup
        runtime = json.loads((path / "training.json").read_text())["runtime"]
        candidates.append((runtime["next_group"], path, manifest))
    removed = []
    for _, path, manifest in sorted(candidates, key=lambda item: item[0])[:-keep]:
        for name in (*manifest, "checkpoint_manifest.json"):
            (path / name).unlink()
        path.rmdir()
        removed.append(path.name)
    return removed


def save_training_state(path, *, optimizer, runtime):
    import mlx.core as mx
    arrays = {}
    def encode(value):
        if isinstance(value, mx.array):
            key = str(len(arrays))
            arrays[key] = value
            return {"type": "array", "key": key}
        if isinstance(value, dict):
            if any(not isinstance(key, str) for key in value):
                raise ValueError("checkpoint dictionary keys must be strings")
            return {"type": "dict", "items": {key: encode(item) for key, item in value.items()}}
        if isinstance(value, (tuple, list)):
            return {"type": "tuple" if isinstance(value, tuple) else "list", "items": [encode(item) for item in value]}
        if value is None or type(value) in (str, int, float, bool):
            return {"type": "scalar", "value": value}
        raise ValueError(f"unsupported checkpoint value type {type(value).__name__}")
    payload = {"schema": "reasoning_rl_training_state_v1",
        "tree": encode({"optimizer": optimizer.state, "mlx_rng": mx.random.state}),
        "runtime": runtime}
    serialized = json.dumps(payload, allow_nan=False, indent=2)
    mx.eval(arrays)
    mx.save_safetensors(str(Path(path) / "training.safetensors"), arrays)
    (Path(path) / "training.json").write_text(serialized)


def restore_training_state(path, *, optimizer):
    import mlx.core as mx
    path = Path(path)
    verify_checkpoint(path)
    payload = json.loads((path / "training.json").read_text())
    if payload.get("schema") != "reasoning_rl_training_state_v1":
        raise ValueError("unsupported training checkpoint schema")
    arrays = mx.load(str(path / "training.safetensors"))
    def decode(node):
        kind = node["type"]
        if kind == "array":
            return arrays[node["key"]]
        if kind == "dict":
            return {key: decode(item) for key, item in node["items"].items()}
        if kind in {"tuple", "list"}:
            items = [decode(item) for item in node["items"]]
            return tuple(items) if kind == "tuple" else items
        if kind == "scalar":
            return node["value"]
        raise ValueError("invalid training checkpoint tree")
    state = decode(payload["tree"])
    optimizer.state = state["optimizer"]
    mx.random.state = state["mlx_rng"]
    mx.eval(optimizer.state, mx.random.state)
    return payload["runtime"]
