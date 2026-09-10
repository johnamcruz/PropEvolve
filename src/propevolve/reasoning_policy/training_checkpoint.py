"""Small resumable learner snapshots, separate from inference adapters."""
import json
from pathlib import Path


def save_training_state(destination, model, optimizer, receipt):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    arrays = {}

    def encode(value):
        if isinstance(value, mx.array):
            key = f"a{len(arrays)}"
            arrays[key] = value
            return {"array": key}
        if isinstance(value, dict):
            return {"dict": [[key, encode(item)] for key, item in value.items()]}
        if isinstance(value, (list, tuple)):
            return {"tuple" if isinstance(value, tuple) else "list": [encode(x) for x in value]}
        if value is None or isinstance(value, (str, int, float, bool)):
            return {"scalar": value}
        raise TypeError(f"unsupported training state: {type(value)}")

    state = encode({"optimizer": optimizer.state, "random": list(mx.random.state)})
    mx.save_safetensors(str(destination / "weights.safetensors"),
                        dict(tree_flatten(model.trainable_parameters())))
    mx.save_safetensors(str(destination / "state.safetensors"), arrays)
    (destination / "receipt.json").write_text(json.dumps(
        {"state": state, "receipt": receipt}, allow_nan=False))


def load_training_state(source, model, optimizer):
    import mlx.core as mx
    source = Path(source)
    metadata = json.loads((source / "receipt.json").read_text())
    arrays = mx.load(str(source / "state.safetensors"))

    def decode(value):
        if "array" in value:
            return arrays[value["array"]]
        if "dict" in value:
            return {key: decode(item) for key, item in value["dict"]}
        if "list" in value:
            return [decode(item) for item in value["list"]]
        if "tuple" in value:
            return tuple(decode(item) for item in value["tuple"])
        return value["scalar"]

    state = decode(metadata["state"])
    model.load_weights(str(source / "weights.safetensors"), strict=False)
    optimizer.state = state["optimizer"]
    if len(mx.random.state) != len(state["random"]):
        raise ValueError("checkpoint random-state layout differs from this MLX runtime")
    for current, saved in zip(mx.random.state, state["random"]):
        current[:] = saved
    mx.eval(model.parameters(), optimizer.state, mx.random.state)
    return metadata["receipt"]
