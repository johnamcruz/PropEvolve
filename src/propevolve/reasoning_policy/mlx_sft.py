"""Audited, prompt-masked QLoRA through MLX-LM's native trainer.

Importing this module loads neither MLX nor a model. Preparing/launching are
explicit operations, never side effects of importing the C51 application.
"""

import argparse
import json
from pathlib import Path
import math
from dataclasses import dataclass
from .integrity import file_digest
from .model_config import validate_model_settings, model_defaults, read_recipe
from .tokenization import encode_completion


def read_sft_config(path: str | Path) -> dict:
    payload = {**model_defaults(), **read_recipe(path)}
    required = {
        "model", "data", "adapter_path", "train", "fine_tune_type", "mask_prompt",
        "num_layers", "batch_size", "iters", "learning_rate", "max_seq_length",
        "grad_checkpoint", "grad_accumulation_steps", "lora_parameters",
    }
    if not required.issubset(payload):
        raise ValueError(f"missing SFT settings: {sorted(required - set(payload))}")
    validate_model_settings(payload)
    if payload["adapter_path"] is None:
        raise ValueError("SFT requires an adapter output path")
    if (payload["fine_tune_type"] != "lora" or payload["train"] is not True
            or payload["mask_prompt"] is not True or payload.get("trust_remote_code") is not False):
        raise ValueError("challenger requires prompt-masked LoRA and no remote code")
    for name in ("num_layers", "batch_size", "iters", "max_seq_length", "grad_accumulation_steps"):
        if type(payload[name]) is not int or payload[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    rank = payload["lora_parameters"].get("rank")
    if type(rank) is not int or rank < 1:
        raise ValueError("LoRA rank must be a positive integer")
    if (isinstance(payload["learning_rate"], bool)
            or not math.isfinite(float(payload["learning_rate"])) or payload["learning_rate"] <= 0):
        raise ValueError("learning_rate must be finite and positive")
    return payload


def verify_dataset(path: str | Path) -> dict:
    """Reject unreviewed or changed data before loading any large model."""
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text())
    audit = json.loads((root / "audit.json").read_text())
    manifest_hash = file_digest(root / "manifest.json")
    if (manifest.get("schema") != "propevolve_reasoning_dataset_v1"
            or audit.get("status") != "PASS"
            or audit.get("manifest_sha256") != manifest_hash
            or audit.get("specialist_score_mode") not in {"out_of_fold", "post_fit"}
            or audit.get("sealed_touched") is not False):
        raise ValueError("dataset needs a matching passed causal/specialist audit")
    sealed_start = manifest.get("sealed_start_ns")
    if (type(sealed_start) is not int or any(
            bounds[1] > sealed_start for bounds in manifest["splits"].values())):
        raise ValueError("dataset crosses sealed final-validation boundary")
    for role in ("train", "valid"):
        filename = root / f"{role}.jsonl"
        digest = file_digest(filename)
        if digest != manifest["files"][role]:
            raise ValueError(f"{role} dataset changed after audit")
    return manifest


def prepare_mlx_view(config_path: str | Path, output: str | Path, *, tokenizer) -> Path:
    """Make encoded native datasets without target/prompt leakage/truncation.

The supplied tokenizer is the model's actual tokenizer. This is a deliberate
system boundary for tests; the production caller loads it with MLX-LM.
    """
    import os
    import shutil
    import tempfile

    config = read_sft_config(config_path)
    source = Path(config["data"])
    manifest = verify_dataset(source)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"MLX view already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".mlx-sft-view-", dir=output.parent))
    try:
        for role in ("train", "valid"):
            count = 0
            with (source / f"{role}.jsonl").open() as raw, (temporary / f"{role}.jsonl").open("x") as target:
                for line in raw:
                    record = json.loads(line)
                    messages = record["messages"]
                    if [item["role"] for item in messages] != ["system", "user", "assistant"]:
                        raise ValueError("unexpected SFT conversation schema")
                    completion = messages[-1]["content"]
                    tokens, offset = encode_completion(tokenizer, messages[:-1], completion,
                        max_seq_length=config["max_seq_length"],
                        chat_template_kwargs=config["chat_template_kwargs"])
                    target.write(json.dumps({"tokens": tokens, "offset": offset}) + "\n")
                    count += 1
            if count != manifest["counts"][role] or count < config["batch_size"]:
                raise ValueError(f"{role} sample count mismatch or incomplete batch")
        effective = {**config, "data": str(output.resolve())}
        (temporary / "sft.json").write_text(json.dumps(effective, indent=2))
        (temporary / "source_manifest.json").write_text(json.dumps(manifest, indent=2))
        (temporary / "view_manifest.json").write_text(json.dumps({
            "config": config, "source_manifest": manifest,
            "files": {role: file_digest(temporary / f"{role}.jsonl") for role in ("train", "valid")},
        }, indent=2))
        os.rename(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output / "sft.json"


def verify_mlx_view(config_path, output):
    """Reuse preparation only if its source, recipe and rendered files match."""
    config = read_sft_config(config_path)
    manifest = verify_dataset(config["data"])
    output = Path(output)
    receipt = json.loads((output / "view_manifest.json").read_text())
    if receipt["config"] != config or receipt["source_manifest"] != manifest:
        raise ValueError("prepared view source or config changed")
    for role in ("train", "valid"):
        if file_digest(output / f"{role}.jsonl") != receipt["files"][role]:
            raise ValueError("prepared view changed")
    effective = output / "sft.json"
    if json.loads(effective.read_text()) != {**config, "data": str(output.resolve())}:
        raise ValueError("prepared SFT config changed")
    return effective


@dataclass(frozen=True)
class EncodedExample:
    tokens: tuple[int, ...]
    offset: int

    def __len__(self):
        return len(self.tokens)


class EncodedDataset:
    """MLX-LM dataset interface without applying a second chat template."""
    def __init__(self, path):
        self.rows = []
        with Path(path).open() as stream:
            for line in stream:
                row = json.loads(line)
                tokens, offset = tuple(row["tokens"]), row["offset"]
                if (type(offset) is not int or not 0 < offset < len(tokens)
                        or any(type(token) is not int or token < 0 for token in tokens)):
                    raise ValueError("invalid encoded supervision")
                self.rows.append(EncodedExample(tokens, offset))

    def __getitem__(self, index):
        return self.rows[index]

    def __len__(self):
        return len(self.rows)

    def process(self, row):
        return list(row.tokens), row.offset


def train_prepared(config_path, view):
    """Use native MLX-LM optimizer/LoRA training with already verified tokens."""
    from types import SimpleNamespace
    from mlx_lm import load
    from mlx_lm.lora import train_model
    effective = verify_mlx_view(config_path, view)
    config = json.loads(effective.read_text())
    if Path(config["adapter_path"]).exists():
        raise FileExistsError("adapter output exists; choose a new path")
    if config["resume_adapter_file"] is not None:
        from .model_config import verify_adapter_base
        parent = Path(config["resume_adapter_file"]).parent
        verify_adapter_base(config["model"], parent)
        metadata = json.loads((parent / "adapter_config.json").read_text())
        for key in ("lora_parameters", "num_layers", "chat_template_kwargs"):
            if metadata.get(key) != config[key]:
                raise ValueError(f"SFT warm-start contract differs at {key}")
    import numpy as np
    np.random.seed(config["seed"])
    model, _ = load(config["model"], tokenizer_config={"trust_remote_code": False})
    if not any("Quantized" in type(module).__name__ for _, module in model.named_modules()):
        raise ValueError("QLoRA requires a quantized base")
    train_model(SimpleNamespace(**config), model, EncodedDataset(Path(view) / "train.jsonl"),
                EncodedDataset(Path(view) / "valid.jsonl"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--view", required=True)
    parser.add_argument("--train", action="store_true", help="Explicitly launch native MLX-LM QLoRA")
    args = parser.parse_args(argv)
    config = read_sft_config(args.config)
    verify_dataset(config["data"])
    if Path(config["adapter_path"]).exists():
        raise FileExistsError("adapter output exists; choose a new path to preserve checkpoints")
    if Path(args.view).exists():
        effective = verify_mlx_view(args.config, args.view)
    else:
        # Optional runtime is imported only after the cheap integrity checks.
        from mlx_lm import load
        model, tokenizer = load(config["model"], tokenizer_config={"trust_remote_code": False})
        if not any("Quantized" in type(module).__name__ for _, module in model.named_modules()):
            raise ValueError("initial challenger requires a quantized base for QLoRA")
        effective = prepare_mlx_view(args.config, args.view, tokenizer=tokenizer)
        del model
        import mlx.core as mx
        mx.synchronize()
        mx.clear_cache()
    if args.train:
        train_prepared(args.config, args.view)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
