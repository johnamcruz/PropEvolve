"""Audited, prompt-masked QLoRA through MLX-LM's native trainer.

Importing this module loads neither MLX nor a model. Preparing/launching are
explicit operations, never side effects of importing the C51 application.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
from .integrity import file_digest
from .model_config import validate_model_settings, model_defaults, template_options


def read_sft_config(path: str | Path) -> dict:
    payload = {**model_defaults(), **json.loads(Path(path).read_text())}
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
    """Make native completion datasets without target/prompt leakage/truncation.

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
                    prompt = tokenizer.apply_chat_template(
                        messages[:-1], tokenize=False, add_generation_prompt=True,
                        **template_options(config["chat_template_kwargs"]),
                    )
                    completion = messages[-1]["content"]
                    token_count = len(tokenizer.encode(prompt + completion + tokenizer.eos_token))
                    if token_count > config["max_seq_length"]:
                        raise ValueError("SFT example exceeds token budget; refusing silent truncation")
                    target.write(json.dumps({"prompt": prompt, "completion": completion}) + "\n")
                    count += 1
            if count != manifest["counts"][role] or count < config["batch_size"]:
                raise ValueError(f"{role} sample count mismatch or incomplete batch")
        effective = {**config, "data": str(output.resolve())}
        (temporary / "sft.json").write_text(json.dumps(effective, indent=2))
        (temporary / "source_manifest.json").write_text(json.dumps(manifest, indent=2))
        os.rename(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output / "sft.json"


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
    # Optional runtime is imported only after the cheap integrity checks.
    from mlx_lm import load
    model, tokenizer = load(config["model"])
    if not any("Quantized" in type(module).__name__ for _, module in model.named_modules()):
        raise ValueError("initial challenger requires a quantized base for QLoRA")
    effective = prepare_mlx_view(args.config, args.view, tokenizer=tokenizer)
    del model
    import mlx.core as mx
    mx.synchronize()
    mx.clear_cache()
    if args.train:
        subprocess.run([sys.executable, "-m", "mlx_lm.lora", "--config", str(effective)], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
