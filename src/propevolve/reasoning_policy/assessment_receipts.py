"""Content-addressed frozen assessment reuse across campaign output folders."""

import fcntl
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import shutil

from .integrity import file_digest
from .workflow import atomic_json


def assessment_identity(config_path, view, role, *, root=None):
    """Bind scores AND metrics to frozen inputs, excluding training-only knobs.

    Dataset/view integrity must additionally be checked by the assessment caller;
    manifests bind external embedding shards verified by verify_mlx_view.
    """
    from .mlx_sft import read_sft_config
    if role not in {"train", "valid"}:
        raise ValueError("assessment role must be train or valid")
    config = read_sft_config(config_path, root=root)
    model = Path(config["model"])
    if not model.is_dir():
        from huggingface_hub import snapshot_download
        model = Path(snapshot_download(config["model"], local_files_only=True))
    model_files = {str(path.relative_to(model)): file_digest(path)
                   for path in sorted(model.rglob("*")) if path.is_file()
                   and ".cache" not in path.relative_to(model).parts}
    if not model_files:
        raise ValueError("assessment base model has no local artifacts")
    adapter_files = None
    if config["adapter_path"] is not None:
        adapter = Path(config["adapter_path"])
        names = ["adapters.safetensors", "adapter_config.json"]
        if config["input_mode"] == "embeddings":
            names.append("projector.safetensors")
        adapter_files = {name: file_digest(adapter / name) for name in names}
    # Validation reports include loss, so its supervision weights, seed and
    # numerical batch settings belong in the identity even without weight updates.
    settings = {name: config.get(name) for name in (
        "max_seq_length", "input_mode", "projector", "action_verbalizers",
        "chat_template_kwargs", "decision_objective", "action_supervision",
        "seed", "validation_batch_size")}
    view, data = Path(view), Path(config["data"])
    files = {"view_manifest": file_digest(view / "view_manifest.json"),
             "prepared_rows": file_digest(view / f"{role}.jsonl"),
             "source_rows": file_digest(data / f"{role}.jsonl")}
    source_manifest = data / "manifest.json"
    if source_manifest.exists():
        files["source_manifest"] = file_digest(source_manifest)
    modules = ("learning_audit.py", "supervised_trainer.py", "supervision.py",
               "policy.py", "projector.py", "tokenization.py", "mlx_sft.py",
               "decision_schema.py", "assessment_receipts.py")
    code = {name: file_digest(Path(__file__).with_name(name)) for name in modules}
    packages = {}
    for name in ("mlx", "mlx-lm", "numpy", "transformers", "tokenizers"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    payload = {"schema": "frozen_assessment_identity_v1", "role": role,
               "model": model_files, "adapter": adapter_files, "settings": settings,
               "data": files, "code": code, "packages": packages}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _verify(receipt):
    descriptor = json.loads(receipt.read_text())
    output = Path(descriptor["path"])
    for name in ("scores.jsonl", "summary.json"):
        if file_digest(output / name) != descriptor[name]:
            raise ValueError("assessment receipt artifact changed; refusing silent recomputation")
    summary = json.loads((output / "summary.json").read_text())
    count = sum(bool(line.strip()) for line in (output / "scores.jsonl").open())
    if summary.get("weights_updated") is not False or count < 1 or summary.get("rows") != count:
        raise ValueError("assessment receipt is incomplete")
    return descriptor, summary


def cached_assessment(policy, view, role, output, log, *, cache_root, run, root=None):
    """Execute assessment only on an identity miss; never overwrite evidence.

    run is the external assessment-process boundary. Receipts reference completed
    output files rather than duplicating their storage in the registry.
    """
    policy, view, output, log = map(Path, (policy, view, output, log))
    if cache_root is None:
        return run(policy, view, role, output, log)
    identity = assessment_identity(policy, view, role, root=root)
    cache = Path(cache_root)
    cache.mkdir(parents=True, exist_ok=True)
    receipt = cache / f"{identity}.json"
    with (cache / f"{identity}.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("matching assessment is already in progress") from error
        if receipt.exists():
            descriptor, summary = _verify(receipt)
            if descriptor["identity"] != identity or summary.get("role") != role:
                raise ValueError("assessment receipt identity differs")
            source = Path(descriptor["path"])
            if output.resolve() != source.resolve():
                output.mkdir(parents=True, exist_ok=False)
                shutil.copyfile(source / "scores.jsonl", output / "scores.jsonl")
                summary["config_sha256"] = file_digest(policy)
                atomic_json(output / "summary.json", summary)
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("a") as stream:
                stream.write(f"[assessment] status=reused role={role} receipt={receipt}\n")
            return
        run(policy, view, role, output, log)
        if assessment_identity(policy, view, role, root=root) != identity:
            raise ValueError("assessment inputs changed during inference")
        descriptor = {"identity": identity, "path": str(output.resolve()),
                      **{name: file_digest(output / name)
                         for name in ("scores.jsonl", "summary.json")}}
        # Publish only a complete, frozen assessment. Never register partial output.
        summary = json.loads((output / "summary.json").read_text())
        count = sum(bool(line.strip()) for line in (output / "scores.jsonl").open())
        if (summary.get("weights_updated") is not False or summary.get("role") != role
                or count < 1 or summary.get("rows") != count
                or summary.get("config_sha256") != file_digest(policy)
                or summary.get("view_manifest_sha256") != file_digest(view / "view_manifest.json")):
            raise ValueError("assessment output is incomplete or unauthenticated")
        atomic_json(receipt, descriptor)
