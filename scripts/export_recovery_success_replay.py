#!/usr/bin/env python3
"""Export a compact, authenticated V22 recovery-pass replay artifact."""

from __future__ import annotations

import argparse
import codecs
import hashlib
import mmap
import os
import pickle
from pathlib import Path
import struct
from typing import Mapping
import zipfile

import numpy as np
import torch


RECOVERY_SCHEMA = "propevolve_recovery_success_replay_v1"
HEALTHY_SCHEMA = "propevolve_healthy_pass_replay_v1"


class _LazyBytes:
    def __init__(self, offset: int, length: int) -> None:
        self.offset = offset
        self.length = length


class _LazyUnicode(_LazyBytes):
    pass


class _LazyEncodedBytes:
    def __init__(self, value: _LazyUnicode, encoding: str) -> None:
        self.value = value
        self.encoding = encoding


class _LazyArray:
    def __init__(self) -> None:
        self.state = None

    def __setstate__(self, state) -> None:
        self.state = state

    def materialize(self, source: mmap.mmap) -> np.ndarray:
        if not isinstance(self.state, tuple) or len(self.state) != 5:
            raise ValueError("lazy NumPy replay state is invalid")
        _, shape, dtype, fortran, raw = self.state
        dtype = np.dtype(dtype)
        count = int(np.prod(shape, dtype=np.int64))
        if isinstance(raw, _LazyEncodedBytes):
            encoded = source[
                raw.value.offset:raw.value.offset + raw.value.length
            ]
            decoded = encoded.decode("utf-8", "surrogatepass")
            buffer = decoded.encode(raw.encoding)
            array = np.frombuffer(buffer, dtype=dtype, count=count)
        elif isinstance(raw, _LazyBytes):
            expected = count * dtype.itemsize
            if raw.length != expected:
                raise ValueError("lazy NumPy replay buffer length drifted")
            array = np.frombuffer(
                source,
                dtype=dtype,
                count=count,
                offset=raw.offset,
            )
        else:
            array = np.frombuffer(raw, dtype=dtype, count=count)
        order = "F" if fortran else "C"
        return array.reshape(tuple(shape), order=order).copy()


class _DiscardedTensor:
    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs

    def __setstate__(self, state) -> None:
        del state


def _lazy_numpy_reconstruct(*args, **kwargs) -> _LazyArray:
    del args, kwargs
    return _LazyArray()


def _discard_tensor(*args, **kwargs) -> _DiscardedTensor:
    del args, kwargs
    return _DiscardedTensor()


def _lazy_encode(value, encoding="utf-8", errors="strict"):
    if isinstance(value, _LazyUnicode):
        if errors != "strict":
            raise ValueError("lazy replay encoding errors contract drifted")
        return _LazyEncodedBytes(value, encoding)
    return codecs.encode(value, encoding, errors)


class _BoundedCheckpointUnpickler(pickle._Unpickler):
    dispatch = pickle._Unpickler.dispatch.copy()

    def _lazy_bytes(self, length: int) -> None:
        offset = self.read(0) and 0
        del offset
        position = self._source_stream.tell()
        self._source_stream.seek(length, os.SEEK_CUR)
        self.append(_LazyBytes(position, length))

    def load_binbytes(self) -> None:
        self._lazy_bytes(struct.unpack("<I", self.read(4))[0])

    def load_binbytes8(self) -> None:
        self._lazy_bytes(struct.unpack("<Q", self.read(8))[0])

    def load_short_binbytes(self) -> None:
        length = self.read(1)[0]
        self.append(self.read(length))

    def load_bytearray8(self) -> None:
        self._lazy_bytes(struct.unpack("<Q", self.read(8))[0])

    def load_binunicode(self) -> None:
        length = struct.unpack("<I", self.read(4))[0]
        if length <= 4096:
            self.append(self.read(length).decode("utf-8", "surrogatepass"))
            return
        position = self._source_stream.tell()
        self._source_stream.seek(length, os.SEEK_CUR)
        self.append(_LazyUnicode(position, length))

    def load_binunicode8(self) -> None:
        length = struct.unpack("<Q", self.read(8))[0]
        if length <= 4096:
            self.append(self.read(length).decode("utf-8", "surrogatepass"))
            return
        position = self._source_stream.tell()
        self._source_stream.seek(length, os.SEEK_CUR)
        self.append(_LazyUnicode(position, length))

    def find_class(self, module: str, name: str):
        if (
            module in {"numpy.core.multiarray", "numpy._core.multiarray"}
            and name == "_reconstruct"
        ):
            return _lazy_numpy_reconstruct
        if module == "torch._utils" and name.startswith("_rebuild"):
            return _discard_tensor
        if module == "_codecs" and name == "encode":
            return _lazy_encode
        if module == "torch.nn.parameter" and name == "Parameter":
            return _DiscardedTensor
        return super().find_class(module, name)

    def persistent_load(self, pid):
        del pid
        return _DiscardedTensor()


_BoundedCheckpointUnpickler.dispatch[pickle.BINBYTES[0]] = (
    _BoundedCheckpointUnpickler.load_binbytes
)
_BoundedCheckpointUnpickler.dispatch[pickle.BINBYTES8[0]] = (
    _BoundedCheckpointUnpickler.load_binbytes8
)
_BoundedCheckpointUnpickler.dispatch[pickle.SHORT_BINBYTES[0]] = (
    _BoundedCheckpointUnpickler.load_short_binbytes
)
_BoundedCheckpointUnpickler.dispatch[pickle.BYTEARRAY8[0]] = (
    _BoundedCheckpointUnpickler.load_bytearray8
)
_BoundedCheckpointUnpickler.dispatch[pickle.BINUNICODE[0]] = (
    _BoundedCheckpointUnpickler.load_binunicode
)
_BoundedCheckpointUnpickler.dispatch[pickle.BINUNICODE8[0]] = (
    _BoundedCheckpointUnpickler.load_binunicode8
)


def _pickle_data_offset(checkpoint: Path) -> tuple[int, str]:
    with zipfile.ZipFile(checkpoint) as archive:
        names = [name for name in archive.namelist() if name.endswith("/data.pkl")]
        if len(names) != 1:
            raise ValueError("checkpoint pickle member is ambiguous")
        info = archive.getinfo(names[0])
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError("checkpoint pickle member must be uncompressed")
    with checkpoint.open("rb") as stream:
        stream.seek(info.header_offset)
        header = stream.read(30)
        if len(header) != 30 or header[:4] != b"PK\x03\x04":
            raise ValueError("checkpoint local header is invalid")
        name_length, extra_length = struct.unpack("<HH", header[26:30])
    return info.header_offset + 30 + name_length + extra_length, names[0]


def _load_checkpoint_without_array_buffers(checkpoint: Path) -> Mapping[str, object]:
    data_offset, _ = _pickle_data_offset(checkpoint)
    with checkpoint.open("rb") as stream:
        stream.seek(data_offset)
        unpickler = _BoundedCheckpointUnpickler(stream)
        unpickler._source_stream = stream
        payload = unpickler.load()
    if not isinstance(payload, Mapping):
        raise ValueError("source checkpoint payload is invalid")
    return payload


def _materialize(value, source: mmap.mmap):
    if isinstance(value, _LazyArray):
        return value.materialize(source)
    if isinstance(value, Mapping):
        return {key: _materialize(item, source) for key, item in value.items()}
    if isinstance(value, list):
        return [_materialize(item, source) for item in value]
    if isinstance(value, tuple):
        return tuple(_materialize(item, source) for item in value)
    if isinstance(value, _LazyBytes):
        return bytes(source[value.offset:value.offset + value.length])
    if isinstance(value, _LazyEncodedBytes):
        encoded = source[
            value.value.offset:value.value.offset + value.value.length
        ]
        return encoded.decode("utf-8", "surrogatepass").encode(
            value.encoding
        )
    if isinstance(value, _DiscardedTensor):
        raise ValueError("target replay episode unexpectedly contains a tensor")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_successful_recovery_pass(episode: Mapping[str, object]) -> bool:
    if episode.get("outcome") != "pass":
        return False
    recovery_active = np.asarray(
        episode.get("recovery_active", ()),
        dtype=np.bool_,
    )
    return bool(
        recovery_active.size >= 2
        and np.any(recovery_active[:-1] & ~recovery_active[1:])
    )


def _is_healthy_pass(episode: Mapping[str, object]) -> bool:
    if episode.get("outcome") != "pass":
        return False
    recovery_active = np.asarray(
        episode.get("recovery_active", ()),
        dtype=np.bool_,
    )
    return bool(recovery_active.size and np.any(~recovery_active))


def _with_recovery_state(
    episode: Mapping[str, object],
) -> Mapping[str, object]:
    if "recovery_active" in episode:
        return episode
    observations = np.asarray(episode.get("observations"), dtype=np.float32)
    actions = np.asarray(episode.get("actions"), dtype=np.int8)
    account_and_management_width = 18
    if (
        observations.ndim != 2
        or observations.shape[0] != actions.size + 1
        or observations.shape[1] <= account_and_management_width
    ):
        raise ValueError("legacy recovery observation contract is invalid")
    realized_pnl_fraction = observations[
        :-1,
        -account_and_management_width,
    ]
    if not np.isfinite(realized_pnl_fraction).all():
        raise ValueError("legacy recovery PnL channel is non-finite")
    return {
        **episode,
        "recovery_active": realized_pnl_fraction < 0.0,
    }


def export(
    checkpoints: tuple[Path, ...],
    output: Path,
    *,
    episode_prefixes: tuple[str, ...] = (),
    kind: str = "recovery",
) -> dict[str, object]:
    if kind not in {"recovery", "healthy"}:
        raise ValueError("pass replay export kind is invalid")
    predicate = (
        _is_successful_recovery_pass
        if kind == "recovery"
        else _is_healthy_pass
    )
    schema = RECOVERY_SCHEMA if kind == "recovery" else HEALTHY_SCHEMA
    contract = None
    schema_version = None
    random_state = None
    episodes: list[Mapping[str, object]] = []
    episode_ids: set[str] = set()
    sources = []
    for checkpoint in checkpoints:
        checkpoint = checkpoint.resolve(strict=True)
        payload = _load_checkpoint_without_array_buffers(checkpoint)
        if payload.get("schema") != "propevolve_recurrent_c51_v1":
            raise ValueError("source checkpoint schema is invalid")
        manifest = payload.get("manifest")
        if not isinstance(manifest, Mapping):
            raise ValueError("source checkpoint manifest is invalid")
        replay_state = manifest.get("replay_state")
        if not isinstance(replay_state, Mapping):
            raise ValueError("source checkpoint lacks replay state")
        if contract is None:
            contract = replay_state.get("contract")
            schema_version = replay_state.get("schema_version")
            random_state = replay_state.get("random_state")
        elif replay_state.get("contract") != contract:
            raise ValueError("source replay contracts do not match")
        source_schema_version = replay_state.get("schema_version")
        if source_schema_version not in {10, 11}:
            raise ValueError("source replay schema cannot preserve recovery state")
        schema_version = max(int(schema_version), int(source_schema_version))
        source_episodes = replay_state.get("episodes")
        if not isinstance(source_episodes, list):
            raise ValueError("source replay episodes are invalid")
        with checkpoint.open("rb") as stream:
            source = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
            try:
                for episode in source_episodes:
                    if not isinstance(episode, Mapping):
                        raise ValueError("source replay episode is invalid")
                    episode_id = str(episode.get("episode_id", ""))
                    if episode_prefixes and not any(
                        episode_id.startswith(prefix)
                        for prefix in episode_prefixes
                    ):
                        continue
                    materialized = _with_recovery_state(
                        _materialize(episode, source)
                    )
                    if not predicate(materialized):
                        continue
                    if not episode_id or episode_id in episode_ids:
                        raise ValueError(
                            "pass replay episode identity collided"
                        )
                    episodes.append(materialized)
                    episode_ids.add(episode_id)
            finally:
                source.close()
        resume_identity = manifest.get("resume_identity")
        if not isinstance(resume_identity, str) or not resume_identity:
            raise ValueError("source checkpoint resume identity is invalid")
        sources.append({
            "causal_identity_sha256": resume_identity,
            "resume_identity": resume_identity,
        })
        del payload
    if not episodes or contract is None or random_state is None:
        raise ValueError("source checkpoints contain no retained pass")
    replay_state = {
        "schema_version": schema_version,
        "contract": contract,
        "random_state": random_state,
        "sample_calls": 0,
        "episodes": episodes,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save({
        "schema": schema,
        "source_checkpoints": sources,
        "replay_state": replay_state,
    }, temporary)
    os.replace(temporary, output)
    return {
        "artifact": str(output),
        "sha256": _sha256(output),
        "source_checkpoints": len(sources),
        "kind": kind,
        "retained_passes": len(episodes),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episode-prefix", action="append", default=[])
    parser.add_argument(
        "--kind",
        choices=("recovery", "healthy"),
        default="recovery",
    )
    args = parser.parse_args()
    receipt = export(
        tuple(Path(value) for value in args.checkpoint),
        Path(args.output),
        episode_prefixes=tuple(args.episode_prefix),
        kind=args.kind,
    )
    print(receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
