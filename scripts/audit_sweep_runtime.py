"""Bounded production Optuna training prefix; no study or source-run writes.

Keep the supplied trial's full curriculum budget. Stop only after the requested
completed episode diagnostic, so warmup, replay, and learning remain unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time


class PrefixComplete(BaseException):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--audit-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=12)
    parser.add_argument('--maximum-footprint-gib', type=float, default=3)
    parser.add_argument('--runtime-config', type=Path,
                        help='JSON runtime-only overrides for the matched audit')
    parser.add_argument('--trim-torch-cache', action='store_true',
                        help='diagnostic ablation after each complete replay update')
    args = parser.parse_args()
    if args.episodes < 1 or args.maximum_footprint_gib <= 0:
        parser.error('positive episode and memory bounds required')
    if args.output.exists() or args.audit_config.exists():
        parser.error('audit config and output must be new, preserving prior evidence')
    payload = json.loads(args.config.read_text())
    if args.runtime_config:
        payload['runtime'].update(json.loads(args.runtime_config.read_text()))
    payload['output'] = str(args.output.resolve())
    args.audit_config.parent.mkdir(parents=True, exist_ok=True)
    args.audit_config.write_text(json.dumps(payload, indent=2) + '\n')
    args.output.mkdir(parents=True)
    from propevolve import training
    from propevolve.optuna_trial import run_optuna_trial
    from propevolve.mlx_backend import mlx_memory_metrics, shutdown_mlx_backend
    from propevolve import mlx_backend
    import torch
    worker_call = mlx_backend._MlxExecutionWorker.call
    signatures = set()

    def measured_call(worker, operation, *values):
        before = time.monotonic()
        tensors = values[0] if operation == 'backward' else values
        signature = (operation, tuple(tuple(value.shape) for value in tensors)) if operation != 'memory' else ('memory',)
        first = signature not in signatures
        signatures.add(signature)
        result = worker_call(worker, operation, *values)
        elapsed = time.monotonic() - before
        if elapsed >= 1:
            print('[runtime-operation] ' + json.dumps(dict(
                signature=signature, first=first, seconds=elapsed,
                unique_signatures=len(signatures))), flush=True)
        return result

    mlx_backend._MlxExecutionWorker.call = measured_call
    original = training.train_agent
    original_replay = training.train_replay_with_mastery
    update_count = 0

    def measured_replay(*values, **kwargs):
        nonlocal update_count
        before = time.monotonic()
        result = original_replay(*values, **kwargs)
        if args.trim_torch_cache:
            torch.mps.empty_cache()
        update_count += 1
        print('[runtime-update] ' + json.dumps(dict(
            update=update_count, seconds=time.monotonic()-before,
            signatures=len(signatures))), flush=True)
        return result

    training.train_replay_with_mastery = measured_replay
    started = time.monotonic()
    prior = started
    completed = 0

    def measured_train(*values, **kwargs):
        callback = kwargs.get('episode_diagnostic_callback')

        def observe(diagnostic):
            nonlocal prior, completed
            if callback is not None:
                callback(diagnostic)
            completed += 1
            now = time.monotonic()
            vm = subprocess.run(['vmmap', '-summary', str(os.getpid())],
                                capture_output=True, text=True, check=True)
            match = re.search(r'^Physical footprint:\s+([\d.]+)([KMG])',
                              vm.stdout, re.MULTILINE)
            if match is None:
                raise RuntimeError('physical footprint unavailable')
            footprint = float(match[1]) * {'K':1024, 'M':1024**2, 'G':1024**3}[match[2]]
            record = dict(episode=completed, seconds=now-started,
                          episode_seconds=now-prior, footprint_bytes=footprint,
                          pid=os.getpid(), updates=diagnostic['updates'],
                          torch_live_bytes=torch.mps.current_allocated_memory(),
                          torch_driver_bytes=torch.mps.driver_allocated_memory(),
                          **mlx_memory_metrics())
            if completed >= int(payload['training']['warmup_episodes']):
                assert diagnostic['updates'] == int(payload['training']['updates_per_episode']), 'optimizer work was skipped'
            prior = now
            with (args.output / 'runtime-audit.jsonl').open('a') as stream:
                stream.write(json.dumps(record) + '\n')
            print('[runtime-audit] ' + json.dumps(record), flush=True)
            if footprint > args.maximum_footprint_gib * 1024**3:
                raise MemoryError('full training prefix exceeded physical memory bound')
            if completed >= args.episodes:
                raise PrefixComplete()

        kwargs['episode_diagnostic_callback'] = observe
        return original(*values, **kwargs)

    training.train_agent = measured_train
    try:
        try:
            run_optuna_trial(args.audit_config, result_path=args.output / 'trial-result.json')
        except PrefixComplete:
            pass
        if completed < args.episodes:
            raise AssertionError('training ended before the requested prefix')
        (args.output / 'prefix-complete.json').write_text(json.dumps(dict(
            completed_episodes=completed, source_config_sha256=hashlib.sha256(
                args.config.read_bytes()).hexdigest(), seconds=time.monotonic()-started)) + '\n')
    finally:
        training.train_agent = original
        training.train_replay_with_mastery = original_replay
        mlx_backend._MlxExecutionWorker.call = worker_call
        shutdown_mlx_backend()


if __name__ == '__main__':
    main()
