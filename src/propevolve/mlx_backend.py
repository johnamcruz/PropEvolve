"""Optional MLX recurrent tensor adapter for the shared C51 learner.

PropEvolve keeps one production pipeline. Replay, targets, C51 projection,
auxiliary losses, PCGrad, AdamW boundary repair, diagnostics, checkpoints, and
teacher-free validation remain in :class:`RecurrentC51Agent`. This module only
executes the encoder/GRU forward and backward math with MLX over shared Metal
storage.
"""

from __future__ import annotations

import atexit
import gc
from queue import Queue
import threading
from typing import Any


def _mlx_core():
    try:
        import mlx.core as mx
    except (ImportError, RuntimeError) as error:
        raise RuntimeError(
            "the optional MLX learner backend requires PropEvolve[mlx] "
            "and an available Apple Metal device"
        ) from error
    return mx


def _mlx_recurrent_primitive(
    observations,
    hidden,
    norm_weight,
    norm_bias,
    input_weight,
    input_bias,
    recurrent_input_weight,
    recurrent_hidden_weight,
    recurrent_input_bias,
    recurrent_hidden_bias,
):
    """Exact LayerNorm/Linear/SiLU/PyTorch-GRU equations in MLX."""
    mx = _mlx_core()
    mean = observations.mean(axis=-1, keepdims=True)
    variance = ((observations - mean) ** 2).mean(axis=-1, keepdims=True)
    encoded = (
        (observations - mean) * mx.rsqrt(variance + 1e-5) * norm_weight
        + norm_bias
    )
    encoded = encoded @ input_weight.T + input_bias
    encoded = encoded * mx.sigmoid(encoded)
    outputs = []
    current = hidden
    for index in range(encoded.shape[-2]):
        input_projection = (
            encoded[..., index, :] @ recurrent_input_weight.T
            + recurrent_input_bias
        )
        hidden_projection = (
            current @ recurrent_hidden_weight.T + recurrent_hidden_bias
        )
        input_reset, input_update, input_new = mx.split(
            input_projection, 3, axis=-1
        )
        hidden_reset, hidden_update, hidden_new = mx.split(
            hidden_projection, 3, axis=-1
        )
        reset = mx.sigmoid(input_reset + hidden_reset)
        update = mx.sigmoid(input_update + hidden_update)
        new = mx.tanh(input_new + reset * hidden_new)
        current = (1.0 - update) * new + update * current
        outputs.append(current)
    return [mx.stack(outputs, axis=-2), current]


def _torch_to_mlx(values):
    mx = _mlx_core()
    return [mx.asarray(value.detach()) for value in values]


def _mlx_to_torch(values):
    import torch

    return [torch.as_tensor(value) for value in values]


_TORCH_RECURRENT_OPERATION = None
_MLX_RECURRENT_FUNCTION = None
_MLX_RECURRENT_VJP = None
_MLX_WORKER = None
_MLX_WORKER_LOCK = threading.Lock()
_MLX_ATEXIT_REGISTERED = False


def _mlx_recurrent_function():
    """Compile the fixed-shape recurrent primitive once per MLX process."""
    global _MLX_RECURRENT_FUNCTION
    if _MLX_RECURRENT_FUNCTION is None:
        mx = _mlx_core()
        _MLX_RECURRENT_FUNCTION = mx.compile(_mlx_recurrent_primitive)
    return _MLX_RECURRENT_FUNCTION


def _mlx_recurrent_vjp():
    """Compile the recurrent reverse pass instead of rebuilding it per loss."""
    global _MLX_RECURRENT_VJP
    if _MLX_RECURRENT_VJP is None:
        mx = _mlx_core()

        def recurrent_vjp(*values):
            inputs = values[:-2]
            cotangents = values[-2:]
            _, gradients = mx.vjp(
                _mlx_recurrent_function(),
                inputs,
                cotangents,
            )
            return gradients

        _MLX_RECURRENT_VJP = mx.compile(recurrent_vjp)
    return _MLX_RECURRENT_VJP


class _MlxExecutionWorker:
    """Own MLX compiled functions and destroy their streams before exit."""

    def __init__(self) -> None:
        self._requests: Queue[tuple[str, tuple[Any, ...], Queue] | None] = Queue()
        self._thread = threading.Thread(
            target=self._run,
            name="propevolve-mlx",
            # Python waits for non-daemon threads before running atexit. This
            # worker must reach our atexit handler first so it can clear MLX's
            # thread-local streams and then be joined deterministically.
            daemon=True,
        )
        self._thread.start()

    def call(self, operation: str, *values):
        if not self._thread.is_alive():
            raise RuntimeError("the MLX execution worker is unavailable")
        response: Queue = Queue(maxsize=1)
        self._requests.put((operation, values, response))
        succeeded, payload = response.get()
        if not succeeded:
            raise payload
        return payload

    def close(self) -> None:
        if self._thread.is_alive():
            self._requests.put(None)
            self._thread.join()

    def _run(self) -> None:
        global _MLX_RECURRENT_FUNCTION, _MLX_RECURRENT_VJP
        mx = _mlx_core()
        try:
            while True:
                request = self._requests.get()
                if request is None:
                    return
                operation, values, response = request
                try:
                    if operation == "forward":
                        mlx_values = _torch_to_mlx(values)
                        outputs = _mlx_recurrent_function()(*mlx_values)
                        mx.eval(outputs)
                        result = tuple(_mlx_to_torch(outputs))
                    elif operation == "backward":
                        torch_values, torch_cotangents = values
                        mlx_values = _torch_to_mlx(torch_values)
                        outputs = _mlx_recurrent_function()(*mlx_values)
                        mlx_cotangents = [
                            (
                                mx.zeros_like(output)
                                if cotangent is None
                                else mx.asarray(cotangent.detach())
                            )
                            for output, cotangent in zip(
                                outputs,
                                torch_cotangents,
                                strict=True,
                            )
                        ]
                        gradients = _mlx_recurrent_vjp()(
                            *mlx_values,
                            *mlx_cotangents,
                        )
                        mx.eval(gradients)
                        result = tuple(_mlx_to_torch(gradients))
                    elif operation == "memory":
                        mx.eval()
                        result = {
                            "active_memory_bytes": int(mx.get_active_memory()),
                            "cache_memory_bytes": int(mx.get_cache_memory()),
                            "peak_memory_bytes": int(mx.get_peak_memory()),
                        }
                    else:
                        raise ValueError("unknown MLX worker operation")
                except BaseException as error:
                    response.put((False, error))
                else:
                    response.put((True, result))
        finally:
            # Compiled functions are backed by a thread-local cache. They must
            # be released and their streams cleared on the owning thread while
            # the Python and Metal runtimes are still alive.
            _MLX_RECURRENT_VJP = None
            _MLX_RECURRENT_FUNCTION = None
            gc.collect()
            mx.clear_streams()


def _mlx_worker() -> _MlxExecutionWorker:
    global _MLX_WORKER, _MLX_ATEXIT_REGISTERED
    with _MLX_WORKER_LOCK:
        if _MLX_WORKER is None:
            _MLX_WORKER = _MlxExecutionWorker()
        if not _MLX_ATEXIT_REGISTERED:
            atexit.register(shutdown_mlx_backend)
            _MLX_ATEXIT_REGISTERED = True
        return _MLX_WORKER


def shutdown_mlx_backend() -> None:
    """Cleanly stop MLX before Python and PyTorch tear down Metal."""
    global _MLX_WORKER
    with _MLX_WORKER_LOCK:
        worker = _MLX_WORKER
        _MLX_WORKER = None
    if worker is not None:
        worker.close()


def _torch_recurrent_operation():
    """Create the custom autograd operation only when MLX is selected."""
    global _TORCH_RECURRENT_OPERATION
    if _TORCH_RECURRENT_OPERATION is not None:
        return _TORCH_RECURRENT_OPERATION

    import torch

    class Operation(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *torch_values):
            torch.mps.synchronize()
            ctx.save_for_backward(*torch_values)
            return _mlx_worker().call("forward", *torch_values)

        @staticmethod
        def backward(ctx, *torch_cotangents):
            torch.mps.synchronize()
            return _mlx_worker().call(
                "backward",
                ctx.saved_tensors,
                torch_cotangents,
            )

    _TORCH_RECURRENT_OPERATION = Operation
    return Operation


def mlx_torch_recurrent_features(network, observations, hidden=None):
    """Use MLX behind the existing Torch recurrent learner interface."""
    import torch

    if observations.device.type != "mps":
        raise ValueError("the MLX learner backend requires the MPS device")
    if observations.dtype != torch.float32:
        raise ValueError("the MLX learner backend currently requires fp32")
    if hidden is None:
        hidden_value = torch.zeros(
            (observations.shape[0], network.hidden_dim),
            dtype=observations.dtype,
            device=observations.device,
        )
    else:
        if hidden.shape != (1, observations.shape[0], network.hidden_dim):
            raise ValueError("recurrent hidden state shape is invalid")
        hidden_value = hidden[0]
    operation = _torch_recurrent_operation()
    recurrent, final_hidden = operation.apply(
        observations,
        hidden_value,
        network.input[0].weight,
        network.input[0].bias,
        network.input[1].weight,
        network.input[1].bias,
        network.recurrent.weight_ih_l0,
        network.recurrent.weight_hh_l0,
        network.recurrent.bias_ih_l0,
        network.recurrent.bias_hh_l0,
    )
    return recurrent, final_hidden.unsqueeze(0)


def mlx_memory_metrics() -> dict[str, int]:
    """Return synchronized MLX allocator evidence for runtime benchmarks."""
    return _mlx_worker().call("memory")
