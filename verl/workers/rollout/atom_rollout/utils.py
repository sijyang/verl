import os
from typing import Callable

import torch

# Index of the device_id field in the args tuple produced by
# torch.multiprocessing.reductions.reduce_tensor(). See
# torch/multiprocessing/reductions.py::rebuild_cuda_tensor().
_IPC_HANDLE_DEVICE_ID_INDEX = 6


def rebuild_ipc_handle(handle: tuple[Callable, tuple], device_id: int | None = None) -> torch.Tensor:
    """Rebuild a CUDA tensor from its IPC handle.

    When two processes have different CUDA_VISIBLE_DEVICES, the device_id
    in the handle may be wrong. This function fixes it by overriding
    the device_id field in the args tuple.

    Args:
        handle: A tuple of (rebuild_function, args) from reduce_tensor().
        device_id: Override the device_id in the handle. If None, use the
            original device_id from the handle.

    Returns:
        The reconstructed CUDA tensor sharing the same GPU memory.
    """
    func, args = handle
    list_args = list(args)
    if device_id is not None:
        list_args[_IPC_HANDLE_DEVICE_ID_INDEX] = device_id
    buffer = func(*list_args)
    return buffer


def get_device_uuid(device_id: int) -> str:
    """Get a unique identifier for the CUDA device.

    Used to create unique ZMQ IPC socket paths per GPU to avoid conflicts
    when multiple processes share the same node.

    Args:
        device_id: The local CUDA device index.

    Returns:
        A unique string identifier for the device.
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        uuid = pynvml.nvmlDeviceGetUUID(handle)
        return uuid
    except Exception:
        # Fallback: use CUDA_VISIBLE_DEVICES mapping or raw device_id
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if visible:
            devices = visible.split(",")
            if device_id < len(devices):
                return f"GPU-{devices[device_id]}"
        return f"GPU-{device_id}"
