import os
import pickle
import subprocess
from typing import Any, Iterator, List, Tuple

import torch


def get_device_uuid(device_id: int) -> str:
    """Get the UUID for a GPU device. Works with both CUDA and ROCm/HIP."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        )
        uuids = result.stdout.strip().split("\n")
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible:
            visible = [int(d) for d in cuda_visible.split(",")]
            return uuids[visible[device_id]].strip()
        return uuids[device_id].strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    try:
        result = subprocess.run(
            ["rocm-smi", "--showuniqueid", "--csv"],
            capture_output=True, text=True, check=True,
        )
        lines = [l for l in result.stdout.strip().split("\n") if l and not l.startswith("device")]
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("ROCR_VISIBLE_DEVICES")
        if cuda_visible:
            visible = [int(d) for d in cuda_visible.split(",")]
            idx = visible[device_id]
        else:
            idx = device_id
        if idx < len(lines):
            parts = lines[idx].split(",")
            return f"GPU-{parts[-1].strip()}" if len(parts) > 1 else f"GPU-{idx}"
        return f"GPU-{idx}"
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    return f"GPU-{device_id}"


def serialize_tensors(named_tensors: List[Tuple[str, torch.Tensor]]) -> bytes:
    # Move tensors to CPU before serialization
    cpu_tensors = [
        (name, tensor.cpu() if tensor.is_cuda else tensor)
        for name, tensor in named_tensors
    ]
    return pickle.dumps(cpu_tensors)


def deserialize_tensors(data: bytes) -> List[Tuple[str, torch.Tensor]]:
    return pickle.loads(data)


def get_named_tensor_buckets(
    iterable: Iterator[Tuple[str, torch.Tensor]],
    bucket_bytes: int
) -> Iterator[List[Tuple[str, torch.Tensor]]]:
    if bucket_bytes <= 0:
        raise ValueError(f"bucket_bytes must be greater than 0, got {bucket_bytes}")

    current_bucket = []
    current_size = 0
    
    for name, tensor in iterable:
        tensor_size = tensor.element_size() * tensor.numel()
        if current_size + tensor_size > bucket_bytes:
            if current_bucket:
                yield current_bucket
            current_bucket = [(name, tensor)]
            current_size = tensor_size
        else:
            current_bucket.append((name, tensor))
            current_size += tensor_size

    if current_bucket:
        yield current_bucket
