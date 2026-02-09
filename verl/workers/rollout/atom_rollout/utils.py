import pickle
from typing import Any, Iterator, List, Tuple

import torch


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
