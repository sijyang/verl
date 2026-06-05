ATOM Backend
============

Last updated: 06/05/2026.

Introduction
------------
`ATOM <https://github.com/AMD/atom>`_ is a high-performance LLM inference engine optimized for AMD GPUs (ROCm/MI300X).

The verl integration of ATOM uses a decoupled architecture where ``ATOMHttpServer`` (a Ray actor) owns the engine, and ``ServerAdapter`` acts as a thin client in the trainer worker that delegates operations via ``ray.get_actor()``.

Architecture
------------

The ATOM rollout integration consists of three main components:

- **ServerAdapter** (``verl/workers/rollout/atom_rollout/atom_rollout.py``): A thin client that implements ``BaseRollout``. Lives in the trainer worker process. Handles weight synchronization via ZMQ + CUDA IPC (or SHM fallback on ROCm).
- **ATOMHttpServer** (``verl/workers/rollout/atom_rollout/atom_async_server.py``): A Ray actor that owns the ATOM ``AsyncLLMEngine``. Runs in a separate process with explicit ``CUDA_VISIBLE_DEVICES`` control.
- **ATOMReplica** (``verl/workers/rollout/atom_rollout/atom_async_server.py``): Manages the lifecycle of ``ATOMHttpServer`` actors, one per node.

Weight Synchronization
~~~~~~~~~~~~~~~~~~~~~~

Weight transfer from the training engine to the ATOM rollout engine uses a two-stage pipeline:

1. **ServerAdapter → ATOMHttpServer** (ZMQ + CUDA IPC): The ServerAdapter serializes model weights into buckets and sends them via ZMQ IPC sockets. On platforms that support CUDA IPC (``hipIpcOpenMemHandle``), the data is transferred as IPC handles for zero-copy GPU-to-GPU transfer. On ROCm/MI300X where cross-GPU IPC is not supported, a POSIX shared memory (SHM) fallback is used.

2. **ATOMHttpServer → ModelRunner subprocesses** (engine.load_weights): The server opens the sender's IPC handle on cuda:0 (same-GPU, always safe), collects all weight tensors, then delegates to ``engine.load_weights(mode="ipc")``. The engine's internal ``load_weights_via_ipc`` handles per-GPU buffer allocation and IPC distribution to ModelRunner subprocesses, ensuring each subprocess only opens IPC handles for its own GPU.

Installation
------------

Ensure the ``atom`` package is installed in your environment. The verl integration will raise a ``PackageNotFoundError`` at startup if ``atom`` is not found.

Configuration
-------------

To use ATOM as the rollout engine, set:

.. code-block:: yaml

    actor_rollout_ref:
      rollout:
        name: atom
        engine_kwargs:
          atom:
            bucket_size_mb: 256  # Weight transfer bucket size in MB

Key configuration parameters:

- ``name: atom``: Select ATOM as the rollout engine.
- ``engine_kwargs.atom.bucket_size_mb``: Size of weight transfer buckets in MB (default: 256).
- ``engine_kwargs.atom.use_cuda_ipc``: Force CUDA IPC on/off for weight transfer. When unset, auto-detected via ``is_support_ipc()``.
- ``free_cache_engine``: When ``true``, the engine releases GPU memory (KV cache / weights) during training steps and restores them before rollout.
- ``tensor_model_parallel_size``: Tensor parallelism degree.
- ``data_parallel_size``: Data parallelism degree. A ``data_parallel_master_port`` is automatically allocated per replica to avoid port conflicts.

Limitations
-----------

- Pipeline parallelism (``pipeline_model_parallel_size > 1``) is not supported.
- LoRA adapters are not supported.
- Multimodal inputs (image/video) are not supported.
- Request-level abort is not yet implemented (``abort_all_requests`` is a no-op).
