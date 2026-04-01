import asyncio
import gc
import logging
import os
from multiprocessing import shared_memory
from typing import Any
from uuid import uuid4

import ray
import torch
import zmq
from ray.actor import ActorHandle

from verl.single_controller.ray import RayClassWithInitArgs
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import get_resource_name, get_visible_devices_keyword
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.replica import RolloutMode, RolloutReplica, TokenOutput
from verl.utils.net_utils import get_free_port, is_valid_ipv6_address
from verl.workers.rollout.utils import (
    get_max_position_embeddings,
    run_uvicorn,
)
from verl.workers.rollout.atom_rollout.constants import ATOMDefaults, IPCConfig, SleepLevel
from verl.workers.rollout.atom_rollout.utils import get_device_uuid

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ATOMHttpServer:

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        rollout_mode: RolloutMode,
        workers: list[ActorHandle],
        replica_rank: int,
        node_rank: int,
        gpus_per_node: int,
        nnodes: int,
        cuda_visible_devices: str,
    ):
        os.environ[get_visible_devices_keyword()] = cuda_visible_devices

        self.config: RolloutConfig = omega_conf_to_dataclass(config)
        self.model_config: HFModelConfig = omega_conf_to_dataclass(
            model_config, dataclass_type=HFModelConfig
        )

        if self.config.max_model_len is None or self.config.max_model_len <= 0:
            self.config.max_model_len = get_max_position_embeddings(self.model_config.hf_config)

        self.rollout_mode = rollout_mode
        self.workers = workers
        self.replica_rank = replica_rank
        self.node_rank = node_rank
        self.gpus_per_node = gpus_per_node
        self.nnodes = nnodes
        self.engine = None

        # HTTP server
        self._server_address = ray.util.get_node_ip_address().strip("[]")
        self._server_port = None

        if self.node_rank == 0:
            self._master_address = self._server_address
            self._master_port, _ = get_free_port(self._server_address)

        # Batch request collection for DP parallelism
        self._batch_size, self._batch_timeout = self._compute_batch_params()
        self._pending_requests: list[tuple] = []
        self._batch_lock = asyncio.Lock()
        self._batch_event = asyncio.Event()
        self._batch_processor_task = None

        logger.info(
            f"ATOMHttpServer, replica_rank: {self.replica_rank}, node_rank: {self.node_rank}, "
            f"{get_visible_devices_keyword()}: {cuda_visible_devices}, "
            f"master_address: {getattr(self, '_master_address', None)}, "
            f"master_port: {getattr(self, '_master_port', None)}"
        )

    def _compute_batch_params(self) -> tuple:
        dp_size = self.config.data_parallel_size
        max_seqs = self.config.max_num_seqs
        batch_size = dp_size * max_seqs
        batch_timeout = ATOMDefaults.BATCH_TIMEOUT
        logger.info(
            f"Batch collection params: batch_size={batch_size}, "
            f"batch_timeout={batch_timeout*1000:.0f}ms "
            f"(dp_size={dp_size}, max_num_seqs={max_seqs})"
        )
        return batch_size, batch_timeout

    def get_master_address(self):
        return self._master_address, self._master_port

    def get_server_address(self):
        assert self._server_port is not None
        return self._server_address, self._server_port

    async def launch_server(self, master_address: str = None, master_port: int = None):
        """Launch the ATOM HTTP server and create the engine."""
        if self.node_rank != 0:
            self._master_address = master_address
            self._master_port = master_port

        engine_kwargs = self._build_engine_kwargs()
        from atom.rollout.async_engine import AsyncLLMEngine
        self.engine = AsyncLLMEngine(**engine_kwargs)
        logger.info("ATOMHttpServer: AsyncLLMEngine created")

        if self.node_rank == 0:
            await self._launch_http_server()

    def _build_sampling_params(self, sampling_params: dict[str, Any]):
        """Build ATOM SamplingParams from request parameters."""
        from atom.sampling_params import SamplingParams

        max_tokens = sampling_params.pop("max_tokens", self.config.response_length)
        temperature = sampling_params.pop("temperature", ATOMDefaults.TEMPERATURE)
        return_logprobs = sampling_params.pop("logprobs", False)

        unsupported_keys = ("top_p", "top_k", "repetition_penalty", "max_new_tokens")
        for key in unsupported_keys:
            if key in sampling_params:
                logger.debug(f"Dropping unsupported sampling param: {key}={sampling_params.pop(key)}")


        return SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=return_logprobs,
        )

    def _build_engine_kwargs(self) -> dict[str, Any]:
        """Build engine configuration dictionary."""
        engine_kwargs = {
            "model": self.model_config.local_path,
            "tensor_parallel_size": self.config.tensor_model_parallel_size,
            "data_parallel_size": self.config.data_parallel_size,
            "enable_expert_parallel": self.config.expert_parallel_size > 1,
            "max_num_seqs": self.config.max_num_seqs,
            "max_model_len": self.config.max_model_len,
            "gpu_memory_utilization": self.config.gpu_memory_utilization,
            "enforce_eager": self.config.enforce_eager,
            "trust_remote_code": self.model_config.trust_remote_code,
            "load_dummy": self.config.load_format == "dummy",
        }

        if self.config.cudagraph_capture_sizes:
            engine_kwargs["compilation_config"] = {
                "cudagraph_capture_sizes": self.config.cudagraph_capture_sizes,
            }

        if hasattr(self.config, "engine_kwargs") and self.config.engine_kwargs:
            # Copy to avoid mutating the original config (bucket_size_mb is
            # read later by _update_weights_from_zmq_sync).
            atom_kwargs = dict(self.config.engine_kwargs.get("atom", {}) or {})
            # Remove verl-specific keys that aren't engine params
            atom_kwargs.pop("bucket_size_mb", None)
            atom_kwargs.pop("use_cuda_ipc", None)
            engine_kwargs.update(atom_kwargs)

        return engine_kwargs

    async def _launch_http_server(self):
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel

        app = FastAPI(title="ATOM verl Server")

        class GenerateRequest(BaseModel):
            prompt_ids: list[int]
            sampling_params: dict[str, Any] = {}
            request_id: str = ""

        @app.post("/generate")
        async def generate_endpoint(request: GenerateRequest):
            try:
                result = await self.generate(
                    prompt_ids=request.prompt_ids,
                    sampling_params=request.sampling_params,
                    request_id=request.request_id or str(uuid4()),
                )
                return result.model_dump()
            except Exception as e:
                logger.error(f"Generate error: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))

        @app.get("/health")
        async def health():
            return {"status": "ok"}

        self._server_port, self._server_task = await run_uvicorn(
            app, None, self._server_address
        )
        logger.info(f"HTTP server started at {self._server_address}:{self._server_port}")

        self._batch_processor_task = asyncio.create_task(self._batch_processor_loop())
        logger.info(f"Batch processor started with batch_size={self._batch_size}, timeout={self._batch_timeout}s")

    async def _batch_processor_loop(self):
        """Background task that processes batched requests."""
        while True:
            try:
                await self._batch_event.wait()

                deadline = asyncio.get_event_loop().time() + self._batch_timeout
                while True:
                    async with self._batch_lock:
                        if len(self._pending_requests) >= self._batch_size:
                            break
                    remaining = deadline - asyncio.get_event_loop().time()
                    if remaining <= 0:
                        break
                    await asyncio.sleep(min(0.005, remaining))

                async with self._batch_lock:
                    if not self._pending_requests:
                        self._batch_event.clear()
                        continue
                    batch = self._pending_requests
                    self._pending_requests = []
                    self._batch_event.clear()

                if batch:
                    logger.info(f"Dispatching batch of {len(batch)} requests")
                    await self._process_batch(batch)

            except Exception as e:
                logger.error(f"Batch processor error: {e}", exc_info=True)
                async with self._batch_lock:
                    for _, _, _, future in self._pending_requests:
                        if not future.done():
                            future.set_exception(e)
                    self._pending_requests = []

    async def _process_batch(self, batch: list[tuple]):
        """Process a batch of requests together, calling the engine directly."""
        all_prompts = []
        all_request_ids = []
        futures = []

        first_sp = batch[0][1] if batch else None

        for prompt_ids, sp, request_id, future in batch:
            all_prompts.append(prompt_ids)
            all_request_ids.append(request_id)
            futures.append(future)

        logger.debug(f"Processing batch of {len(batch)} requests")

        loop = asyncio.get_event_loop()

        def _generate_blocking():
            return self.engine.generate(
                all_prompts,
                first_sp,
                request_ids=all_request_ids,
            )

        try:
            outputs = await loop.run_in_executor(None, _generate_blocking)

            for i, future in enumerate(futures):
                if i < len(outputs):
                    output = outputs[i]
                    token_ids = output.get("token_ids", [])
                    log_probs = output.get("logprobs", None)
                    finish_reason = output.get("finish_reason", "stop")
                    stop_reason = "completed" if finish_reason in ("stop", "length") else finish_reason

                    result = TokenOutput(
                        token_ids=token_ids,
                        log_probs=log_probs,
                        routed_experts=None,
                        stop_reason=stop_reason,
                    )
                    future.set_result(result)
                else:
                    future.set_exception(RuntimeError(f"Missing output for request {i}"))

        except Exception as e:
            logger.error(f"Batch processing error: {e}", exc_info=True)
            for future in futures:
                if not future.done():
                    future.set_exception(e)

    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: list[Any] | None = None,
        video_data: list[Any] | None = None,
    ) -> TokenOutput:
        """Generate sequence with batch collection for DP parallelism."""
        if image_data or video_data:
            logger.warning("ATOM does not support multimodal inputs. image_data/video_data will be ignored.")

        sp = self._build_sampling_params(sampling_params)

        loop = asyncio.get_event_loop()
        future = loop.create_future()

        async with self._batch_lock:
            self._pending_requests.append((prompt_ids, sp, request_id, future))
            if len(self._pending_requests) >= self._batch_size:
                self._batch_event.set()

        self._batch_event.set()

        return await future

    async def wake_up(self, tags=None):
        """Wake up the engine (restore weights/kv_cache to GPU)."""
        if self.rollout_mode in (RolloutMode.HYBRID, RolloutMode.COLOCATED):
            if self.engine is not None:
                self.engine.wake_up(tags=tags or ["kv_cache", "weights"])

    async def sleep(self, level=None):
        """Put the engine to sleep (release weights/kv_cache from GPU)."""
        if not self.config.free_cache_engine:
            return
        if self.rollout_mode in (RolloutMode.HYBRID, RolloutMode.COLOCATED):
            if self.engine is not None:
                sleep_level = level or ATOMDefaults.SLEEP_LEVEL
                self.engine.sleep(level=sleep_level)

    async def update_weights_from_zmq(self, use_shm=False, **kwargs):
        """Receive weights from training worker via ZMQ IPC and load into engine.

        Runs synchronously (blocking the actor event loop) because:
        1. engine.load_weights() is not thread-safe — calling from a background
           thread causes deadlocks with the engine's internal SHM distribution.
        2. The client waits for this method to complete anyway, so blocking is fine.
        """
        self._update_weights_from_zmq_sync(use_shm=use_shm)

    def _update_weights_from_zmq_sync(self, use_shm=False):
        """Synchronous ZMQ weight receive and engine weight load.

        Weight transfer strategy:
        - When use_shm=False (CUDA IPC): ATOMHttpServer opens the IPC handle
          from ServerAdapter (same-GPU IPC on cuda:0, always safe), then
          copies weight data to per-GPU buffers via D2D copy and distributes
          per-GPU IPC handles to ModelRunner subprocesses. Each ModelRunner
          opens ONLY its own GPU's handle — always same-GPU IPC, no cross-GPU
          hipIpcOpenMemHandle. This avoids the ROCm/MI300X crash where opening
          an IPC handle from a different physical GPU causes a "Memory access
          fault".
        - When use_shm=True (SHM fallback): weights are read from POSIX SHM,
          copied to GPU, then distributed via engine.load_weights().
        """
        from torch.multiprocessing.reductions import reduce_tensor

        ctx = zmq.Context()
        socket = ctx.socket(zmq.REP)
        # Use device 0 since ATOMHttpServer's CUDA_VISIBLE_DEVICES
        # is set to the exact GPUs for this replica
        device_uuid = get_device_uuid(0)
        zmq_handle = f"ipc:///tmp/rl-colocate-zmq-atom-{device_uuid}.sock"
        socket.connect(zmq_handle)  # ServerAdapter binds, server connects

        # Receive IPC handle or shared memory metadata
        comm_metadata = socket.recv_pyobj()
        socket.send(b"")

        if not use_shm:
            # Per-GPU IPC path: open the IPC handle from ServerAdapter on
            # cuda:0 (same-GPU, always safe), then replicate to per-GPU
            # buffers and send per-GPU IPC handles to ModelRunners.
            from atom.rollout.weight_sync import rebuild_ipc_handle

            ipc_buffer = rebuild_ipc_handle(comm_metadata, device_id=0)
            bucket_size = ipc_buffer.numel()
            logger.info(
                f"update_weights_from_zmq: opened IPC buffer "
                f"(size={bucket_size} bytes, device={ipc_buffer.device})"
            )

            # Determine total number of GPUs used by this engine
            tp_size = self.config.tensor_model_parallel_size
            dp_size = self.config.data_parallel_size
            num_gpus = tp_size * dp_size

            # Allocate per-GPU buffers and create IPC handles
            per_gpu_buffers = {}
            per_gpu_ipc_handles = {}
            for i in range(num_gpus):
                buf = torch.empty(bucket_size, dtype=torch.uint8, device=f"cuda:{i}")
                per_gpu_buffers[i] = buf
                per_gpu_ipc_handles[i] = reduce_tensor(buf)
            logger.info(
                f"update_weights_from_zmq: allocated {num_gpus} per-GPU IPC "
                f"buffers ({bucket_size / (1 << 20):.1f} MiB each)"
            )

            while True:
                metadata = socket.recv_pyobj()
                raw_bucket_meta = metadata["bucket_meta"]
                is_last = metadata["is_last"]

                # Convert bucket_meta from ZMQ format (dtype=torch.dtype, no nbytes)
                # to weight_updater format (dtype=str, with nbytes)
                bucket_meta = {}
                used_bytes = 0
                for name, meta in raw_bucket_meta.items():
                    shape = meta["shape"]
                    dtype = meta["dtype"]
                    offset = meta["offset"]
                    nbytes = dtype.itemsize * torch.Size(shape).numel()
                    bucket_meta[name] = {
                        "shape": tuple(shape),
                        "dtype": str(dtype),
                        "offset": offset,
                        "nbytes": nbytes,
                    }
                    end = offset + nbytes
                    if end > used_bytes:
                        used_bytes = end

                # D2D copy from cuda:0 IPC buffer to each per-GPU buffer
                src_slice = ipc_buffer[:used_bytes]
                for i in range(num_gpus):
                    per_gpu_buffers[i][:used_bytes].copy_(src_slice, non_blocking=True)
                # Synchronize all GPUs to ensure D2D copies are complete
                for i in range(num_gpus):
                    torch.cuda.synchronize(i)

                # Send per-GPU IPC handles to ModelRunners
                self.engine.core_mgr.broadcast_utility_command_sync(
                    "update_weights_ipc",
                    ipc_handle=None,
                    ipc_handles=per_gpu_ipc_handles,
                    bucket_meta=bucket_meta,
                    is_last=is_last,
                )

                # ACK to ServerAdapter
                socket.send(b"")
                if is_last:
                    break

            socket.close()
            ctx.term()

            # Cleanup per-GPU buffers and IPC handles.
            # CRITICAL: must call empty_cache() on EVERY GPU to return
            # the 4 GB per-GPU IPC buffer memory to HIP.  Without this,
            # PyTorch's caching allocator keeps the memory allocated,
            # reducing available GPU memory for KV cache in ModelRunner
            # subprocesses — which causes Memory access faults on
            # memory-constrained GPUs (especially DP rank 1).
            del per_gpu_buffers
            del per_gpu_ipc_handles
            del ipc_buffer
            gc.collect()
            torch.cuda.ipc_collect()
            for i in range(num_gpus):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
        else:
            # SHM fallback: collect all weights to GPU, then use engine.load_weights()
            shm = shared_memory.SharedMemory(name=comm_metadata["name"])
            buffer = torch.frombuffer(shm.buf[:comm_metadata["size"]], dtype=torch.uint8)
            all_weights = []
            while True:
                metadata = socket.recv_pyobj()
                for name, meta in metadata["bucket_meta"].items():
                    shape, dtype, offset = meta["shape"], meta["dtype"], meta["offset"]
                    nbytes = dtype.itemsize * torch.Size(shape).numel()
                    tensor = buffer[offset:offset + nbytes].view(dtype=dtype).view(shape)
                    tensor = tensor.to("cuda:0")
                    all_weights.append((name, tensor))
                torch.cuda.synchronize()
                socket.send(b"")
                if metadata["is_last"]:
                    break

            socket.close()
            ctx.term()
            del buffer
            shm.close()

            atom_kwargs = (getattr(self.config, "engine_kwargs", {}) or {}).get("atom", {}) or {}
            bucket_size_mb = atom_kwargs.get("bucket_size_mb", IPCConfig.DEFAULT_BUCKET_SIZE_MB)
            logger.info(f"update_weights_from_zmq: loading {len(all_weights)} weight tensors via SHM")
            self.engine.load_weights(iter(all_weights), bucket_size_mb=bucket_size_mb)
            del all_weights

        logger.info("update_weights_from_zmq: load_weights completed")
        gc.collect()
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

    async def clear_kv_cache(self):
        """Clear KV cache in the engine."""
        if self.engine is not None:
            if hasattr(self.engine, 'core_mgr'):
                self.engine.core_mgr.broadcast_utility_command('clear_kv_cache')
            elif hasattr(self.engine, 'clear_kv_cache'):
                self.engine.clear_kv_cache()

    async def set_global_steps(self, global_steps: int):
        """Set the global steps of the model weights."""
        self.global_steps = global_steps

    async def abort_all_requests(self) -> dict:
        """Abort all ongoing generation requests.

        ATOM does not currently support request-level abort.
        Returns an empty result for interface compatibility.
        """
        logger.info("abort_all_requests called (no-op for ATOM)")
        return {"aborted_count": 0, "request_ids": []}

    async def resume_generation(self):
        """Resume generation after abort_all_requests.

        No-op for ATOM — abort_all_requests is already a no-op.
        """
        pass

    async def start_profile(self, **kwargs):
        """Start profiling. No-op for ATOM."""
        logger.debug("start_profile called (no-op for ATOM)")

    async def stop_profile(self):
        """Stop profiling. No-op for ATOM."""
        logger.debug("stop_profile called (no-op for ATOM)")

    @property
    def lora_as_adapter(self) -> bool:
        """Whether LoRA is used as adapter. ATOM does not support LoRA."""
        return False


class ATOMReplica(RolloutReplica):
    """ATOM rollout replica manager."""

    def __init__(
        self,
        replica_rank: int,
        config: RolloutConfig,
        model_config: HFModelConfig,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
    ):
        super().__init__(replica_rank, config, model_config, gpus_per_node, is_reward_model)
        self.server_class = ray.remote(ATOMHttpServer)

    def get_ray_class_with_init_args(self) -> RayClassWithInitArgs:
        from verl.workers.rollout.atom_rollout.atom_rollout import ServerAdapter
        _rollout_worker_actor_cls = ray.remote(ServerAdapter)
        return RayClassWithInitArgs(
            cls=_rollout_worker_actor_cls,
            config=self.config,
            model_config=self.model_config,
            device_mesh=None,
        )

    async def launch_servers(self):
        assert len(self.workers) == self.world_size, (
            f"worker number {len(self.workers)} not equal to world size {self.world_size}"
        )

        # 1. Get (node_id, GPU accelerator ID) for each worker
        worker_infos = await asyncio.gather(
            *[
                worker.__ray_call__.remote(
                    lambda self: (
                        ray.get_runtime_context().get_node_id(),
                        ray.get_runtime_context().get_accelerator_ids()[get_resource_name()][0],
                    )
                )
                for worker in self.workers
            ]
        )
        worker_node_ids = [info[0] for info in worker_infos]
        worker_gpu_ids = [info[1] for info in worker_infos]

        # 2. Create server actor per node with node affinity and explicit CUDA_VISIBLE_DEVICES
        nnodes, gpus_per_replica_node = self.nnodes, self.gpus_per_replica_node
        for node_rank in range(nnodes):
            start = node_rank * gpus_per_replica_node
            end = start + gpus_per_replica_node
            node_cuda_visible_devices = ",".join(worker_gpu_ids[start:end])
            node_id = worker_node_ids[start]
            name = f"atom_server_{self.replica_rank}_{node_rank}"

            server = self.server_class.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=node_id,
                    soft=False,
                ),
                runtime_env={"env_vars": {
                    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                }},
                name=name,
            ).remote(
                config=self.config,
                model_config=self.model_config,
                rollout_mode=self.rollout_mode,
                workers=self.workers[start:end],
                replica_rank=self.replica_rank,
                node_rank=node_rank,
                gpus_per_node=gpus_per_replica_node,
                nnodes=nnodes,
                cuda_visible_devices=node_cuda_visible_devices,
            )
            self.servers.append(server)

        # 3. Launch servers
        master_address, master_port = await self.servers[0].get_master_address.remote()
        await asyncio.gather(
            *[
                server.launch_server.remote(
                    master_address=master_address,
                    master_port=master_port,
                )
                for server in self.servers
            ]
        )

        # 4. Get HTTP server address from first server
        server_address, server_port = await self.servers[0].get_server_address.remote()
        self._server_handle = self.servers[0]
        self._server_address = (
            f"[{server_address}]:{server_port}"
            if is_valid_ipv6_address(server_address)
            else f"{server_address}:{server_port}"
        )

        logger.info(f"ATOMReplica {self.replica_rank}: Server launched at {self._server_address}")
