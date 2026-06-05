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
from verl.workers.rollout.atom_rollout.constants import (
    ATOMDefaults,
    IPCConfig,
    SleepLevel,
)

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
            self.config.max_model_len = get_max_position_embeddings(
                self.model_config.hf_config
            )

        self.rollout_mode = rollout_mode
        self.workers = workers
        self.replica_rank = replica_rank
        self.node_rank = node_rank
        self.job_id = ray.get_runtime_context().get_job_id()
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
        if "max_new_tokens" in sampling_params:
            max_tokens = sampling_params.pop("max_new_tokens")
        temperature = sampling_params.pop("temperature", ATOMDefaults.TEMPERATURE)
        return_logprobs = sampling_params.pop("logprobs", False)
        top_k = sampling_params.pop("top_k", -1)
        top_p = sampling_params.pop("top_p", 1.0)
        n = sampling_params.pop("n", 1)

        for key in ("repetition_penalty",):
            if key in sampling_params:
                logger.debug(
                    f"Dropping unsupported sampling param: {key}={sampling_params.pop(key)}"
                )

        return SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=return_logprobs,
            top_k=top_k,
            top_p=top_p,
            n=n,
        )

    def _build_engine_kwargs(self) -> dict[str, Any]:
        """Build engine configuration dictionary."""
        dp_master_port, _ = get_free_port(self._server_address)
        engine_kwargs = {
            "model": self.model_config.local_path,
            "tensor_parallel_size": self.config.tensor_model_parallel_size,
            "data_parallel_size": self.config.data_parallel_size,
            "data_parallel_master_port": dp_master_port,
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
        import time as _time

        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel

        app = FastAPI(title="ATOM verl Server")

        class GenerateRequest(BaseModel):
            prompt_ids: list[int]
            sampling_params: dict[str, Any] = {}
            request_id: str = ""

        class ChatMessage(BaseModel):
            role: str
            content: str

        class ChatCompletionRequest(BaseModel):
            model: str
            messages: list[ChatMessage]
            temperature: float = 1.0
            top_p: float = 1.0
            n: int = 1
            max_tokens: int | None = None
            logprobs: bool = False

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

        @app.post("/v1/chat/completions")
        async def chat_completions(request: ChatCompletionRequest):
            try:
                tokenizer = self.model_config.tokenizer
                messages = [{"role": m.role, "content": m.content} for m in request.messages]
                prompt_ids = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True, return_dict=False
                )

                sampling_params = {
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "n": request.n,
                    "logprobs": request.logprobs,
                }
                if request.max_tokens is not None:
                    sampling_params["max_tokens"] = request.max_tokens

                request_id = str(uuid4())
                result = await self.generate(
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                    request_id=request_id,
                )

                output_text = tokenizer.decode(result.token_ids, skip_special_tokens=True)
                finish_reason = "stop" if result.stop_reason in ("completed", "stop") else "length"

                return {
                    "id": f"chatcmpl-{request_id}",
                    "object": "chat.completion",
                    "created": int(_time.time()),
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": output_text},
                            "finish_reason": finish_reason,
                        }
                    ],
                    "usage": {
                        "prompt_tokens": len(prompt_ids),
                        "completion_tokens": len(result.token_ids),
                        "total_tokens": len(prompt_ids) + len(result.token_ids),
                    },
                }
            except Exception as e:
                logger.error(f"Chat completion error: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))

        class ScoreRequest(BaseModel):
            model: str
            input: str
            tokenize: bool = True

        @app.post("/score")
        async def score_endpoint(request: ScoreRequest):
            """Score endpoint for discriminative reward model."""
            import math

            try:
                tokenizer = self.model_config.tokenizer
                if request.tokenize:
                    prompt_ids = tokenizer.encode(request.input)
                else:
                    prompt_ids = tokenizer.encode(
                        request.input, add_special_tokens=False
                    )

                sampling_params = {
                    "max_tokens": 1,
                    "temperature": 1.0,
                    "logprobs": True,
                }
                request_id = str(uuid4())
                result = await self.generate(
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                    request_id=request_id,
                )

                score = 0.0
                if result.log_probs and len(result.log_probs) > 0:
                    val = float(result.log_probs[0])
                    if not (math.isnan(val) or math.isinf(val)):
                        score = val

                return {
                    "id": f"score-{request_id}",
                    "object": "score",
                    "model": request.model,
                    "score": score,
                    "usage": {
                        "prompt_tokens": len(prompt_ids),
                        "total_tokens": len(prompt_ids) + 1,
                    },
                }
            except Exception as e:
                logger.error(f"Score error: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))

        @app.get("/health")
        async def health():
            return {"status": "ok"}

        self._server_port, self._server_task = await run_uvicorn(
            app, None, self._server_address
        )
        logger.info(
            f"HTTP server started at {self._server_address}:{self._server_port}"
        )

        self._batch_processor_task = asyncio.create_task(self._batch_processor_loop())
        logger.info(
            f"Batch processor started with batch_size={self._batch_size}, timeout={self._batch_timeout}s"
        )

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

    async def _process_batch(self, batch: list[tuple]):
        """Process a batch of requests together, calling the engine directly."""
        all_prompts = []
        all_sampling_params = []
        all_request_ids = []
        futures = []

        for prompt_ids, sp, request_id, future in batch:
            all_prompts.append(prompt_ids)
            all_sampling_params.append(sp)
            all_request_ids.append(request_id)
            futures.append(future)

        logger.debug(f"Processing batch of {len(batch)} requests")

        loop = asyncio.get_event_loop()

        def _generate_blocking():
            return self.engine.generate(
                all_prompts,
                all_sampling_params,
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
                    stop_reason = (
                        "completed"
                        if finish_reason in ("stop", "length")
                        else finish_reason
                    )

                    result = TokenOutput(
                        token_ids=token_ids,
                        log_probs=log_probs,
                        routed_experts=None,
                        stop_reason=stop_reason,
                    )
                    future.set_result(result)
                else:
                    future.set_exception(
                        RuntimeError(f"Missing output for request {i}")
                    )

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
            logger.warning(
                "ATOM does not support multimodal inputs. image_data/video_data will be ignored."
            )

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

        Receives weight buckets from ServerAdapter via ZMQ, then delegates
        to engine.load_weights() which handles per-GPU IPC distribution
        to ModelRunner subprocesses internally.

        The IPC path opens the sender's IPC handle on cuda:0 (same-GPU,
        always safe) and zero-copy views each weight tensor from the
        mapped buffer. All cross-GPU distribution is handled inside
        engine.load_weights(mode="ipc") via load_weights_via_ipc, which
        allocates per-GPU buffers within the EngineCore process tree
        where GPU visibility is correct.
        """
        from verl.workers.rollout.atom_rollout.bucketed_weight_transfer import rebuild_ipc

        ctx = zmq.Context()
        socket = ctx.socket(zmq.REP)
        zmq_handle = f"ipc:///tmp/rl-colocate-zmq-atom-{self.job_id}-replica-{self.replica_rank}-rank-0.sock"
        socket.connect(zmq_handle)

        # Receive IPC handle or shared memory metadata
        comm_metadata = socket.recv_pyobj()
        socket.send(b"")

        if not use_shm:
            ipc_buffer = rebuild_ipc(comm_metadata, device_id=0)
            logger.info(
                f"update_weights_from_zmq: opened IPC buffer "
                f"(size={ipc_buffer.numel()} bytes, device={ipc_buffer.device})"
            )
        else:
            shm = shared_memory.SharedMemory(name=comm_metadata["name"])
            ipc_buffer = torch.frombuffer(
                shm.buf[: comm_metadata["size"]], dtype=torch.uint8
            )

        all_weights = []
        while True:
            metadata = socket.recv_pyobj()
            is_last = metadata["is_last"]
            for name, meta in metadata["bucket_meta"].items():
                shape = meta["shape"]
                dtype = meta["dtype"]
                offset = meta["offset"]
                handle = meta.get("handle")
                if handle is not None:
                    tensor = rebuild_ipc(handle, 0)
                else:
                    nbytes = dtype.itemsize * torch.Size(shape).numel()
                    tensor = ipc_buffer[offset : offset + nbytes].view(dtype=dtype).view(shape)
                    if use_shm:
                        tensor = tensor.to("cuda:0")
                    elif not is_last:
                        tensor = tensor.clone()
                all_weights.append((name, tensor))
            torch.cuda.synchronize()
            socket.send(b"")
            if is_last:
                break

        socket.close()
        ctx.term()

        atom_kwargs = (getattr(self.config, "engine_kwargs", {}) or {}).get(
            "atom", {}
        ) or {}
        bucket_size_mb = atom_kwargs.get(
            "bucket_size_mb", IPCConfig.DEFAULT_BUCKET_SIZE_MB
        )
        logger.info(
            f"update_weights_from_zmq: loading {len(all_weights)} weight tensors"
        )
        num_gpus = self.config.tensor_model_parallel_size * self.config.data_parallel_size
        self.engine.load_weights(
            iter(all_weights), bucket_size_mb=bucket_size_mb,
            num_gpus=num_gpus, mode="ipc",
        )

        del all_weights
        if use_shm:
            del ipc_buffer
            shm.close()
        else:
            del ipc_buffer
            torch.cuda.ipc_collect()

        logger.info("update_weights_from_zmq: load_weights completed")
        gc.collect()
        torch.cuda.empty_cache()

    async def clear_kv_cache(self):
        """Clear KV cache in the engine."""
        if self.engine is not None:
            if hasattr(self.engine, "core_mgr"):
                self.engine.core_mgr.broadcast_utility_command("clear_kv_cache")
            elif hasattr(self.engine, "clear_kv_cache"):
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
        is_teacher_model: bool = False,
        name_suffix: str = "",
    ):
        super().__init__(
            replica_rank, config, model_config, gpus_per_node, is_reward_model, is_teacher_model, name_suffix
        )
        self.server_class = ray.remote(ATOMHttpServer)

    def _get_server_name_prefix(self) -> str:
        if self.is_reward_model:
            return "atom_server_reward"
        elif self.is_teacher_model:
            return "atom_server_teacher"
        return "atom_server"

    async def launch_servers(self):
        assert (
            len(self.workers) == self.world_size
        ), f"worker number {len(self.workers)} not equal to world size {self.world_size}"

        # 1. Get (node_id, GPU accelerator ID) for each worker
        worker_infos = await asyncio.gather(
            *[
                worker.__ray_call__.remote(
                    lambda self: (
                        ray.get_runtime_context().get_node_id(),
                        ray.get_runtime_context().get_accelerator_ids()[
                            get_resource_name()
                        ][0],
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
            prefix = self._get_server_name_prefix()
            name = f"{prefix}_{self.replica_rank}_{node_rank}{self.name_suffix}"

            server = self.server_class.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=node_id,
                    soft=False,
                ),
                runtime_env={
                    "env_vars": {
                        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                    }
                },
                name=name,
                max_concurrency=self.max_concurrency,
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

        logger.info(
            f"ATOMReplica {self.replica_rank}: Server launched at {self._server_address}"
        )
