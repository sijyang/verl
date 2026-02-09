import asyncio
import logging
import os
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from uuid import uuid4

import cloudpickle as pickle
import ray
import zmq
from ray.actor import ActorHandle

from verl.single_controller.ray import RayClassWithInitArgs
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.replica import RolloutMode, RolloutReplica, TokenOutput
from verl.workers.rollout.utils import (
    get_free_port,
    get_max_position_embeddings,
    is_valid_ipv6_address,
    run_unvicorn,
)
from verl.workers.rollout.atom_rollout.atom_rollout import ATOMAsyncRollout
from verl.workers.rollout.atom_rollout.constants import ATOMDefaults

if TYPE_CHECKING:
    from verl.single_controller.ray import RayWorkerGroup
    from verl.trainer.ppo.ray_trainer import RayResourcePool

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


class ZMQExecutor:
    """
    Executor that communicates with ATOM worker via ZeroMQ.
    """

    def __init__(self, worker_address: str):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        if worker_address.startswith("tcp://["):
            self.socket.setsockopt(zmq.IPV6, 1)
        self.socket.connect(worker_address)
        logger.info(f"ZMQExecutor connected to {worker_address}")

    def rpc(self, method: str, *args, **kwargs) -> Any:
        """Send RPC call to worker and get response."""
        message = pickle.dumps((method, args, kwargs))
        self.socket.send(message)
        result = pickle.loads(self.socket.recv())
        if isinstance(result, Exception):
            raise result
        return result

    def close(self):
        """Close the socket."""
        self.socket.close()
        self.context.term()


class ATOMHttpServer:
    """
    ATOM HTTP server that coordinates with worker via ZMQ.
    
    Features batch request collection for better DP utilization:
    - Concurrent requests are collected in a queue
    - When batch_size is reached or timeout expires, requests are sent together
    - This allows ATOM's round-robin to distribute across DP ranks effectively
    """

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        rollout_mode: RolloutMode,
        workers: List[ActorHandle],
        replica_rank: int,
        node_rank: int,
        gpus_per_node: int,
        nnodes: int,
    ):
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
        
        # ZMQ executor (initialized in launch_server)
        self.executor: Optional[ZMQExecutor] = None
        
        # HTTP server
        self._server_address = ray.util.get_node_ip_address().strip("[]")
        self._server_port = None
        
        if self.node_rank == 0:
            self._master_address = self._server_address
            self._master_port, _ = get_free_port(self._server_address)
        
        # Batch request collection for DP parallelism
        # Collect concurrent requests and send them together to utilize all DP ranks
        self._batch_size, self._batch_timeout = self._compute_batch_params()
        self._pending_requests: List[tuple] = []  # [(prompt_ids, sp, request_id, future), ...]
        self._batch_lock = asyncio.Lock()
        self._batch_event = asyncio.Event()
        self._batch_processor_task = None
    
    def _compute_batch_params(self) -> tuple:

        dp_size = self.config.data_parallel_size
        max_seqs = self.config.max_num_seqs
        
        batch_size = max(dp_size * 2, max_seqs)
        batch_timeout = 0.01
        
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
        """Launch the ATOM HTTP server."""
        if self.node_rank != 0:
            self._master_address = master_address
            self._master_port = master_port

        # 1. Get worker ZMQ address
        zmq_address = await self.workers[0].get_zeromq_address.remote()
        logger.info(
            f"ATOMHttpServer replica={self.replica_rank}, node={self.node_rank}: "
            f"worker ZMQ address: {zmq_address}"
        )
        
        # 2. Create ZMQ executor
        self.executor = ZMQExecutor(zmq_address)
        
        # 3. Build engine kwargs
        engine_kwargs = self._build_engine_kwargs()
        
        # 4. Initialize worker's LLMEngine
        self.executor.rpc("init_worker", [engine_kwargs])
        logger.info("ATOMHttpServer: Worker LLMEngine initialized")
        
        # 5. Launch HTTP server
        if self.node_rank == 0:
            await self._launch_http_server()

    def _build_sampling_params(self, sampling_params: Dict[str, Any]):
        """Build ATOM SamplingParams from request parameters.
        
        ATOM SamplingParams only supports a subset of parameters:
        temperature, max_tokens, ignore_eos, stop_strings, logprobs.
        
        Unsupported parameters (top_p, top_k, repetition_penalty, etc.)
        are silently ignored.
        """
        from atom.sampling_params import SamplingParams
        
        # Extract supported parameters with defaults
        max_tokens = sampling_params.pop("max_tokens", self.config.response_length)
        temperature = sampling_params.pop("temperature", ATOMDefaults.TEMPERATURE)
        return_logprobs = sampling_params.pop("logprobs", False)
        
        # Pop unsupported params to avoid errors
        for unsupported in ("top_p", "top_k", "repetition_penalty", "max_new_tokens"):
            sampling_params.pop(unsupported, None)
        
        return SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=return_logprobs,
        )

    def _build_engine_kwargs(self) -> Dict[str, Any]:
        """Build engine configuration dictionary."""
        engine_kwargs = {
            "model": self.model_config.local_path,
            "tensor_parallel_size": self.config.tensor_model_parallel_size,
            "data_parallel_size": self.config.data_parallel_size,
            "max_num_seqs": self.config.max_num_seqs,
            "max_model_len": self.config.max_model_len,
            "gpu_memory_utilization": self.config.gpu_memory_utilization,
            "enforce_eager": self.config.enforce_eager,
            "trust_remote_code": self.model_config.trust_remote_code,
            "load_dummy": self.config.load_format == "dummy",
        }
        
        # Note: ATOM's sleep()/wake_up() methods are always available and don't need
        # enable_sleep_mode config. The memory management is controlled by verl's
        # free_cache_engine config which determines when to call sleep()/wake_up().
        
        # Add cudagraph config
        if self.config.cudagraph_capture_sizes:
            engine_kwargs["compilation_config"] = {
                "cudagraph_capture_sizes": self.config.cudagraph_capture_sizes,
            }
        
        # Add extra engine kwargs from config
        if hasattr(self.config, "engine_kwargs") and self.config.engine_kwargs:
            atom_kwargs = self.config.engine_kwargs.get("atom", {}) or {}
            engine_kwargs.update(atom_kwargs)
        
        return engine_kwargs

    async def _launch_http_server(self):
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel

        app = FastAPI(title="ATOM verl Server")

        class GenerateRequest(BaseModel):
            prompt_ids: List[int]
            sampling_params: Dict[str, Any] = {}
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

        self._server_port, self._server_task = await run_unvicorn(
            app, None, self._server_address
        )
        logger.info(f"HTTP server started at {self._server_address}:{self._server_port}")
        
        # Start batch processor task
        self._batch_processor_task = asyncio.create_task(self._batch_processor_loop())
        logger.info(f"Batch processor started with batch_size={self._batch_size}, timeout={self._batch_timeout}s")

    async def _batch_processor_loop(self):
        """Background task that processes batched requests."""
        while True:
            try:
                # Wait for batch event or timeout
                try:
                    await asyncio.wait_for(self._batch_event.wait(), timeout=self._batch_timeout)
                except asyncio.TimeoutError:
                    pass
                
                # Check if we have pending requests
                async with self._batch_lock:
                    if not self._pending_requests:
                        self._batch_event.clear()
                        continue
                    
                    # Take all pending requests
                    batch = self._pending_requests
                    self._pending_requests = []
                    self._batch_event.clear()
                
                # Process batch
                if batch:
                    await self._process_batch(batch)
                    
            except Exception as e:
                logger.error(f"Batch processor error: {e}", exc_info=True)
                # On error, fail all pending requests
                async with self._batch_lock:
                    for _, _, _, future in self._pending_requests:
                        if not future.done():
                            future.set_exception(e)
                    self._pending_requests = []
    
    async def _process_batch(self, batch: List[tuple]):
        """Process a batch of requests together.
        
        This sends all prompts to ATOM in one call, allowing the engine's
        round-robin to distribute them across DP ranks for true parallelism.
        """
        from atom.sampling_params import SamplingParams
        
        # Extract batch data
        all_prompts = []
        all_request_ids = []
        futures = []
        
        # Use the first request's sampling params (they should all be similar)
        first_sp = batch[0][1] if batch else None
        
        for prompt_ids, sp, request_id, future in batch:
            all_prompts.append(prompt_ids)
            all_request_ids.append(request_id)
            futures.append(future)
        
        logger.debug(f"Processing batch of {len(batch)} requests")
        
        try:
            # Call worker with batch of prompts
            # ATOM's add_request will distribute across DP ranks via round-robin
            outputs = self.executor.rpc(
                "generate",
                all_prompts,  # List of token lists
                first_sp,
                request_ids=all_request_ids,
            )
            
            # Distribute results back to futures
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
        prompt_ids: List[int],
        sampling_params: Dict[str, Any],
        request_id: str,
        image_data: Optional[List[Any]] = None,
        video_data: Optional[List[Any]] = None,
    ) -> TokenOutput:
        """Generate sequence with batch collection for DP parallelism.
        
        Concurrent requests are collected and sent together to ATOM,
        allowing the engine's round-robin to distribute across DP ranks.
        
        Note: image_data and video_data are accepted for API compatibility but
        ATOM currently does not support multimodal inputs.
        """
        if image_data or video_data:
            logger.warning("ATOM does not support multimodal inputs. image_data/video_data will be ignored.")
        
        # Build sampling params
        sp = self._build_sampling_params(sampling_params)
        
        # Create future for this request's result
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        
        # Add to pending requests
        async with self._batch_lock:
            self._pending_requests.append((prompt_ids, sp, request_id, future))
            
            # Signal batch processor if we have enough requests
            if len(self._pending_requests) >= self._batch_size:
                self._batch_event.set()
        
        # Also set event to ensure timeout-based processing
        self._batch_event.set()
        
        # Wait for result
        return await future

    async def wake_up(self):
        if self.rollout_mode == RolloutMode.HYBRID:
            # Call all workers to switch between trainer mode and rollout mode.
            await asyncio.gather(*[worker.wake_up.remote() for worker in self.workers])
        elif self.rollout_mode == RolloutMode.COLOCATED:
            # Directly call engine to wake up without sync weights.
            if self.executor:
                self.executor.rpc("broadcast_utility_command", "resume_memory", tags=["kv_cache", "weights"])
        elif self.rollout_mode == RolloutMode.STANDALONE:
            logger.info("skip wake_up in standalone mode")

    async def sleep(self):
        if self.rollout_mode == RolloutMode.HYBRID:
            # Clear KV cache before sleep
            await self.clear_kv_cache()
            await asyncio.gather(*[worker.sleep.remote() for worker in self.workers])
        elif self.rollout_mode == RolloutMode.COLOCATED:
            await self.clear_kv_cache()
            if self.executor:
                self.executor.rpc("broadcast_utility_command", "release_memory", tags=["kv_cache"])
        elif self.rollout_mode == RolloutMode.STANDALONE:
            logger.info("skip sleep in standalone mode")

    async def clear_kv_cache(self):
        """Clear KV cache."""
        if self.executor:
            self.executor.rpc("broadcast_utility_command", "clear_kv_cache")


# Ray remote worker class
_rollout_worker_actor_cls = ray.remote(ATOMAsyncRollout)


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
        return RayClassWithInitArgs(
            cls=_rollout_worker_actor_cls,
            config=self.config,
            model_config=self.model_config,
            device_mesh=None,
        )

    async def launch_servers(self):

        worker_node_ids = await asyncio.gather(
            *[
                worker.__ray_call__.remote(
                    lambda self: ray.get_runtime_context().get_node_id()
                )
                for worker in self.workers
            ]
        )
        
        # Create server on first worker's node
        node_id = worker_node_ids[0]
        
        server = self.server_class.options(
            scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                node_id=node_id,
                soft=False,
            ),
            name=f"atom_server_{self.replica_rank}_{uuid4().hex[:8]}",
        ).remote(
            config=self.config,
            model_config=self.model_config,
            rollout_mode=self.rollout_mode,
            workers=self.workers,
            replica_rank=self.replica_rank,
            node_rank=0,
            gpus_per_node=self.gpus_per_node,
            nnodes=self.nnodes,
        )
        self.servers.append(server)
        
        # Launch server
        master_address, master_port = await server.get_master_address.remote()
        await server.launch_server.remote(master_address, master_port)
        
        # Get server address
        server_address, server_port = await server.get_server_address.remote()
        self._server_handle = server
        self._server_address = (
            f"[{server_address}]:{server_port}"
            if is_valid_ipv6_address(server_address)
            else f"{server_address}:{server_port}"
        )
        
        logger.info(f"ATOMReplica {self.replica_rank}: Server launched at {self._server_address}")