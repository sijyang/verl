import atexit
import getpass
import logging
import os
import weakref
from typing import Any, Dict, Generator, List, Optional

import cloudpickle as pickle
import ray
import torch
import torch.distributed
import zmq
import zmq.asyncio
from filelock import FileLock
from torch.distributed.device_mesh import DeviceMesh

from verl import DataProto
from verl.utils.ray_utils import get_event_loop
from verl.utils.distributed import initialize_global_process_group_ray
from verl.utils.ray_utils import ray_noset_visible_devices
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.utils import get_free_port, is_valid_ipv6_address
from verl.workers.rollout.atom_rollout.constants import ATOMDefaults, SleepLevel

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class ServerAdapter(BaseRollout):
    """ATOM server adapter used in native async mode.

    Unlike vLLM/SGLang/TRT-LLM ServerAdapters which act as lightweight clients
    to a remote server, ATOM's ServerAdapter directly holds the LLMEngine in
    the worker process and communicates with ATOMHttpServer via ZeroMQ RPC.
    This enables efficient direct weight loading without IPC serialization.

    - hybrid mode: holds the ATOM LLMEngine, handles weight sync and generation.
    - standalone/colocated mode: placeholder to occupy the GPU.
    """

    _RPC_METHOD_MAP = {
        "init_worker": "_init_worker",
        "load_model": "_load_model",
        "generate": "_generate",
        "add_request": "_add_request",
        "step": "_step",
        "is_finished": "_is_finished",
    }

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
    ):
        super().__init__(config, model_config, device_mesh)
        self.tokenizer = self.model_config.tokenizer
        self.inference_engine = None  # LLMEngine, initialized via RPC
        self.address = self._init_zeromq()
        
        # Configure sleep level
        # layered_summon requires sleep_level=1 to avoid memory overflow
        if config.layered_summon:
            logger.warning("Setting sleep_level to 1 for layered_summon mode")
            self.sleep_level = SleepLevel.RELEASE_KV_CACHE_ONLY
        else:
            self.sleep_level = ATOMDefaults.SLEEP_LEVEL

    def _init_zeromq(self) -> str:
        """Initialize ZeroMQ socket for RPC communication with server."""
        tensor_parallel_size = self.config.tensor_model_parallel_size
        local_world_size = int(os.environ.get("RAY_LOCAL_WORLD_SIZE", "1"))
        
        # Use IPC for single node, TCP for multi-node
        socket_type = "ipc" if tensor_parallel_size <= local_world_size else "tcp"

        with FileLock(f"/tmp/verl_atom_zmq_{getpass.getuser()}.lock"):
            context = zmq.asyncio.Context()
            self.socket = context.socket(zmq.REP)
            
            if socket_type == "ipc":
                pid = os.getpid()
                address = f"ipc:///tmp/verl_atom_zmq_{pid}_{getpass.getuser()}.ipc"
            else:
                ip = ray.util.get_node_ip_address().strip("[]")
                port, _ = get_free_port(ip)
                if is_valid_ipv6_address(ip):
                    address = f"tcp://[{ip}]:{port}"
                    self.socket.setsockopt(zmq.IPV6, 1)
                else:
                    address = f"tcp://{ip}:{port}"
            
            self.socket.bind(address)

        loop = get_event_loop()
        self.zmq_loop_task = loop.create_task(self._loop_forever())
        
        return address

    async def _loop_forever(self):
        """Main ZMQ message loop - handles RPC calls from server."""
        while True:
            try:
                message = await self.socket.recv()
                method, args, kwargs = pickle.loads(message)
                result = await self._execute_method(method, *args, **kwargs)
                await self.socket.send(pickle.dumps(result))
            except Exception as e:
                logger.exception(f"ATOM ServerAdapter _loop_forever error: {e}")
                await self.socket.send(pickle.dumps(e))
                break

    def _init_worker(self, all_kwargs: List[Dict[str, Any]]):
        """
        Initialize the ATOM LLMEngine.
        
        Args:
            all_kwargs: List containing engine configuration dict
        """
        if not torch.distributed.is_initialized():
            initialize_global_process_group_ray()
        
        rank = int(os.environ.get("RANK", "0"))
        device_name = "GPU"
        local_rank = (
            0
            if not ray_noset_visible_devices()
            else int(ray.get_runtime_context().get_accelerator_ids()[device_name][0])
        )
        
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["RANK"] = str(rank)
        
        engine_kwargs = all_kwargs[0].copy()
        
        logger.info(
            f"ATOM worker init: rank={rank}, local_rank={local_rank}, "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}"
        )
        
        # Create LLMEngine with VerlEngineCore/VerlModelRunner injection
        from atom.model_engine.llm_engine import LLMEngine
        
        self.inference_engine = LLMEngine(**engine_kwargs)
        
        # Register cleanup handler
        weak_self = weakref.ref(self)
        def _cleanup_on_exit():
            worker = weak_self()
            if worker is not None:
                try:
                    worker.shutdown()
                except Exception:
                    pass
        atexit.register(_cleanup_on_exit)
        
        logger.info(f"ATOM LLMEngine initialized on rank={rank}")

    def _load_model(self, *args, **kwargs):
        """Model is loaded during LLMEngine initialization."""
        pass

    async def _execute_method(self, method: str, *args, **kwargs):
        """Execute RPC method called from server.
        
        Uses method mapping for known methods, falls back to inference engine
        for unknown methods (e.g., broadcast_utility_command).
        """
        # Check known method mapping first
        if method in self._RPC_METHOD_MAP:
            handler = getattr(self, self._RPC_METHOD_MAP[method])
            return handler(*args, **kwargs)
        
        # Forward unknown methods to inference engine
        if self.inference_engine is not None:
            # Try inference_engine first, then core_mgr
            for target in (self.inference_engine, self.inference_engine.core_mgr):
                if hasattr(target, method):
                    return getattr(target, method)(*args, **kwargs)
        
        raise ValueError(f"Unknown method: {method}")

    def _generate(self, prompts, sampling_params, request_ids=None):
        """Generate sequences using LLMEngine.
        
        This method uses add_request() + step() loop instead of generate()
        to avoid resetting the round-robin counter, which ensures proper
        distribution of requests across DP ranks.
        """
        if self.inference_engine is None:
            raise RuntimeError("Engine not initialized")
        
        # Original implementation (resets _rr_counter, all requests go to DP rank 0):
        # return self.inference_engine.generate(prompts, sampling_params, request_ids=request_ids)
        
        # New implementation: use add_request() + step() to preserve round-robin counter
        # This is equivalent to generate() but without resetting _rr_counter
        self.inference_engine.add_request(prompts, sampling_params, request_ids=request_ids)
        
        outputs = {}
        while not self.inference_engine.is_finished() and (
            self.inference_engine.core_mgr.is_alive() or self.inference_engine.core_mgr.is_rest()
        ):
            seqs = self.inference_engine.step()
            outs = self.inference_engine.io_processor.postprocess(seqs)
            outputs.update(outs)
        
        # Sort outputs by seq_id to maintain consistent ordering
        outputs = [outputs[seq_id] for seq_id in sorted(outputs)]
        return outputs

    def _add_request(self, prompts, sampling_params, request_ids=None):
        """Add requests to LLMEngine."""
        if self.inference_engine is None:
            raise RuntimeError("Engine not initialized")
        return self.inference_engine.add_request(prompts, sampling_params, request_ids=request_ids)

    def _step(self):
        """Execute one step of generation."""
        if self.inference_engine is None:
            raise RuntimeError("Engine not initialized")
        return self.inference_engine.step()

    def _is_finished(self):
        """Check if all requests are finished."""
        if self.inference_engine is None:
            return True
        return self.inference_engine.is_finished()

    # ==================== BaseRollout interface ====================
    
    def _is_engine_worker(self) -> bool:
        """Check if this worker has the inference engine initialized."""
        return self.inference_engine is not None

    async def resume(self, tags: List[str]):
        """
        Resume rollout weights or kv cache in GPU memory.
        
        Following vLLM pattern: directly call inference_engine.wake_up()
        
        """
        if not self._is_engine_worker():
            return
        if self.config.free_cache_engine:
            self.inference_engine.wake_up(tags=tags)

    async def release(self):
        """
        Release weights and kv cache in GPU memory.
        
        Following vLLM pattern: directly call inference_engine.sleep()
        """
        if not self._is_engine_worker():
            return
        if self.config.free_cache_engine:
            self.inference_engine.sleep(level=self.sleep_level)

    async def update_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        **kwargs
    ):
        """
        Update the weights of the rollout model.
        
        Following vLLM pattern: stream weights directly to model.load_weights()
        This avoids the need for IPC serialization and handles memory more efficiently.
        """
        if not self._is_engine_worker():
            # Consume the generator to avoid issues with caller expecting it to be consumed
            for _ in weights:
                pass
            return
        
        self.inference_engine.load_weights(weights)
        
        logger.info("ATOM weight update completed")

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Not supported in async mode."""
        raise NotImplementedError(
            "ATOM ServerAdapter does not support synchronous generate_sequences(). "
            "Please use the async server interface via ATOMReplica and ATOMHttpServer."
        )


    def get_zeromq_address(self):
        """Return ZMQ address for server to connect."""
        return self.address

    def shutdown(self):
        """Shutdown the engine and cleanup resources."""
        if getattr(self, '_is_shutdown', False):
            return
        self._is_shutdown = True
        
        logger.info("ATOM ServerAdapter shutting down...")
        
        if hasattr(self, 'inference_engine') and self.inference_engine is not None:
            try:
                self.inference_engine.core_mgr.close()
            except Exception as e:
                logger.warning(f"Error shutting down LLMEngine: {e}")
            self.inference_engine = None
        
        if hasattr(self, 'socket') and self.socket is not None:
            try:
                self.socket.close()
            except Exception as e:
                logger.warning(f"Error closing ZMQ socket: {e}")
            self.socket = None
        
        logger.info("ATOM ServerAdapter shutdown complete")

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass
