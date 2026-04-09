import atexit
import getpass
import logging
import os
import time
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
from verl.utils.device import get_device_id, is_support_ipc
from verl.utils.distributed import initialize_global_process_group_ray
from verl.utils.net_utils import get_free_port, is_valid_ipv6_address
from verl.utils.ray_utils import get_event_loop, ray_noset_visible_devices
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.atom_rollout.constants import ATOMDefaults, SleepLevel
from verl.workers.rollout.atom_rollout.utils import get_device_uuid
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightSender

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class ServerAdapter(BaseRollout):
    # ZMQ RPC dispatch table: method name → handler attribute
    _RPC_METHOD_MAP = {
        "init_worker": "_rpc_init_worker",
        "load_model": "_rpc_load_model",
        "generate": "_rpc_generate",
        "add_request": "_rpc_add_request",
        "step": "_rpc_step",
        "is_finished": "_rpc_is_finished",
    }

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
    ):
        super().__init__(config, model_config, device_mesh)
        self.tokenizer = self.model_config.tokenizer

        # ── Rank computation (same as vLLM ServerAdapter) ──
        rank = int(os.environ.get("RANK", "0"))
        local_world_size = int(os.environ.get("RAY_LOCAL_WORLD_SIZE", "1"))
        rollout_world_size = (
            self.config.tensor_model_parallel_size
            * self.config.data_parallel_size
            * getattr(self.config, "pipeline_model_parallel_size", 1)
        )
        self.replica_rank = rank // rollout_world_size
        self.rollout_rank = rank % rollout_world_size
        self.node_rank = self.rollout_rank // local_world_size

        # ── Sleep level ──
        if config.layered_summon:
            logger.warning("Setting sleep_level to 1 for layered_summon mode")
            self.sleep_level = SleepLevel.RELEASE_KV_CACHE_ONLY
        else:
            self.sleep_level = ATOMDefaults.SLEEP_LEVEL

        # ── Weight transfer (ZMQ IPC / SHM) ──
        self.device_uuid = get_device_uuid(get_device_id())
        self.zmq_handle = f"ipc:///tmp/rl-colocate-zmq-atom-{self.device_uuid}.sock"
        self.use_shm = not is_support_ipc()

        # ── Server actor handle (lazy) ──
        self.server_handle: Optional[ray.actor.ActorHandle] = None

        # ── ZMQ RPC worker (for hybrid mode callbacks from ATOMHttpServer) ──
        self.inference_engine = None
        self.address = self._init_rpc_socket()

    def _get_server_handle(self) -> ray.actor.ActorHandle:
        """Lazy-init ATOMHttpServer Ray actor handle."""
        if self.server_handle is None:
            self.server_handle = ray.get_actor(
                f"atom_server_{self.replica_rank}_{self.node_rank}"
            )
        return self.server_handle

    async def resume(self, tags: List[str]):
        """Resume rollout weights or kv cache in GPU memory."""
        if not self.config.free_cache_engine or self.rollout_rank != 0:
            return
        await self._get_server_handle().wake_up.remote(tags=tags)

    async def release(self):
        """Release weights and kv cache in GPU memory."""
        if not self.config.free_cache_engine or self.rollout_rank != 0:
            return
        await self._get_server_handle().sleep.remote(level=self.sleep_level)

    @torch.no_grad()
    async def update_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        global_steps: int = None,
        **kwargs,
    ):
        """Send updated weights to ATOMHttpServer via ZMQ."""
        if self.rollout_rank != 0:
            for _ in weights:
                pass
            return

        start_time = time.time()

        # Materialize to compute minimum bucket size for the largest tensor
        weight_list = list(weights)
        bucket_size_mb = self.config.checkpoint_engine.update_weights_bucket_megabytes
        if weight_list:
            max_bytes = max(w.nbytes for _, w in weight_list)
            min_mb = (max_bytes >> 20) + 1
            if min_mb > bucket_size_mb:
                bucket_size_mb = min_mb
                logger.info(
                    f"Auto-increased bucket_size_mb to {bucket_size_mb} "
                    f"for largest weight"
                )

        server = self._get_server_handle()
        future = server.update_weights_from_zmq.remote(use_shm=self.use_shm)

        sender = BucketedWeightSender(
            zmq_handle=self.zmq_handle,
            bucket_size_mb=bucket_size_mb,
            use_shm=self.use_shm,
        )
        await sender.async_send_weights(iter(weight_list))
        await future

        if self.replica_rank == 0 and self.rollout_rank == 0:
            logger.info(
                f"update_weights done, cost: {time.time() - start_time:.2f}s"
            )

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        raise NotImplementedError(
            "ATOM ServerAdapter does not support synchronous generate_sequences(). "
            "Use the async server interface via ATOMReplica and ATOMHttpServer."
        )

    def get_zeromq_address(self) -> str:
        """Return the ZMQ address for ATOMHttpServer to connect to."""
        return self.address

    def _init_rpc_socket(self) -> str:
        """Bind a ZMQ REP socket for RPC from ATOMHttpServer."""
        tp_size = self.config.tensor_model_parallel_size
        local_world_size = int(os.environ.get("RAY_LOCAL_WORLD_SIZE", "1"))
        use_ipc = tp_size <= local_world_size

        with FileLock(f"/tmp/verl_atom_zmq_{getpass.getuser()}.lock"):
            ctx = zmq.asyncio.Context()
            self.socket = ctx.socket(zmq.REP)

            if use_ipc:
                address = (
                    f"ipc:///tmp/verl_atom_zmq_{os.getpid()}"
                    f"_{getpass.getuser()}.ipc"
                )
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
        self._rpc_loop_task = loop.create_task(self._rpc_loop())
        return address

    async def _rpc_loop(self):
        """Main ZMQ message loop — dispatches RPC calls from ATOMHttpServer."""
        while True:
            try:
                msg = await self.socket.recv()
                method, args, kwargs = pickle.loads(msg)
                result = await self._rpc_dispatch(method, *args, **kwargs)
                await self.socket.send(pickle.dumps(result))
            except Exception as e:
                logger.exception(f"RPC loop error: {e}")
                await self.socket.send(pickle.dumps(e))
                break

    async def _rpc_dispatch(self, method: str, *args, **kwargs):
        """Route an RPC method to its handler."""
        if method in self._RPC_METHOD_MAP:
            handler = getattr(self, self._RPC_METHOD_MAP[method])
            return handler(*args, **kwargs)

        # Forward unknown methods to inference engine
        if self.inference_engine is not None:
            for target in (self.inference_engine, self.inference_engine.core_mgr):
                if hasattr(target, method):
                    return getattr(target, method)(*args, **kwargs)

        raise ValueError(f"Unknown RPC method: {method}")

    # ── RPC handlers ──

    def _rpc_init_worker(self, all_kwargs: List[Dict[str, Any]]):
        """Initialize the ATOM LLMEngine in this worker process."""
        if not torch.distributed.is_initialized():
            initialize_global_process_group_ray()

        rank = int(os.environ.get("RANK", "0"))
        local_rank = (
            0
            if not ray_noset_visible_devices()
            else int(
                ray.get_runtime_context().get_accelerator_ids()["GPU"][0]
            )
        )
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["RANK"] = str(rank)

        from atom.model_engine.llm_engine import LLMEngine

        self.inference_engine = LLMEngine(**all_kwargs[0])

        weak_self = weakref.ref(self)

        def _cleanup():
            obj = weak_self()
            if obj is not None:
                try:
                    obj.shutdown()
                except Exception:
                    pass

        atexit.register(_cleanup)
        logger.info(f"ATOM LLMEngine initialized on rank={rank}")

    def _rpc_load_model(self, *args, **kwargs):
        pass  # Model is loaded during LLMEngine init

    def _rpc_generate(self, prompts, sampling_params, request_ids=None):
        """Generate via add_request + step loop (preserves DP round-robin)."""
        if self.inference_engine is None:
            raise RuntimeError("Engine not initialized")

        self.inference_engine.add_request(
            prompts, sampling_params, request_ids=request_ids
        )
        outputs = {}
        while not self.inference_engine.is_finished() and (
            self.inference_engine.core_mgr.is_alive()
            or self.inference_engine.core_mgr.is_rest()
        ):
            seqs = self.inference_engine.step()
            outputs.update(self.inference_engine.io_processor.postprocess(seqs))

        return [outputs[sid] for sid in sorted(outputs)]

    def _rpc_add_request(self, prompts, sampling_params, request_ids=None):
        if self.inference_engine is None:
            raise RuntimeError("Engine not initialized")
        return self.inference_engine.add_request(
            prompts, sampling_params, request_ids=request_ids
        )

    def _rpc_step(self):
        if self.inference_engine is None:
            raise RuntimeError("Engine not initialized")
        return self.inference_engine.step()

    def _rpc_is_finished(self):
        if self.inference_engine is None:
            return True
        return self.inference_engine.is_finished()

    def shutdown(self):
        """Shutdown the engine and clean up resources."""
        if getattr(self, "_is_shutdown", False):
            return
        self._is_shutdown = True
        logger.info("ATOM ServerAdapter shutting down...")

        if getattr(self, "inference_engine", None) is not None:
            try:
                self.inference_engine.core_mgr.close()
            except Exception as e:
                logger.warning(f"Error shutting down LLMEngine: {e}")
            self.inference_engine = None

        if getattr(self, "socket", None) is not None:
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
