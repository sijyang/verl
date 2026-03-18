import gc
import logging
import os
from multiprocessing import shared_memory
from typing import Generator
from uuid import uuid4

import ray
import torch
import zmq
from torch.distributed.device_mesh import DeviceMesh
from torch.multiprocessing.reductions import reduce_tensor

from verl import DataProto
from verl.utils.device import get_device_id, is_support_ipc
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.utils import ensure_async_iterator
from verl.workers.rollout.atom_rollout.constants import ATOMDefaults, IPCConfig, SleepLevel
from verl.workers.rollout.atom_rollout.utils import get_device_uuid

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class ServerAdapter(BaseRollout):

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
    ):
        super().__init__(config, model_config, device_mesh)
        self.server_handle = None  # lazy, via ray.get_actor()

        rank = int(os.environ["RANK"])
        local_world_size = int(os.environ["RAY_LOCAL_WORLD_SIZE"])
        rollout_world_size = (
            config.tensor_model_parallel_size
            * config.data_parallel_size
            * config.pipeline_model_parallel_size
        )
        self.replica_rank = rank // rollout_world_size
        self.rollout_rank = rank % rollout_world_size
        self.node_rank = self.rollout_rank // local_world_size

        if config.layered_summon:
            logger.warning("Setting sleep_level to 1 for layered_summon mode")
            self.sleep_level = SleepLevel.RELEASE_KV_CACHE_ONLY
        else:
            self.sleep_level = ATOMDefaults.SLEEP_LEVEL

        self.device_uuid = get_device_uuid(get_device_id())
        self.zmq_context = zmq.Context()
        self.zmq_handle = f"ipc:///tmp/rl-colocate-zmq-atom-{self.device_uuid}.sock"

        self.use_shm = not is_support_ipc()

    async def _execute_method(self, method, non_block=False, args=(), kwargs=None):
        if self.rollout_rank != 0:
            return None

        if self.server_handle is None:
            self.server_handle = ray.get_actor(
                f"atom_server_{self.replica_rank}_{self.node_rank}"
            )

        future = getattr(self.server_handle, method).remote(
            *(args or ()), **(kwargs or {})
        )
        return future if non_block else await future

    async def resume(self, tags: list[str]):
        """Resume rollout weights or kv cache in GPU memory."""
        if self.config.free_cache_engine:
            await self._execute_method("wake_up", kwargs={"tags": tags})

    async def release(self):
        """Release weights and kv cache in GPU memory."""
        if self.config.free_cache_engine:
            await self._execute_method("sleep", kwargs={"level": self.sleep_level})

    @torch.no_grad()
    async def update_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        **kwargs,
    ):
        if self.rollout_rank != 0:
            # Non rank-0 workers must drain the generator to avoid blocking
            async for _ in ensure_async_iterator(weights):
                pass
            return

        # Phase 1: Non-blocking trigger of server-side ZMQ receive
        future = await self._execute_method(
            "update_weights_from_zmq",
            non_block=True,
            kwargs={**kwargs, "use_shm": self.use_shm},
        )

        # Phase 2: Bind ZMQ REQ socket, allocate GPU buffer, send IPC handle
        atom_kwargs = (getattr(self.config, "engine_kwargs", {}) or {}).get("atom", {}) or {}
        bucket_size_mb = atom_kwargs.get("bucket_size_mb", IPCConfig.DEFAULT_BUCKET_SIZE_MB)
        bucket_size = int(bucket_size_mb) << 20
        s = self.zmq_context.socket(zmq.REQ)
        s.bind(self.zmq_handle)

        buffer, shm = None, None
        if not self.use_shm:
            buffer = torch.empty(bucket_size, dtype=torch.uint8, device=f"cuda:{get_device_id()}")
            handle = reduce_tensor(buffer)
            s.send_pyobj(handle)
        else:
            shm_name = f"verl_atom_{uuid4().hex}"
            shm = shared_memory.SharedMemory(name=shm_name, create=True, size=bucket_size)
            buffer = torch.frombuffer(shm.buf, dtype=torch.uint8)
            s.send_pyobj({"name": shm_name, "size": bucket_size})
        s.recv()  # Wait for server ACK

        # Phase 3: Stream weight buckets
        offset = 0
        bucket_meta = {}
        async for name, weight in ensure_async_iterator(weights):
            nbytes = weight.nbytes
            # Flush current bucket if it would overflow
            if offset + nbytes > bucket_size and bucket_meta:
                torch.cuda.synchronize()
                s.send_pyobj({"bucket_meta": bucket_meta, "is_last": False})
                s.recv()
                bucket_meta = {}
                offset = 0

            assert offset + nbytes <= bucket_size, (
                f"Weight {name}({weight.shape}, {weight.dtype}) is too large to fit in the bucket. "
                f"Please increase engine_kwargs.atom.bucket_size_mb ({bucket_size_mb} MB)."
            )
            bucket_meta[name] = {
                "shape": weight.shape,
                "dtype": weight.dtype,
                "offset": offset,
            }
            buffer[offset:offset + nbytes].copy_(
                weight.view(-1).view(torch.uint8), non_blocking=True
            )
            offset += nbytes

        # Send the last bucket
        torch.cuda.synchronize()
        s.send_pyobj({"bucket_meta": bucket_meta, "is_last": True})
        s.recv()

        # Phase 4: Cleanup
        s.close()
        del buffer
        if shm is not None:
            shm.close()
            shm.unlink()
            del shm
        gc.collect()
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

        if future is not None:
            await future

        # Clear prefix cache after weight update
        await self._execute_method("clear_kv_cache")

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Not supported in async mode."""
        raise NotImplementedError(
            "ATOM ServerAdapter does not support synchronous generate_sequences(). "
            "Please use the async server interface via ATOMReplica and ATOMHttpServer."
        )
