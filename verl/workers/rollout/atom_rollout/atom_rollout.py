import logging
import os
import time
from typing import Generator, List, Optional

import ray
import torch
import torch.distributed
from torch.distributed.device_mesh import DeviceMesh

from verl import DataProto
from verl.utils.device import is_support_ipc
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.atom_rollout.bucketed_weight_transfer import BucketedWeightSender
from verl.workers.rollout.atom_rollout.constants import ATOMDefaults, SleepLevel
from verl.workers.rollout.base import BaseRollout

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class ServerAdapter(BaseRollout):

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
        replica_rank: int = -1,
    ):
        super().__init__(config, model_config, device_mesh)
        self.tokenizer = self.model_config.tokenizer

        rank = int(os.environ.get("RANK", "0"))
        local_world_size = int(os.environ.get("RAY_LOCAL_WORLD_SIZE", "1"))
        rollout_world_size = (
            self.config.tensor_model_parallel_size
            * self.config.data_parallel_size
            * getattr(self.config, "pipeline_model_parallel_size", 1)
        )
        if replica_rank == -1:
            self.replica_rank = rank // rollout_world_size
        else:
            self.replica_rank = replica_rank
        self.rollout_rank = rank % rollout_world_size
        self.node_rank = self.rollout_rank // local_world_size

        # ── Sleep level ──
        if config.layered_summon:
            logger.warning("Setting sleep_level to 1 for layered_summon mode")
            self.sleep_level = SleepLevel.RELEASE_KV_CACHE_ONLY
        else:
            self.sleep_level = ATOMDefaults.SLEEP_LEVEL

        # ── Weight transfer (ZMQ IPC / SHM) ──
        local_rank = self.rollout_rank % local_world_size
        job_id = ray.get_runtime_context().get_job_id()
        self.zmq_handle = f"ipc:///tmp/rl-colocate-zmq-atom-{job_id}-replica-{self.replica_rank}-rank-{local_rank}.sock"

        ipc_path = self.zmq_handle[len("ipc://"):]
        try:
            os.remove(ipc_path)
        except OSError:
            pass

        atom_kwargs = (getattr(self.config, "engine_kwargs", {}) or {}).get("atom", {}) or {}
        use_cuda_ipc = atom_kwargs.get("use_cuda_ipc", None)
        if use_cuda_ipc is not None:
            self.use_shm = not use_cuda_ipc
        else:
            self.use_shm = not is_support_ipc()

        # ── Server actor handle (lazy) ──
        self.server_handle: Optional[ray.actor.ActorHandle] = None

    def _get_server_name_prefix(self) -> str:
        return "atom_server"

    def _get_server_handle(self) -> ray.actor.ActorHandle:
        """Lazy-init ATOMHttpServer Ray actor handle."""
        if self.server_handle is None:
            prefix = self._get_server_name_prefix()
            self.server_handle = ray.get_actor(
                f"{prefix}_{self.replica_rank}_{self.node_rank}"
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
            logger.info(f"update_weights done, cost: {time.time() - start_time:.2f}s")

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        raise NotImplementedError(
            "ATOM ServerAdapter does not support synchronous generate_sequences(). "
            "Use the async server interface via ATOMReplica and ATOMHttpServer."
        )
