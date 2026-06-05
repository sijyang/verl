# Copyright 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for ATOMHttpServer and ATOMReplica.

Usage:
    pytest tests/workers/rollout/rollout_atom/test_atom_async_server.py -v
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from verl.workers.rollout.atom_rollout.constants import ATOMDefaults
from verl.workers.rollout.replica import RolloutMode


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_server(*, node_rank=0, replica_rank=0, rollout_config_overrides=None):
    """Create an ATOMHttpServer with all external deps mocked."""
    from verl.workers.rollout.atom_rollout.atom_async_server import ATOMHttpServer

    config = MagicMock()
    config.tensor_model_parallel_size = 2
    config.data_parallel_size = 1
    config.expert_parallel_size = 1
    config.pipeline_model_parallel_size = 1
    config.max_num_seqs = 32
    config.max_model_len = 2048
    config.gpu_memory_utilization = 0.7
    config.enforce_eager = True
    config.load_format = "auto"
    config.free_cache_engine = True
    config.response_length = 512
    config.cudagraph_capture_sizes = None
    config.engine_kwargs = {"atom": {"bucket_size_mb": 2048, "use_cuda_ipc": True}}
    config.layered_summon = False
    if rollout_config_overrides:
        for k, v in rollout_config_overrides.items():
            setattr(config, k, v)

    model_config = MagicMock()
    model_config.local_path = "/tmp/test_model"
    model_config.trust_remote_code = False
    model_config.hf_config = MagicMock()

    with patch(
        "verl.workers.rollout.atom_rollout.atom_async_server.omega_conf_to_dataclass",
        side_effect=lambda cfg, **kw: cfg,
    ), patch(
        "verl.workers.rollout.atom_rollout.atom_async_server.ray"
    ) as mock_ray, patch(
        "verl.workers.rollout.atom_rollout.atom_async_server.get_free_port",
        return_value=(12345, None),
    ):
        mock_ray.util.get_node_ip_address.return_value = "127.0.0.1"
        mock_ctx = MagicMock()
        mock_ctx.get_job_id.return_value = "test_job_42"
        mock_ray.get_runtime_context.return_value = mock_ctx

        server = ATOMHttpServer(
            config=config,
            model_config=model_config,
            rollout_mode=RolloutMode.HYBRID,
            workers=[],
            replica_rank=replica_rank,
            node_rank=node_rank,
            gpus_per_node=2,
            nnodes=1,
            cuda_visible_devices="0,1",
        )
    return server


# ---------------------------------------------------------------------------
# Init & config
# ---------------------------------------------------------------------------


class TestATOMHttpServerInit:
    def test_basic_init(self):
        server = _make_server()
        assert server.replica_rank == 0
        assert server.node_rank == 0
        assert server.engine is None
        assert server._master_port == 12345
        assert server.job_id == "test_job_42"

    def test_batch_params(self):
        server = _make_server()
        assert server._batch_size == 1 * 32  # dp_size * max_num_seqs
        assert server._batch_timeout == ATOMDefaults.BATCH_TIMEOUT

    def test_non_master_node(self):
        server = _make_server(node_rank=1)
        assert server.node_rank == 1
        assert not hasattr(server, "_master_port")


# ---------------------------------------------------------------------------
# Engine kwargs
# ---------------------------------------------------------------------------


class TestBuildEngineKwargs:
    def test_basic_kwargs(self):
        server = _make_server()
        kwargs = server._build_engine_kwargs()
        assert kwargs["model"] == "/tmp/test_model"
        assert kwargs["tensor_parallel_size"] == 2
        assert kwargs["enforce_eager"] is True

    def test_verl_keys_excluded(self):
        server = _make_server()
        kwargs = server._build_engine_kwargs()
        assert "bucket_size_mb" not in kwargs
        assert "use_cuda_ipc" not in kwargs

    def test_cudagraph_capture_sizes(self):
        server = _make_server(
            rollout_config_overrides={"cudagraph_capture_sizes": [1, 2, 4]}
        )
        kwargs = server._build_engine_kwargs()
        assert kwargs["compilation_config"]["cudagraph_capture_sizes"] == [1, 2, 4]


# ---------------------------------------------------------------------------
# Sleep / Wake
# ---------------------------------------------------------------------------


class TestSleepWake:
    @pytest.mark.asyncio
    async def test_sleep(self):
        server = _make_server()
        server.engine = MagicMock()
        await server.sleep()
        server.engine.sleep.assert_called_once_with(level=ATOMDefaults.SLEEP_LEVEL)

    @pytest.mark.asyncio
    async def test_sleep_disabled(self):
        server = _make_server(rollout_config_overrides={"free_cache_engine": False})
        server.engine = MagicMock()
        await server.sleep()
        server.engine.sleep.assert_not_called()

    @pytest.mark.asyncio
    async def test_wake_up(self):
        server = _make_server()
        server.engine = MagicMock()
        await server.wake_up()
        server.engine.wake_up.assert_called_once_with(tags=["kv_cache", "weights"])

    @pytest.mark.asyncio
    async def test_wake_up_custom_tags(self):
        server = _make_server()
        server.engine = MagicMock()
        await server.wake_up(tags=["weights"])
        server.engine.wake_up.assert_called_once_with(tags=["weights"])


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------


class TestGenerate:
    @pytest.mark.asyncio
    async def test_generate_queues_request(self):
        server = _make_server()
        with patch.object(server, "_build_sampling_params", return_value=MagicMock()):
            task = asyncio.create_task(
                server.generate(
                    prompt_ids=[1, 2, 3],
                    sampling_params={},
                    request_id="req_1",
                )
            )
            await asyncio.sleep(0.01)
            assert len(server._pending_requests) == 1
            assert server._pending_requests[0][2] == "req_1"
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


# ---------------------------------------------------------------------------
# ATOMReplica
# ---------------------------------------------------------------------------


class TestATOMReplica:
    def test_get_ray_class_with_init_args(self):
        with patch(
            "verl.workers.rollout.atom_rollout.atom_async_server.ray"
        ) as mock_ray, patch(
            "verl.workers.rollout.replica.omega_conf_to_dataclass",
            side_effect=lambda cfg, **kw: cfg,
        ):
            mock_ray.remote = lambda cls: cls
            from verl.workers.rollout.atom_rollout.atom_async_server import ATOMReplica

            config = MagicMock()
            config.tensor_model_parallel_size = 2
            config.data_parallel_size = 1
            config.pipeline_model_parallel_size = 1
            model_config = MagicMock()
            replica = ATOMReplica(
                replica_rank=0,
                config=config,
                model_config=model_config,
                gpus_per_node=8,
            )
            result = replica.get_ray_class_with_init_args()
            assert result is not None
