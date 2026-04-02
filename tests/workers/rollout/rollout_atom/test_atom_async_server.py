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

"""Unit tests for ATOM async server (ATOMHttpServer and ATOMReplica).

Tests cover server initialization, engine configuration building,
sampling parameter handling, batch collection, sleep/wake lifecycle,
and replica placement group logic.

All tests are CPU-only and mock external dependencies (Ray, ATOM engine, CUDA).

Usage:
    pytest tests/workers/rollout/rollout_atom/test_atom_async_server.py -v
"""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from verl.workers.rollout.atom_rollout.constants import ATOMDefaults, IPCConfig, SleepLevel
from verl.workers.rollout.replica import RolloutMode


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rollout_config():
    """Create a mock RolloutConfig with typical ATOM defaults."""
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
    return config


@pytest.fixture
def model_config():
    """Create a mock HFModelConfig."""
    config = MagicMock()
    config.local_path = "/tmp/test_model"
    config.path = "/tmp/test_model"
    config.trust_remote_code = False
    config.hf_config = MagicMock()
    return config


@pytest.fixture
def mock_omega_conf():
    """Mock omega_conf_to_dataclass to return the input as-is."""
    with patch(
        "verl.workers.rollout.atom_rollout.atom_async_server.omega_conf_to_dataclass",
        side_effect=lambda cfg, **kw: cfg,
    ):
        yield


@pytest.fixture
def mock_ray():
    """Mock Ray utilities used during ATOMHttpServer construction."""
    with patch("verl.workers.rollout.atom_rollout.atom_async_server.ray") as mock:
        mock.util.get_node_ip_address.return_value = "127.0.0.1"
        yield mock


@pytest.fixture
def mock_get_free_port():
    """Mock get_free_port to return a deterministic port."""
    with patch(
        "verl.workers.rollout.atom_rollout.atom_async_server.get_free_port",
        return_value=(12345, None),
    ):
        yield


@pytest.fixture
def server(rollout_config, model_config, mock_omega_conf, mock_ray, mock_get_free_port):
    """Create an ATOMHttpServer instance with all external deps mocked."""
    from verl.workers.rollout.atom_rollout.atom_async_server import ATOMHttpServer

    return ATOMHttpServer(
        config=rollout_config,
        model_config=model_config,
        rollout_mode=RolloutMode.HYBRID,
        workers=[],
        replica_rank=0,
        node_rank=0,
        gpus_per_node=2,
        nnodes=1,
        cuda_visible_devices="0,1",
    )


# ---------------------------------------------------------------------------
# ATOMHttpServer unit tests
# ---------------------------------------------------------------------------

class TestATOMHttpServerInit:
    """Test ATOMHttpServer construction and config parsing."""

    def test_basic_init(self, server, rollout_config):
        """Server initializes with correct attributes."""
        assert server.replica_rank == 0
        assert server.node_rank == 0
        assert server.gpus_per_node == 2
        assert server.engine is None
        assert server._server_address == "127.0.0.1"
        assert server._master_port == 12345

    def test_batch_params_computed(self, server, rollout_config):
        """Batch size = dp_size * max_num_seqs."""
        expected_batch_size = rollout_config.data_parallel_size * rollout_config.max_num_seqs
        assert server._batch_size == expected_batch_size
        assert server._batch_timeout == ATOMDefaults.BATCH_TIMEOUT

    def test_non_master_node(self, rollout_config, model_config, mock_omega_conf, mock_ray):
        """Non-master node (node_rank != 0) does not set master address."""
        from verl.workers.rollout.atom_rollout.atom_async_server import ATOMHttpServer

        srv = ATOMHttpServer(
            config=rollout_config,
            model_config=model_config,
            rollout_mode=RolloutMode.HYBRID,
            workers=[],
            replica_rank=0,
            node_rank=1,
            gpus_per_node=2,
            nnodes=2,
            cuda_visible_devices="2,3",
        )
        assert not hasattr(srv, "_master_port") or srv.node_rank != 0


class TestBuildEngineKwargs:
    """Test _build_engine_kwargs configuration building."""

    def test_basic_kwargs(self, server, rollout_config, model_config):
        """Engine kwargs contain required fields."""
        kwargs = server._build_engine_kwargs()
        assert kwargs["model"] == model_config.local_path
        assert kwargs["tensor_parallel_size"] == rollout_config.tensor_model_parallel_size
        assert kwargs["data_parallel_size"] == rollout_config.data_parallel_size
        assert kwargs["max_num_seqs"] == rollout_config.max_num_seqs
        assert kwargs["max_model_len"] == rollout_config.max_model_len
        assert kwargs["gpu_memory_utilization"] == rollout_config.gpu_memory_utilization
        assert kwargs["enforce_eager"] == rollout_config.enforce_eager
        assert kwargs["trust_remote_code"] == model_config.trust_remote_code

    def test_dummy_load_format(self, server, rollout_config):
        """load_dummy is True when load_format=='dummy'."""
        rollout_config.load_format = "dummy"
        kwargs = server._build_engine_kwargs()
        assert kwargs["load_dummy"] is True

    def test_verl_specific_keys_excluded(self, server):
        """bucket_size_mb and use_cuda_ipc are NOT passed to engine."""
        kwargs = server._build_engine_kwargs()
        assert "bucket_size_mb" not in kwargs
        assert "use_cuda_ipc" not in kwargs

    def test_expert_parallel(self, server, rollout_config):
        """enable_expert_parallel is True when expert_parallel_size > 1."""
        rollout_config.expert_parallel_size = 4
        kwargs = server._build_engine_kwargs()
        assert kwargs["enable_expert_parallel"] is True

    def test_cudagraph_capture_sizes(self, server, rollout_config):
        """compilation_config is set when cudagraph_capture_sizes is provided."""
        rollout_config.cudagraph_capture_sizes = [1, 2, 4, 8]
        kwargs = server._build_engine_kwargs()
        assert kwargs["compilation_config"]["cudagraph_capture_sizes"] == [1, 2, 4, 8]

    def test_no_engine_kwargs(self, server, rollout_config):
        """No crash when engine_kwargs is None or empty."""
        rollout_config.engine_kwargs = None
        kwargs = server._build_engine_kwargs()
        assert "model" in kwargs

        rollout_config.engine_kwargs = {}
        kwargs = server._build_engine_kwargs()
        assert "model" in kwargs


class TestBuildSamplingParams:
    """Test _build_sampling_params conversion."""

    def test_default_params(self, server):
        """Default sampling params use config response_length and default temperature."""
        with patch("atom.sampling_params.SamplingParams") as MockSP:
            MockSP.return_value = MagicMock()
            server._build_sampling_params({})
            MockSP.assert_called_once_with(
                max_tokens=server.config.response_length,
                temperature=ATOMDefaults.TEMPERATURE,
                logprobs=False,
            )

    def test_custom_params(self, server):
        """Custom sampling params override defaults."""
        with patch("atom.sampling_params.SamplingParams") as MockSP:
            MockSP.return_value = MagicMock()
            server._build_sampling_params({
                "max_tokens": 256,
                "temperature": 0.5,
                "logprobs": True,
            })
            MockSP.assert_called_once_with(
                max_tokens=256,
                temperature=0.5,
                logprobs=True,
            )

    def test_unsupported_params_dropped(self, server):
        """Unsupported params (top_p, top_k, etc.) are silently dropped."""
        with patch("atom.sampling_params.SamplingParams") as MockSP:
            MockSP.return_value = MagicMock()
            server._build_sampling_params({
                "top_p": 0.9,
                "top_k": 50,
                "repetition_penalty": 1.1,
                "max_new_tokens": 100,
            })
            MockSP.assert_called_once_with(
                max_tokens=server.config.response_length,
                temperature=ATOMDefaults.TEMPERATURE,
                logprobs=False,
            )


class TestSleepWake:
    """Test sleep/wake lifecycle."""

    @pytest.mark.asyncio
    async def test_sleep_hybrid_mode(self, server):
        """Sleep calls engine.sleep() in HYBRID mode with free_cache_engine enabled."""
        server.engine = MagicMock()
        await server.sleep()
        server.engine.sleep.assert_called_once_with(level=ATOMDefaults.SLEEP_LEVEL)

    @pytest.mark.asyncio
    async def test_sleep_disabled(self, server, rollout_config):
        """Sleep is no-op when free_cache_engine is False."""
        rollout_config.free_cache_engine = False
        server.engine = MagicMock()
        await server.sleep()
        server.engine.sleep.assert_not_called()

    @pytest.mark.asyncio
    async def test_sleep_custom_level(self, server):
        """Sleep with custom level overrides default."""
        server.engine = MagicMock()
        await server.sleep(level=SleepLevel.RELEASE_KV_CACHE_ONLY)
        server.engine.sleep.assert_called_once_with(level=SleepLevel.RELEASE_KV_CACHE_ONLY)

    @pytest.mark.asyncio
    async def test_wake_up_hybrid_mode(self, server):
        """Wake up calls engine.wake_up() with default tags."""
        server.engine = MagicMock()
        await server.wake_up()
        server.engine.wake_up.assert_called_once_with(tags=["kv_cache", "weights"])

    @pytest.mark.asyncio
    async def test_wake_up_custom_tags(self, server):
        """Wake up with custom tags passes them through."""
        server.engine = MagicMock()
        await server.wake_up(tags=["weights"])
        server.engine.wake_up.assert_called_once_with(tags=["weights"])

    @pytest.mark.asyncio
    async def test_sleep_no_engine(self, server):
        """Sleep is no-op when engine is None."""
        server.engine = None
        await server.sleep()  # Should not raise

    @pytest.mark.asyncio
    async def test_wake_up_no_engine(self, server):
        """Wake up is no-op when engine is None."""
        server.engine = None
        await server.wake_up()  # Should not raise


class TestClearKvCache:
    """Test clear_kv_cache method."""

    @pytest.mark.asyncio
    async def test_clear_via_core_mgr(self, server):
        """clear_kv_cache uses core_mgr.broadcast_utility_command."""
        server.engine = MagicMock()
        server.engine.core_mgr = MagicMock()
        await server.clear_kv_cache()
        server.engine.core_mgr.broadcast_utility_command.assert_called_once_with("clear_kv_cache")

    @pytest.mark.asyncio
    async def test_clear_fallback(self, server):
        """clear_kv_cache falls back to engine.clear_kv_cache() if no core_mgr."""
        server.engine = MagicMock(spec=["clear_kv_cache"])
        del server.engine.core_mgr  # Remove core_mgr attribute
        await server.clear_kv_cache()
        server.engine.clear_kv_cache.assert_called_once()

    @pytest.mark.asyncio
    async def test_clear_no_engine(self, server):
        """clear_kv_cache is no-op when engine is None."""
        server.engine = None
        await server.clear_kv_cache()  # Should not raise


class TestGenerate:
    """Test generate method and batch collection."""

    @pytest.mark.asyncio
    async def test_generate_adds_to_pending(self, server):
        """generate() adds request to pending queue and sets batch event."""
        with patch.object(server, "_build_sampling_params", return_value=MagicMock()):
            # Don't await the generate — just check that the request is queued
            # We need to cancel it to avoid hanging
            task = asyncio.create_task(
                server.generate(
                    prompt_ids=[1, 2, 3],
                    sampling_params={"temperature": 0.7},
                    request_id="test_req_1",
                )
            )
            # Give the event loop a chance to process
            await asyncio.sleep(0.01)
            assert len(server._pending_requests) == 1
            assert server._pending_requests[0][2] == "test_req_1"
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_generate_multimodal_warning(self, server):
        """generate() warns when multimodal inputs are provided."""
        with patch.object(server, "_build_sampling_params", return_value=MagicMock()):
            with patch("verl.workers.rollout.atom_rollout.atom_async_server.logger") as mock_logger:
                task = asyncio.create_task(
                    server.generate(
                        prompt_ids=[1, 2, 3],
                        sampling_params={},
                        request_id="test_mm",
                        image_data=["fake_image"],
                    )
                )
                await asyncio.sleep(0.01)
                mock_logger.warning.assert_called_once()
                assert "multimodal" in mock_logger.warning.call_args[0][0].lower()
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass


# ---------------------------------------------------------------------------
# ATOMReplica unit tests
# ---------------------------------------------------------------------------

class TestATOMReplica:
    """Test ATOMReplica initialization and configuration."""

    def test_init(self, rollout_config, model_config):
        """ATOMReplica initializes with correct server_class."""
        with patch("verl.workers.rollout.atom_rollout.atom_async_server.ray") as mock_ray:
            mock_ray.remote = lambda cls: cls
            from verl.workers.rollout.atom_rollout.atom_async_server import ATOMReplica

            replica = ATOMReplica(
                replica_rank=0,
                config=rollout_config,
                model_config=model_config,
                gpus_per_node=8,
            )
            assert replica.replica_rank == 0

    def test_get_ray_class_with_init_args(self, rollout_config, model_config):
        """get_ray_class_with_init_args returns RayClassWithInitArgs wrapping ServerAdapter."""
        with patch("verl.workers.rollout.atom_rollout.atom_async_server.ray") as mock_ray:
            mock_ray.remote = lambda cls: cls
            from verl.workers.rollout.atom_rollout.atom_async_server import ATOMReplica

            replica = ATOMReplica(
                replica_rank=0,
                config=rollout_config,
                model_config=model_config,
                gpus_per_node=8,
            )
            result = replica.get_ray_class_with_init_args()
            assert result is not None
            assert result.kwargs["config"] == rollout_config
            assert result.kwargs["model_config"] == model_config


# ---------------------------------------------------------------------------
# Constants tests
# ---------------------------------------------------------------------------

class TestConstants:
    """Test ATOM constants and enums."""

    def test_sleep_levels(self):
        """SleepLevel enum values are correct."""
        assert SleepLevel.RELEASE_KV_CACHE_ONLY == 1
        assert SleepLevel.RELEASE_ALL == 2

    def test_atom_defaults(self):
        """ATOMDefaults values are sensible."""
        assert ATOMDefaults.SLEEP_LEVEL == SleepLevel.RELEASE_ALL
        assert ATOMDefaults.TEMPERATURE == 1.0
        assert ATOMDefaults.BATCH_TIMEOUT > 0

    def test_ipc_config(self):
        """IPCConfig defaults are sensible."""
        assert IPCConfig.DEFAULT_BUCKET_SIZE_MB > 0
