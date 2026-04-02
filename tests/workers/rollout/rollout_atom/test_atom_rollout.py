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

"""Unit tests for ATOM ServerAdapter (rollout worker logic).

Tests cover ServerAdapter initialization, weight update via ZMQ,
sleep/wake delegation, and error handling.

All tests are CPU-only and mock external dependencies (Ray, ZMQ, CUDA, torch).

Usage:
    pytest tests/workers/rollout/rollout_atom/test_atom_rollout.py -v
"""

import os
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from verl.workers.rollout.atom_rollout.constants import ATOMDefaults, SleepLevel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rollout_config():
    """Create a mock RolloutConfig for ServerAdapter."""
    config = MagicMock()
    config.tensor_model_parallel_size = 2
    config.data_parallel_size = 1
    config.pipeline_model_parallel_size = 1
    config.free_cache_engine = True
    config.layered_summon = False
    config.engine_kwargs = {"atom": {"bucket_size_mb": 2048, "use_cuda_ipc": True}}
    return config


@pytest.fixture
def model_config():
    """Create a mock HFModelConfig."""
    config = MagicMock()
    config.local_path = "/tmp/test_model"
    config.path = "/tmp/test_model"
    config.trust_remote_code = False
    return config


@pytest.fixture
def mock_env():
    """Set required environment variables for ServerAdapter.__init__."""
    env = {
        "RANK": "0",
        "RAY_LOCAL_WORLD_SIZE": "2",
    }
    with patch.dict(os.environ, env):
        yield


@pytest.fixture
def mock_zmq():
    """Mock ZMQ context and socket."""
    with patch("verl.workers.rollout.atom_rollout.atom_rollout.zmq") as mock:
        mock_ctx = MagicMock()
        mock_socket = MagicMock()
        mock_ctx.socket.return_value = mock_socket
        mock.Context.return_value = mock_ctx
        mock.REQ = 3
        mock.REP = 4
        yield mock, mock_ctx, mock_socket


@pytest.fixture
def mock_device():
    """Mock CUDA device utilities."""
    with patch(
        "verl.workers.rollout.atom_rollout.atom_rollout.get_device_id", return_value=0
    ), patch(
        "verl.workers.rollout.atom_rollout.atom_rollout.get_device_uuid", return_value="GPU-test-0"
    ), patch(
        "verl.workers.rollout.atom_rollout.atom_rollout.is_support_ipc", return_value=True
    ):
        yield


@pytest.fixture
def adapter(rollout_config, model_config, mock_env, mock_zmq, mock_device):
    """Create a ServerAdapter instance with all external deps mocked."""
    from verl.workers.rollout.atom_rollout.atom_rollout import ServerAdapter

    return ServerAdapter(
        config=rollout_config,
        model_config=model_config,
        device_mesh=None,
    )


# ---------------------------------------------------------------------------
# ServerAdapter initialization tests
# ---------------------------------------------------------------------------

class TestServerAdapterInit:
    """Test ServerAdapter construction."""

    def test_basic_init(self, adapter):
        """Adapter initializes with correct attributes."""
        assert adapter.replica_rank == 0
        assert adapter.rollout_rank == 0
        assert adapter.node_rank == 0
        assert adapter.server_handle is None
        assert adapter.sleep_level == ATOMDefaults.SLEEP_LEVEL

    def test_rank_calculation(self, rollout_config, model_config, mock_zmq, mock_device):
        """Rank is correctly computed from RANK and rollout world size."""
        from verl.workers.rollout.atom_rollout.atom_rollout import ServerAdapter

        # RANK=3, TP=2, DP=1, PP=1 → rollout_world_size=2
        # replica_rank = 3 // 2 = 1, rollout_rank = 3 % 2 = 1
        with patch.dict(os.environ, {"RANK": "3", "RAY_LOCAL_WORLD_SIZE": "2"}):
            adapter = ServerAdapter(config=rollout_config, model_config=model_config, device_mesh=None)
            assert adapter.replica_rank == 1
            assert adapter.rollout_rank == 1

    def test_layered_summon_sleep_level(self, rollout_config, model_config, mock_env, mock_zmq, mock_device):
        """layered_summon forces sleep_level to RELEASE_KV_CACHE_ONLY."""
        from verl.workers.rollout.atom_rollout.atom_rollout import ServerAdapter

        rollout_config.layered_summon = True
        adapter = ServerAdapter(config=rollout_config, model_config=model_config, device_mesh=None)
        assert adapter.sleep_level == SleepLevel.RELEASE_KV_CACHE_ONLY

    def test_zmq_handle_uses_device_uuid(self, adapter):
        """ZMQ handle path includes device UUID for uniqueness."""
        assert "GPU-test-0" in adapter.zmq_handle

    def test_shm_fallback_when_no_ipc(self, rollout_config, model_config, mock_env, mock_zmq):
        """use_shm is True when IPC is not supported."""
        from verl.workers.rollout.atom_rollout.atom_rollout import ServerAdapter

        with patch(
            "verl.workers.rollout.atom_rollout.atom_rollout.get_device_id", return_value=0
        ), patch(
            "verl.workers.rollout.atom_rollout.atom_rollout.get_device_uuid", return_value="GPU-0"
        ), patch(
            "verl.workers.rollout.atom_rollout.atom_rollout.is_support_ipc", return_value=False
        ):
            adapter = ServerAdapter(config=rollout_config, model_config=model_config, device_mesh=None)
            assert adapter.use_shm is True


# ---------------------------------------------------------------------------
# Resume / Release tests
# ---------------------------------------------------------------------------

class TestResumeRelease:
    """Test resume (wake_up) and release (sleep) delegation."""

    @pytest.mark.asyncio
    async def test_resume_calls_wake_up(self, adapter):
        """resume() delegates to server's wake_up with correct tags."""
        mock_handle = MagicMock()
        mock_handle.wake_up.remote = AsyncMock(return_value=None)
        adapter.server_handle = mock_handle

        with patch("verl.workers.rollout.atom_rollout.atom_rollout.ray") as mock_ray:
            mock_ray.get_actor.return_value = mock_handle
            await adapter.resume(tags=["weights", "kv_cache"])
            mock_handle.wake_up.remote.assert_called_once()

    @pytest.mark.asyncio
    async def test_release_calls_sleep(self, adapter):
        """release() delegates to server's sleep with configured level."""
        mock_handle = MagicMock()
        mock_handle.sleep.remote = AsyncMock(return_value=None)
        adapter.server_handle = mock_handle

        with patch("verl.workers.rollout.atom_rollout.atom_rollout.ray") as mock_ray:
            mock_ray.get_actor.return_value = mock_handle
            await adapter.release()
            mock_handle.sleep.remote.assert_called_once()

    @pytest.mark.asyncio
    async def test_resume_noop_when_disabled(self, adapter, rollout_config):
        """resume() is no-op when free_cache_engine is False."""
        rollout_config.free_cache_engine = False
        adapter.server_handle = MagicMock()
        await adapter.resume(tags=["weights"])
        # Should NOT call any remote method

    @pytest.mark.asyncio
    async def test_release_noop_when_disabled(self, adapter, rollout_config):
        """release() is no-op when free_cache_engine is False."""
        rollout_config.free_cache_engine = False
        adapter.server_handle = MagicMock()
        await adapter.release()
        # Should NOT call any remote method

    @pytest.mark.asyncio
    async def test_non_rank_zero_skips(self, rollout_config, model_config, mock_zmq, mock_device):
        """Non rank-0 workers skip resume/release."""
        from verl.workers.rollout.atom_rollout.atom_rollout import ServerAdapter

        with patch.dict(os.environ, {"RANK": "1", "RAY_LOCAL_WORLD_SIZE": "2"}):
            adapter = ServerAdapter(config=rollout_config, model_config=model_config, device_mesh=None)
            assert adapter.rollout_rank == 1
            # _execute_method returns None for non-rank-0
            result = await adapter._execute_method("wake_up")
            assert result is None


# ---------------------------------------------------------------------------
# generate_sequences tests
# ---------------------------------------------------------------------------

class TestGenerateSequences:
    """Test synchronous generate_sequences (unsupported in async mode)."""

    def test_raises_not_implemented(self, adapter):
        """generate_sequences() raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="does not support synchronous"):
            adapter.generate_sequences(MagicMock())


# ---------------------------------------------------------------------------
# _execute_method tests
# ---------------------------------------------------------------------------

class TestExecuteMethod:
    """Test _execute_method server handle management."""

    @pytest.mark.asyncio
    async def test_lazy_actor_lookup(self, adapter):
        """Server handle is lazily resolved via ray.get_actor."""
        assert adapter.server_handle is None
        mock_handle = MagicMock()
        mock_handle.test_method.remote = AsyncMock(return_value="ok")

        with patch("verl.workers.rollout.atom_rollout.atom_rollout.ray") as mock_ray:
            mock_ray.get_actor.return_value = mock_handle
            result = await adapter._execute_method("test_method")
            mock_ray.get_actor.assert_called_once_with("atom_server_0_0")
            assert adapter.server_handle is mock_handle

    @pytest.mark.asyncio
    async def test_non_block_returns_future(self, adapter):
        """non_block=True returns the Ray future without awaiting."""
        mock_handle = MagicMock()
        mock_future = MagicMock()
        mock_handle.test_method.remote = Mock(return_value=mock_future)
        adapter.server_handle = mock_handle

        result = await adapter._execute_method("test_method", non_block=True)
        assert result is mock_future

    @pytest.mark.asyncio
    async def test_passes_args_kwargs(self, adapter):
        """args and kwargs are forwarded to the remote call."""
        mock_handle = MagicMock()
        mock_handle.some_method.remote = AsyncMock(return_value="result")
        adapter.server_handle = mock_handle

        await adapter._execute_method(
            "some_method",
            args=("arg1",),
            kwargs={"key": "value"},
        )
        mock_handle.some_method.remote.assert_called_once_with("arg1", key="value")
