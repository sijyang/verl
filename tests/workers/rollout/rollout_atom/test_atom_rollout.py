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

"""Unit tests for ATOM ServerAdapter.

Usage:
    pytest tests/workers/rollout/rollout_atom/test_atom_rollout.py -v
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from verl.workers.rollout.atom_rollout.constants import ATOMDefaults, SleepLevel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_rollout_config(**overrides):
    config = MagicMock()
    config.tensor_model_parallel_size = 2
    config.data_parallel_size = 1
    config.pipeline_model_parallel_size = 1
    config.free_cache_engine = True
    config.layered_summon = False
    config.engine_kwargs = {"atom": {"bucket_size_mb": 2048}}
    config.checkpoint_engine = MagicMock()
    config.checkpoint_engine.update_weights_bucket_megabytes = 2048
    for k, v in overrides.items():
        setattr(config, k, v)
    return config


def _make_adapter(env_overrides=None, config_overrides=None):
    """Build a ServerAdapter with all externals mocked."""
    from verl.workers.rollout.atom_rollout.atom_rollout import ServerAdapter

    env = {"RANK": "0", "RAY_LOCAL_WORLD_SIZE": "2"}
    if env_overrides:
        env.update(env_overrides)

    config = _make_rollout_config(**(config_overrides or {}))
    model_config = MagicMock()
    model_config.local_path = "/tmp/test_model"

    mock_ctx = MagicMock()
    mock_ctx.get_job_id.return_value = "test_job_42"

    with patch.dict(os.environ, env), patch(
        "verl.workers.rollout.atom_rollout.atom_rollout.is_support_ipc",
        return_value=True,
    ), patch(
        "verl.workers.rollout.atom_rollout.atom_rollout.ray",
    ) as mock_ray:
        mock_ray.get_runtime_context.return_value = mock_ctx
        adapter = ServerAdapter(
            config=config, model_config=model_config, device_mesh=None
        )
    return adapter


# ---------------------------------------------------------------------------
# Socket naming
# ---------------------------------------------------------------------------


class TestSocketNaming:
    def test_zmq_handle_format(self):
        adapter = _make_adapter()
        assert adapter.zmq_handle == (
            "ipc:///tmp/rl-colocate-zmq-atom-test_job_42-replica-0-rank-0.sock"
        )

    def test_zmq_handle_with_different_ranks(self):
        adapter = _make_adapter(env_overrides={"RANK": "3"})
        assert adapter.replica_rank == 1
        assert adapter.rollout_rank == 1
        assert "replica-1-rank-1" in adapter.zmq_handle


# ---------------------------------------------------------------------------
# IPC / SHM switch
# ---------------------------------------------------------------------------


class TestIPCSHMSwitch:
    def test_use_cuda_ipc_true(self):
        adapter = _make_adapter(
            config_overrides={
                "engine_kwargs": {"atom": {"use_cuda_ipc": True}},
            }
        )
        assert adapter.use_shm is False

    def test_use_cuda_ipc_false(self):
        adapter = _make_adapter(
            config_overrides={
                "engine_kwargs": {"atom": {"use_cuda_ipc": False}},
            }
        )
        assert adapter.use_shm is True

    def test_auto_detect_fallback(self):
        """Without use_cuda_ipc, falls back to is_support_ipc()."""
        adapter = _make_adapter(
            config_overrides={"engine_kwargs": {"atom": {}}}
        )
        assert adapter.use_shm is False


# ---------------------------------------------------------------------------
# Sleep level
# ---------------------------------------------------------------------------


class TestSleepLevel:
    def test_default_sleep_level(self):
        adapter = _make_adapter()
        assert adapter.sleep_level == ATOMDefaults.SLEEP_LEVEL

    def test_layered_summon(self):
        adapter = _make_adapter(config_overrides={"layered_summon": True})
        assert adapter.sleep_level == SleepLevel.RELEASE_KV_CACHE_ONLY


# ---------------------------------------------------------------------------
# Resume / Release delegation
# ---------------------------------------------------------------------------


class TestResumeRelease:
    @pytest.mark.asyncio
    async def test_resume_calls_wake_up(self):
        adapter = _make_adapter()
        mock_handle = MagicMock()
        mock_handle.wake_up.remote = AsyncMock()
        adapter.server_handle = mock_handle
        await adapter.resume(tags=["weights", "kv_cache"])
        mock_handle.wake_up.remote.assert_called_once()

    @pytest.mark.asyncio
    async def test_release_calls_sleep(self):
        adapter = _make_adapter()
        mock_handle = MagicMock()
        mock_handle.sleep.remote = AsyncMock()
        adapter.server_handle = mock_handle
        await adapter.release()
        mock_handle.sleep.remote.assert_called_once()

    @pytest.mark.asyncio
    async def test_non_rank_zero_skips(self):
        adapter = _make_adapter(env_overrides={"RANK": "1"})
        assert adapter.rollout_rank == 1
        mock_handle = MagicMock()
        adapter.server_handle = mock_handle
        await adapter.resume(tags=["weights"])
        await adapter.release()
        mock_handle.wake_up.remote.assert_not_called()
        mock_handle.sleep.remote.assert_not_called()


# ---------------------------------------------------------------------------
# generate_sequences
# ---------------------------------------------------------------------------


class TestGenerateSequences:
    def test_raises_not_implemented(self):
        adapter = _make_adapter()
        with pytest.raises(NotImplementedError, match="does not support synchronous"):
            adapter.generate_sequences(MagicMock())
