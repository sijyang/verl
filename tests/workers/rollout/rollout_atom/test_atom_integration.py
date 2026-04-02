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

"""Integration tests for ATOM rollout (requires GPU + Ray).

These tests create real ATOMReplica instances and exercise generate,
sleep/wake, and weight update flows end-to-end.

Environment variables:
    ATOM_TEST_MODEL_PATH: Path to a small HF model (default: ~/models/Qwen/Qwen2.5-0.5B-Instruct)
    ATOM_TEST_TP_SIZE: Tensor parallel size (default: 1)
    ATOM_TEST_GPUS_PER_NODE: GPUs available per node (default: 1)

Usage:
    pytest tests/workers/rollout/rollout_atom/test_atom_integration.py -v -s
    or
    python tests/workers/rollout/rollout_atom/test_atom_integration.py
"""

import asyncio
import os
import subprocess
import time

import pytest


def _get_test_config():
    """Load test configuration from environment."""
    model_root = os.path.expanduser(os.getenv("ATOM_TEST_MODEL_PATH_ROOT", "~/models"))
    return {
        "model_path": os.path.join(model_root, "Qwen/Qwen2.5-0.5B-Instruct"),
        "tp_size": int(os.getenv("ATOM_TEST_TP_SIZE", "1")),
        "gpus_per_node": int(os.getenv("ATOM_TEST_GPUS_PER_NODE", "1")),
    }


def _build_rollout_config(*, response_length=50, free_cache_engine=False):
    """Build rollout and model configs via Hydra compose."""
    from hydra import compose, initialize_config_dir

    config_dir = os.path.abspath("verl/verl/trainer/config")
    if not os.path.exists(config_dir):
        config_dir = os.path.abspath("verl/trainer/config")

    test_cfg = _get_test_config()

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(config_name="ppo_trainer")

    config.trainer.n_gpus_per_node = test_cfg["gpus_per_node"]
    config.trainer.nnodes = 1
    config.actor_rollout_ref.model.path = test_cfg["model_path"]
    config.actor_rollout_ref.rollout.name = "atom"
    config.actor_rollout_ref.rollout.mode = "async"
    config.actor_rollout_ref.rollout.tensor_model_parallel_size = test_cfg["tp_size"]
    config.actor_rollout_ref.rollout.enforce_eager = True
    if response_length is not None:
        config.actor_rollout_ref.rollout.response_length = response_length
    if free_cache_engine:
        config.actor_rollout_ref.rollout.free_cache_engine = True

    return config.actor_rollout_ref.rollout, config.actor_rollout_ref.model


class TestATOMGenerate:
    """Test ATOM generate end-to-end."""

    def test_async_generate(self):
        """Test ATOM generate method with a real model."""
        import ray

        test_cfg = _get_test_config()
        if not os.path.exists(test_cfg["model_path"]):
            pytest.skip(f"Model not found: {test_cfg['model_path']}")

        try:
            ray.init(
                runtime_env={
                    "env_vars": {
                        "TOKENIZERS_PARALLELISM": "true",
                        "NCCL_DEBUG": "WARN",
                    }
                },
                ignore_reinit_error=True,
            )

            rollout_config, model_config = _build_rollout_config(response_length=50)

            from verl.workers.rollout.replica import get_rollout_replica_class

            rollout_server_class = get_rollout_replica_class("atom")
            server = rollout_server_class(
                replica_rank=0,
                config=rollout_config,
                model_config=model_config,
                gpus_per_node=test_cfg["gpus_per_node"],
            )

            asyncio.run(server.init_standalone())
            server_handle = server._server_handle

            # Generate with simple prompt
            prompt_ids = [1, 2, 3, 4, 5]
            sampling_params = {
                "temperature": 1.0,
                "logprobs": True,
            }

            result = ray.get(
                server_handle.generate.remote(
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                    request_id="test_generate_1",
                )
            )

            assert hasattr(result, "token_ids"), "Result should have token_ids"
            assert isinstance(result.token_ids, list), "token_ids should be a list"
            assert len(result.token_ids) > 0, "Should generate at least one token"

            if result.log_probs is not None:
                assert len(result.log_probs) == len(result.token_ids), (
                    "log_probs length should match token_ids"
                )

            print(f"Generated {len(result.token_ids)} tokens")
            print(f"Token IDs: {result.token_ids[:10]}...")

        finally:
            ray.shutdown()
            subprocess.run(["ray", "stop"], capture_output=True)


class TestATOMMemoryManagement:
    """Test ATOM sleep/wake memory management."""

    def test_sleep_wake_cycle(self):
        """Test that sleep reduces memory and wake restores it."""
        import ray
        import torch

        test_cfg = _get_test_config()
        if not os.path.exists(test_cfg["model_path"]):
            pytest.skip(f"Model not found: {test_cfg['model_path']}")

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        try:
            ray.init(
                runtime_env={
                    "env_vars": {
                        "TOKENIZERS_PARALLELISM": "true",
                        "NCCL_DEBUG": "WARN",
                    }
                },
                ignore_reinit_error=True,
            )

            rollout_config, model_config = _build_rollout_config(free_cache_engine=True)

            from verl.workers.rollout.replica import get_rollout_replica_class

            rollout_server_class = get_rollout_replica_class("atom")
            server = rollout_server_class(
                replica_rank=0,
                config=rollout_config,
                model_config=model_config,
                gpus_per_node=test_cfg["gpus_per_node"],
            )

            asyncio.run(server.init_standalone())
            server_handle = server._server_handle

            # Measure baseline memory
            mem_free_before, mem_total = torch.cuda.mem_get_info(0)
            baseline_used_mb = (mem_total - mem_free_before) / (1024**2)
            print(f"Baseline memory: {baseline_used_mb:.2f} MB")

            # Sleep
            ray.get(server_handle.sleep.remote())
            time.sleep(2)

            mem_free_after_sleep, _ = torch.cuda.mem_get_info(0)
            sleep_used_mb = (mem_total - mem_free_after_sleep) / (1024**2)
            freed_mb = baseline_used_mb - sleep_used_mb
            print(f"After sleep: {sleep_used_mb:.2f} MB (freed {freed_mb:.2f} MB)")

            assert freed_mb > 0, "Sleep should free some GPU memory"

            # Wake up
            ray.get(server_handle.wake_up.remote())
            time.sleep(1)

            # Verify generation works after wake
            result = ray.get(
                server_handle.generate.remote(
                    prompt_ids=[1, 2, 3],
                    sampling_params={"temperature": 1.0},
                    request_id="test_after_wake",
                )
            )
            assert hasattr(result, "token_ids")
            assert len(result.token_ids) > 0, "Should generate after wake"
            print(f"Post-wake generation OK: {len(result.token_ids)} tokens")

        finally:
            ray.shutdown()
            subprocess.run(["ray", "stop"], capture_output=True)


if __name__ == "__main__":
    # Run as standalone script
    print("=" * 60)
    print("ATOM Integration Tests")
    print("=" * 60)
    test_gen = TestATOMGenerate()
    test_gen.test_async_generate()
    print("\nGenerate test passed!")

    test_mem = TestATOMMemoryManagement()
    test_mem.test_sleep_wake_cycle()
    print("\nSleep/Wake test passed!")
