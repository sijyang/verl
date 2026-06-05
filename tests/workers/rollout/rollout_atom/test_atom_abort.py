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

"""
Test ATOM abort_all_requests and resume_generation functionality.

ATOM does not currently support request-level abort, so abort_all_requests
is a no-op. This test verifies that the no-op path does not break generation.

Usage:
    pytest tests/workers/rollout/rollout_atom/test_atom_abort.py -v -s
    or
    python tests/workers/rollout/rollout_atom/test_atom_abort.py

Environment variables:
    ATOM_TEST_MODEL_PATH_ROOT: parent directory containing model weights (default: ~/models)
    ATOM_TEST_TP_SIZE: tensor parallel size (default: 1)
    ATOM_TEST_GPUS_PER_NODE: number of GPUs (default: 1)
"""

import asyncio
import os
import subprocess
import time
from uuid import uuid4

import pytest


def _get_test_config():
    model_root = os.path.expanduser(os.getenv("ATOM_TEST_MODEL_PATH_ROOT", "~/models"))
    return {
        "model_path": os.path.join(model_root, "Qwen3-0.6B"),
        "tp_size": int(os.getenv("ATOM_TEST_TP_SIZE", "1")),
        "gpus_per_node": int(os.getenv("ATOM_TEST_GPUS_PER_NODE", "1")),
    }


def _build_rollout_config(*, response_length=512):
    from hydra import compose, initialize_config_dir

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
    config.actor_rollout_ref.rollout.response_length = response_length

    return config.actor_rollout_ref.rollout, config.actor_rollout_ref.model


def test_atom_abort():
    NUM_PROMPTS = 4
    ABORT_DELAY = 0.5

    test_cfg = _get_test_config()

    if not os.path.exists(test_cfg["model_path"]):
        pytest.skip(f"Model not found: {test_cfg['model_path']}")

    import ray
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
            }
        },
        ignore_reinit_error=True,
    )

    try:
        rollout_config, model_config = _build_rollout_config(response_length=512)

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

        # ── Prepare prompts ──
        prompt_ids = [[1, 2, 3, 4, 5]] * NUM_PROMPTS
        sampling_params = {"temperature": 1.0, "logprobs": False}

        # ── Start generations ──
        generate_refs = []
        for i in range(NUM_PROMPTS):
            request_id = f"abort_test_{i}_{uuid4().hex[:8]}"
            ref = server_handle.generate.remote(
                prompt_ids=prompt_ids[i],
                sampling_params=sampling_params,
                request_id=request_id,
            )
            generate_refs.append((i, request_id, ref))

        time.sleep(ABORT_DELAY)

        # ── Abort ──
        abort_start = time.perf_counter()
        abort_result = ray.get(server_handle.abort_all_requests.remote())
        abort_time = time.perf_counter() - abort_start

        assert "aborted_count" in abort_result
        assert abort_time < 1.0, f"Abort should be fast, took {abort_time:.2f}s"

        # ── Wait for all generations ──
        for i, request_id, ref in generate_refs:
            output = ray.get(ref, timeout=30.0)
            assert output is not None, f"Request {request_id} should not timeout"
            assert hasattr(output, "token_ids")

        # ── Resume and verify generation still works ──
        ray.get(server_handle.resume_generation.remote())

        post_resume_ref = server_handle.generate.remote(
            prompt_ids=[1, 2, 3],
            sampling_params=sampling_params,
            request_id=f"post_resume_{uuid4().hex[:8]}",
        )
        output = ray.get(post_resume_ref, timeout=30.0)
        assert output is not None
        assert len(output.token_ids) > 0, "Should generate after resume"

    finally:
        ray.shutdown()
        subprocess.run(["ray", "stop"], capture_output=True)


if __name__ == "__main__":
    test_atom_abort()
