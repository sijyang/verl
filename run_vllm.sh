#!/usr/bin/env bash
set -euo pipefail


YOUR_PROJECT_NAME=r1-verl-ppo-upstream
YOUR_RUN_NAME=r1-training_ppo-upstream
GPUS_PER_NODE=2
MODEL_PATH=/data/models/Qwen3-8B
ENGINE=vllm
TRAIN=./data/gsm8k/train.parquet
VAL=./data/gsm8k/test.parquet
LOG=verl_demo.log



export HIP_VISIBLE_DEVICES=0,1
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
# export SGLANG_USE_AITER=True                            # AITER 支持 FP8 (vllm 不需要)
# export CUDA_VISIBLE_DEVICES=""                          # 禁用 CUDA，避免 libcuda.so.1 错误
export HIP_FORCE_DEV_KERNARG=1                          # ROCm 优化
export GPU_FORCE_64BIT_PTR=1                            # 避免内存指针问题
export HSA_FORCE_FINE_GRAIN_PCIE=1                      # 改善 PCIe 通信稳定性
export GPU_COREDUMP_ENABLE=0                            # 禁用 GPU core dump
export HSA_ENABLE_COREDUMP=0                            # 禁用 HSA core dump
export VLLM_USE_V1=1                                    # verl 代码强制使用 V1 引擎

ray stop --force 2>/dev/null || true

# python3 examples/data_preprocess/gsm8k.py --local_save_dir ./data/gsm8k
# python3 -c "import transformers; transformers.pipeline('text-generation', model='$MODEL_PATH')"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
  data.train_files="$TRAIN" \
  data.val_files="$VAL" \
  data.train_batch_size=64 \
  data.val_batch_size=64 \
  data.max_prompt_length=256 \
  data.max_response_length=128 \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name="$ENGINE" \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
  critic.optim.lr=1e-5 \
  critic.model.path="$MODEL_PATH" \
  critic.ppo_micro_batch_size_per_gpu=2 \
  algorithm.kl_ctrl.kl_coef=0.001 \
  trainer.logger=console \
  trainer.project_name="$YOUR_PROJECT_NAME" \
  trainer.experiment_name="$YOUR_RUN_NAME" \
  trainer.val_before_train=False \
  trainer.n_gpus_per_node="$GPUS_PER_NODE" \
  trainer.nnodes=1 \
  +ray_kwargs.ray_init.num_gpus=2 \
  +ray_kwargs.ray_init.runtime_env.env.VLLM_USE_V1=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=500 \
  trainer.total_epochs=15 \
  ++actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  ++critic.model.override_config.attn_implementation=sdpa \
  ++actor_rollout_ref.model.use_fused_kernels=False \
  ++actor_rollout_ref.actor.use_torch_compile=False \
  ++actor_rollout_ref.actor.fsdp_config.use_torch_compile=False \
  ++actor_rollout_ref.ref.fsdp_config.use_torch_compile=False \
  ++critic.model.fsdp_config.use_torch_compile=False \
  ++critic.model.use_remove_padding=False \
  ++actor_rollout_ref.rollout.enable_sleep_mode=False \
  ++actor_rollout_ref.rollout.free_cache_engine=False \
  2>&1 | tee "$LOG"