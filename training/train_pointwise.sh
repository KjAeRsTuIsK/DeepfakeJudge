#!/bin/bash

# =============================================================================
# DeepfakeJudge — Pointwise Training Script (NVIDIA GPUs)
#
# Fine-tunes a Qwen2.5-VL model for pointwise reasoning evaluation using
# LoRA with the ms-swift framework.
#
# Usage:
#   bash train_pointwise.sh
#
# Before running, update the following variables:
#   MODEL        — Base model name or local path
#   DATASET      — Path to the pointwise training JSONL file
#   OUTPUT_DIR   — Directory to save checkpoints
#   NUM_GPUS     — Number of GPUs to use
#   GPU_IDS      — Comma-separated GPU device IDs
# =============================================================================

# ---- User Configuration ----
MODEL="Qwen/Qwen2.5-VL-7B-Instruct"        # Options: Qwen/Qwen2.5-VL-3B-Instruct, Qwen/Qwen2.5-VL-7B-Instruct
DATASET="/path/to/dfj-meta-pointwise/train/data.jsonl"
OUTPUT_DIR="./output/pointwise_7b"
NUM_GPUS=2
GPU_IDS="0,1"

# ---- Image Processing (Qwen-VL defaults) ----
export MAX_PIXELS=1003520   # 1280 * 28 * 28
export IMAGE_FACTOR=28
export MIN_PIXELS=3136      # 4 * 28 * 28

# ---- Training ----
CUDA_VISIBLE_DEVICES=${GPU_IDS} \
NPROC_PER_NODE=${NUM_GPUS} \
swift sft \
    --model ${MODEL} \
    --use_hf true \
    --dataset ${DATASET} \
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 16 \
    --learning_rate 1e-6 \
    --lora_rank 32 \
    --lora_alpha 64 \
    --target_modules all-linear \
    --freeze_vit true \
    --gradient_accumulation_steps 1 \
    --save_strategy epoch \
    --save_total_limit 5 \
    --logging_steps 1 \
    --max_length 4096 \
    --output_dir ${OUTPUT_DIR} \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 4 \
    --bf16 true \
    --report_to wandb

echo "Training completed!"
echo "Model saved to: ${OUTPUT_DIR}"
