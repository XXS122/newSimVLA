#!/bin/bash
# SimVLA Training Script for VLABench (Small Model)
#
# Key features:
#   - 384x384 image resolution
#   - 4 views: front, wrist, image_0, image_1
#   - vlabench_joint action mode (7-dim action, 7-dim proprio)
#   - Smaller action transformer configuration

set -e

# Auto-load paths.env if present and vars not already set
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${SCRIPT_DIR}/paths.env" ] && [ -z "${SIMVLA_SMOLVLM_MODEL}" ]; then
    source "${SCRIPT_DIR}/paths.env"
fi

# =============================================================================
# Command line arguments (with defaults)
# =============================================================================

BATCH_SIZE=${1:-32}
LEARNING_COEF=${2:-0.1}
OUTPUT_DIR=${3:-./runs/simvla_vlabench_small}
RESUME_CKPT=${4:-""}

echo "Training parameters:"
echo "   batch_size: $BATCH_SIZE"
echo "   learning_coef: $LEARNING_COEF"
echo "   output_dir: $OUTPUT_DIR"
echo "   resume_ckpt: ${RESUME_CKPT:-'None (training from scratch)'}"

# GPU configuration
export CUDA_VISIBLE_DEVICES="${SIMVLA_CUDA_DEVICES:-0,1,2,3}"
NUM_GPUS="${SIMVLA_NUM_GPUS:-4}"

# Suppress TensorFlow logs
export TF_CPP_MIN_LOG_LEVEL=2

# =============================================================================
# Path configuration
# =============================================================================
SMOLVLM_MODEL="${SIMVLA_SMOLVLM_MODEL:-HuggingFaceTB/SmolVLM-500M-Instruct}"
VLABENCH_DATA_DIR="${SIMVLA_VLABENCH_DATA:-./datasets/vlabench/data/1.0.0}"
NORM_STATS_PATH="./norm_stats/vlabench_norm.json"
TRAIN_METAS_PATH="./datasets/metas/vlabench_train.json"

# =============================================================================
# Training hyperparameters
# =============================================================================
LEARNING_RATE=1e-4
NUM_ACTIONS=10
ITERS=200000
WARMUP_STEPS=0
FREEZE_STEPS=1000
SAVE_INTERVAL=10000
LOG_INTERVAL=20
NUM_WORKERS=0
MAX_GRAD_NORM=1.0
NUM_VIEWS=4
# Model architecture (Small configuration)
HIDDEN_SIZE=768
DEPTH=12
NUM_HEADS=12
USE_ADALN=false

# =============================================================================
# Step 1: Create training metadata (if not exists)
# =============================================================================
if [ ! -f "$TRAIN_METAS_PATH" ]; then
    echo "Creating training metadata..."
    python create_vlabench_meta.py \
        --data_dir "$VLABENCH_DATA_DIR" \
        --output "$TRAIN_METAS_PATH"
fi

# =============================================================================
# Step 2: Compute normalization statistics (if not exists)
# =============================================================================
if [ ! -f "$NORM_STATS_PATH" ]; then
    echo "Computing normalization statistics..."
    python compute_vlabench_norm_stats.py \
        --data_dir "$VLABENCH_DATA_DIR" \
        --output "$NORM_STATS_PATH"
fi

# =============================================================================
# Step 3: Build training arguments
# =============================================================================
ARGS="--output_dir ${OUTPUT_DIR} \
    --train_metas_path ${TRAIN_METAS_PATH} \
    --smolvlm_model_path ${SMOLVLM_MODEL} \
    --action_mode vlabench_joint \
    --num_views ${NUM_VIEWS} \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --learning_coef ${LEARNING_COEF} \
    --num_actions ${NUM_ACTIONS} \
    --iters ${ITERS} \
    --warmup_steps ${WARMUP_STEPS} \
    --freeze_steps ${FREEZE_STEPS} \
    --hidden_size ${HIDDEN_SIZE} \
    --depth ${DEPTH} \
    --num_heads ${NUM_HEADS} \
    --num_workers ${NUM_WORKERS} \
    --save_interval ${SAVE_INTERVAL} \
    --log_interval ${LOG_INTERVAL} \
    --image_size 384 \
    --norm_stats_path ${NORM_STATS_PATH} \
    --max_grad_norm ${MAX_GRAD_NORM}"

if [ "${USE_ADALN}" = true ]; then
    ARGS="${ARGS} --use_adaln"
fi

if [ -n "${RESUME_CKPT}" ]; then
    ARGS="${ARGS} --models ${RESUME_CKPT} --resume"
    echo "Resuming from ${RESUME_CKPT}"
fi

# =============================================================================
# Step 4: Start training
# =============================================================================
echo "============================================================"
echo "Starting SimVLA Training on VLABench (Small Action Transformer)"
echo "============================================================"
echo "SmolVLM backbone: ${SMOLVLM_MODEL}"
echo "VLABench data dir: ${VLABENCH_DATA_DIR}"
echo "Normalization stats: ${NORM_STATS_PATH}"
echo "Action mode: vlabench_joint"
echo "Num views: 4"
echo "Batch size: ${BATCH_SIZE}"
echo "Learning rate: ${LEARNING_RATE}"
echo "Learning coef: ${LEARNING_COEF}"
echo "Num actions: ${NUM_ACTIONS}"
echo "Image size: 384x384"
echo "============================================================"
echo "Action Transformer configuration:"
echo "   Hidden size: ${HIDDEN_SIZE}"
echo "   Depth: ${DEPTH}"
echo "   Num heads: ${NUM_HEADS}"
echo "   Use AdaLN: ${USE_ADALN}"
echo "============================================================"
echo "GPU config: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, num_processes=${NUM_GPUS}"
echo "Output directory: ${OUTPUT_DIR}"
echo "============================================================"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
accelerate launch \
    --num_processes=${NUM_GPUS} \
    --main_process_port 29505 \
    --mixed_precision fp16 \
    train_smolvlm.py ${ARGS}

echo "Training completed!"
