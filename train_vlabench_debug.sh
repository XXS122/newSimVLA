#!/bin/bash
# SimVLA Debug Training Script for VLABench
#
# 用于快速调试：只用前 50 个 shard，训练 10000 步（约 1-2 小时）
# 参数与全量训练保持一致（image_size=384, num_views=4, fp16）
# 用法：CUDA_VISIBLE_DEVICES=2,3 bash train_vlabench_debug.sh

set -e

# Auto-load paths.env if present and vars not already set
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${SCRIPT_DIR}/paths.env" ] && [ -z "${SIMVLA_SMOLVLM_MODEL}" ]; then
    source "${SCRIPT_DIR}/paths.env"
fi

# =============================================================================
# Command line arguments (with defaults)
# =============================================================================

BATCH_SIZE=${1:-16}
LEARNING_COEF=${2:-0.1}
RESUME_CKPT=${3:-""}

# 时间戳子目录（精确到小时）
TIMESTAMP=$(date +"%Y%m%d_%H")
BASE_OUTPUT_DIR="${SIMVLA_OUTPUT_DIR:-./runs}"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/vlabench_debug_${TIMESTAMP}"

echo "Debug training parameters:"
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
TRAIN_METAS_PATH="./datasets/metas/vlabench_debug_train.json"
DEBUG_MAX_FILES=50

# =============================================================================
# Debug-specific hyperparameters
# =============================================================================
LEARNING_RATE=1e-4
NUM_ACTIONS=10
ITERS=10000
WARMUP_STEPS=0
FREEZE_STEPS=500
SAVE_INTERVAL=2000
LOG_INTERVAL=10
NUM_WORKERS=0
MAX_GRAD_NORM=1.0
NUM_VIEWS=4

# Model architecture (same as full training)
HIDDEN_SIZE=768
DEPTH=12
NUM_HEADS=12
USE_ADALN=false
USE_DUAL_STREAM=false          # 启用双流融合
DUAL_STREAM_FUSION=cross_attn # add | concat_linear | cross_attn

# =============================================================================
# Step 1: Create debug metadata (50 shards)
# =============================================================================
if [ ! -f "$TRAIN_METAS_PATH" ]; then
    echo "Creating debug metadata (${DEBUG_MAX_FILES} shards)..."
    python create_vlabench_meta.py \
        --data_dir "$VLABENCH_DATA_DIR" \
        --max_files "$DEBUG_MAX_FILES" \
        --output "$TRAIN_METAS_PATH"
fi

# =============================================================================
# Step 2: Compute normalization statistics (reuse full stats if exists)
# =============================================================================
if [ ! -f "$NORM_STATS_PATH" ]; then
    echo "Computing normalization statistics (using ${DEBUG_MAX_FILES} shards for speed)..."
    python compute_vlabench_norm_stats.py \
        --data_dir "$VLABENCH_DATA_DIR" \
        --max_files "$DEBUG_MAX_FILES" \
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

if [ "${USE_DUAL_STREAM}" = true ]; then
    ARGS="${ARGS} --use_dual_stream --dual_stream_fusion ${DUAL_STREAM_FUSION}"
fi

if [ -n "${RESUME_CKPT}" ]; then
    ARGS="${ARGS} --models ${RESUME_CKPT} --resume"
    echo "Resuming from ${RESUME_CKPT}"
fi

# =============================================================================
# Step 4: Start debug training
# =============================================================================

# 创建输出目录并写 run_info.txt
mkdir -p "${OUTPUT_DIR}"
cat > "${OUTPUT_DIR}/run_info.txt" << EOF
===== SimVLA VLABench Debug Training =====
时间：$(date "+%Y-%m-%d %H:%M:%S")
算法：SimVLA + VLABench RLDS
双流融合：${USE_DUAL_STREAM} (${DUAL_STREAM_FUSION})
Action Transformer：hidden=${HIDDEN_SIZE}, depth=${DEPTH}, heads=${NUM_HEADS}, adaln=${USE_ADALN}

数据：
  VLABench data dir: ${VLABENCH_DATA_DIR}
  Debug shards: ${DEBUG_MAX_FILES} / 512
  Norm stats: ${NORM_STATS_PATH}
  Train metas: ${TRAIN_METAS_PATH}

模型：
  SmolVLM backbone: ${SMOLVLM_MODEL}
  Action mode: vlabench_joint
  Num views: ${NUM_VIEWS}
  Num actions: ${NUM_ACTIONS}
  Image size: 384x384

训练参数：
  Batch size: ${BATCH_SIZE}
  Learning rate: ${LEARNING_RATE}
  Learning coef: ${LEARNING_COEF}
  Iters: ${ITERS}
  Freeze steps: ${FREEZE_STEPS}
  Save interval: ${SAVE_INTERVAL}

Resume checkpoint: ${RESUME_CKPT:-None}
GPU: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, num_processes=${NUM_GPUS}
Output dir: ${OUTPUT_DIR}
EOF

echo "============================================================"
echo "Starting SimVLA DEBUG Training on VLABench"
echo "============================================================"
echo "SmolVLM backbone: ${SMOLVLM_MODEL}"
echo "VLABench data dir: ${VLABENCH_DATA_DIR}"
echo "Debug shards: ${DEBUG_MAX_FILES} / 512"
echo "Normalization stats: ${NORM_STATS_PATH}"
echo "Action mode: vlabench_joint"
echo "Num views: 4"
echo "Batch size: ${BATCH_SIZE} (debug)"
echo "Iters: ${ITERS} (debug, full=200000)"
echo "Save interval: ${SAVE_INTERVAL}"
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
    --main_process_port 29506 \
    --mixed_precision fp16 \
    train_smolvlm.py ${ARGS}

echo "Debug training completed! Checkpoints saved to ${OUTPUT_DIR}"
