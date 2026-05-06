#!/bin/bash
# SimVLA Training Script for VLABench - Dual-Stream Multi-View Fusion
#
# 双流融合模式：静态流（front/image_0/image_1）+ 动态流（wrist）
# 通过 Cross-Attention 融合两路视觉信息，提升末端运动感知能力

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${SCRIPT_DIR}/paths.env" ] && [ -z "${SIMVLA_SMOLVLM_MODEL}" ]; then
    source "${SCRIPT_DIR}/paths.env"
fi

# =============================================================================
# Command line arguments
# =============================================================================
BATCH_SIZE=${1:-32}
LEARNING_COEF=${2:-0.1}
RESUME_CKPT=${3:-""}
FUSION_TYPE=${4:-"cross_attn"}   # add | concat_linear | cross_attn

TIMESTAMP=$(date +"%Y%m%d_%H")
BASE_OUTPUT_DIR="${SIMVLA_OUTPUT_DIR:-./runs}"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/vlabench_dualstream_${FUSION_TYPE}_${TIMESTAMP}"

echo "Training parameters:"
echo "   batch_size:    $BATCH_SIZE"
echo "   learning_coef: $LEARNING_COEF"
echo "   fusion_type:   $FUSION_TYPE"
echo "   output_dir:    $OUTPUT_DIR"
echo "   resume_ckpt:   ${RESUME_CKPT:-'None (training from scratch)'}"

export CUDA_VISIBLE_DEVICES="${SIMVLA_CUDA_DEVICES:-0,1,2,3}"
NUM_GPUS="${SIMVLA_NUM_GPUS:-4}"
export TF_CPP_MIN_LOG_LEVEL=2

# =============================================================================
# Paths
# =============================================================================
SMOLVLM_MODEL="${SIMVLA_SMOLVLM_MODEL:-HuggingFaceTB/SmolVLM-500M-Instruct}"
VLABENCH_DATA_DIR="${SIMVLA_VLABENCH_DATA:-./datasets/vlabench/data/1.0.0}"
NORM_STATS_PATH="./norm_stats/vlabench_norm.json"
TRAIN_METAS_PATH="./datasets/metas/vlabench_train.json"

# =============================================================================
# Hyperparameters
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
HIDDEN_SIZE=768
DEPTH=12
NUM_HEADS=12

# =============================================================================
# Step 1: Create metadata (if not exists)
# =============================================================================
if [ ! -f "$TRAIN_METAS_PATH" ]; then
    echo "Creating training metadata..."
    python create_vlabench_meta.py \
        --data_dir "$VLABENCH_DATA_DIR" \
        --output "$TRAIN_METAS_PATH"
fi

# =============================================================================
# Step 2: Compute norm stats (if not exists)
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
    --max_grad_norm ${MAX_GRAD_NORM} \
    --use_dual_stream \
    --dual_stream_fusion ${FUSION_TYPE}"

if [ -n "${RESUME_CKPT}" ]; then
    ARGS="${ARGS} --models ${RESUME_CKPT} --resume"
    echo "Resuming from ${RESUME_CKPT}"
fi

# =============================================================================
# Step 4: Write run info
# =============================================================================
mkdir -p "${OUTPUT_DIR}"
cat > "${OUTPUT_DIR}/run_info.txt" << EOF
===== SimVLA VLABench Dual-Stream Training =====
时间：$(date "+%Y-%m-%d %H:%M:%S")
算法：SimVLA + 双流多视角融合（fusion=${FUSION_TYPE}）
Action Transformer：hidden=${HIDDEN_SIZE}, depth=${DEPTH}, heads=${NUM_HEADS}

双流配置：
  静态流视角：front / image_0 / image_1（场景语义）
  动态流视角：wrist（末端运动信息）
  融合方式：${FUSION_TYPE}

数据：
  VLABench data dir: ${VLABENCH_DATA_DIR}
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
echo "Starting SimVLA Dual-Stream Training on VLABench"
echo "============================================================"
echo "SmolVLM backbone:  ${SMOLVLM_MODEL}"
echo "Dual-stream fusion: ${FUSION_TYPE}"
echo "  Static stream:   front / image_0 / image_1"
echo "  Dynamic stream:  wrist"
echo "Batch size:        ${BATCH_SIZE}"
echo "Learning rate:     ${LEARNING_RATE} (VLM coef: ${LEARNING_COEF})"
echo "Num views:         4"
echo "Image size:        384x384"
echo "GPU config:        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, num_processes=${NUM_GPUS}"
echo "Output directory:  ${OUTPUT_DIR}"
echo "============================================================"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
accelerate launch \
    --num_processes=${NUM_GPUS} \
    --main_process_port 29506 \
    --mixed_precision fp16 \
    train_smolvlm.py ${ARGS}

echo "Training completed!"
