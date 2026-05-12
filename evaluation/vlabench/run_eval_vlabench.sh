#!/bin/bash
# SimVLA VLABench 评估脚本
#
# 需要两个终端：
#   终端1（simvla 环境）：启动策略服务器
#   终端2（vlabench 环境）：运行本脚本
#
# 用法：
#   bash run_eval_vlabench.sh <port> <n_episode> <eval_track> <save_name>
#
# 示例（调试用简化测试集）：
#   bash run_eval_vlabench.sh 8200 10 track_debug_simple debug_run
#
# 示例（完整 track_1）：
#   bash run_eval_vlabench.sh 8200 50 track_1_in_distribution full_run

PORT=${1:-8200}
N_EPISODE=${2:-10}
EVAL_TRACK=${3:-track_debug_simple}
SAVE_NAME=${4:-simvla_eval}
CHECKPOINT=${5:-"unknown"}
SAVE_VIDEO=${6:-true}   # 是否保存视频，默认开启

# Auto-load paths.env if present and vars not already set
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [ -f "${REPO_ROOT}/paths.env" ] && [ -z "${SIMVLA_VLABENCH_CODE}" ]; then
    source "${REPO_ROOT}/paths.env"
fi

VLABENCH_DIR="${SIMVLA_VLABENCH_CODE:-/datasets/code/VLABench}"

# 时间戳子目录（精确到小时）
TIMESTAMP=$(date +"%Y%m%d_%H")
SAVE_DIR="${SIMVLA_EVAL_RESULTS:-./eval_results}/${SAVE_NAME}_${TIMESTAMP}"

export VLABENCH_ROOT="${VLABENCH_DIR}/VLABench"
export MUJOCO_GL=egl

# 创建保存目录并写 eval_info.txt
mkdir -p "${SAVE_DIR}"
cat > "${SAVE_DIR}/eval_info.txt" << EOF
===== SimVLA VLABench Evaluation =====
时间：$(date "+%Y-%m-%d %H:%M:%S")
Checkpoint 路径：${CHECKPOINT}
评估 track：${EVAL_TRACK}
每个任务 episodes：${N_EPISODE}
服务器端口：${PORT}
保存视频：${SAVE_VIDEO}
VLABench 目录：${VLABENCH_DIR}
结果保存目录：${SAVE_DIR}
EOF

echo "============================================================"
echo "SimVLA VLABench Evaluation"
echo "============================================================"
echo "Server:     localhost:${PORT}"
echo "Track:      ${EVAL_TRACK}"
echo "Episodes:   ${N_EPISODE} per task"
echo "Checkpoint: ${CHECKPOINT}"
echo "Save video: ${SAVE_VIDEO}"
echo "Save dir:   ${SAVE_DIR}"
echo "============================================================"

# 把 simvla_policy.py 临时加入 VLABench policy 目录
POLICY_DIR="${VLABENCH_ROOT}/evaluation/model/policy"
POLICY_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/simvla_policy.py"

if [ ! -f "${POLICY_DIR}/simvla_policy.py" ]; then
    cp "${POLICY_SRC}" "${POLICY_DIR}/simvla_policy.py"
    echo "Copied simvla_policy.py to ${POLICY_DIR}"
fi

cd "${VLABENCH_DIR}"

EVAL_ARGS="--eval-track ${EVAL_TRACK} \
    --policy simvla \
    --host localhost \
    --port ${PORT} \
    --n-episode ${N_EPISODE} \
    --save-dir ${SAVE_DIR} \
    --metrics success_rate progress_score"

if [ "${SAVE_VIDEO}" = "true" ]; then
    EVAL_ARGS="${EVAL_ARGS} --visulization"
fi

python scripts/evaluate_policy.py ${EVAL_ARGS}

echo "Results saved to ${SAVE_DIR}"
