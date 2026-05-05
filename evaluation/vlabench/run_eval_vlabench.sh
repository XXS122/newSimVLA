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

VLABENCH_DIR="${SIMVLA_VLABENCH_CODE:-/datasets/code/VLABench}"
SAVE_DIR="${SIMVLA_EVAL_RESULTS:-./eval_results}/${SAVE_NAME}"

export VLABENCH_ROOT="${VLABENCH_DIR}/VLABench"
export MUJOCO_GL=egl

echo "============================================================"
echo "SimVLA VLABench Evaluation"
echo "============================================================"
echo "Server:     localhost:${PORT}"
echo "Track:      ${EVAL_TRACK}"
echo "Episodes:   ${N_EPISODE} per task"
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

python scripts/evaluate_policy.py \
    --eval-track "${EVAL_TRACK}" \
    --policy simvla \
    --host localhost \
    --port "${PORT}" \
    --n-episode "${N_EPISODE}" \
    --save-dir "${SAVE_DIR}" \
    --metrics success_rate progress_score

echo "Results saved to ${SAVE_DIR}"
