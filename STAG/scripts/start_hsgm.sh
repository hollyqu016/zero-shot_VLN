#!/usr/bin/env bash
# 一键在后台启动 HSGM R2R 实验。
#
# 直接运行（使用默认参数）：
#   bash scripts/start_hsgm.sh
#
# 可选参数：
#   bash scripts/start_hsgm.sh [BEGIN_IDX] [END_IDX] [MAX_STEPS] [MODEL_NAME]
#
# 示例：
#   bash scripts/start_hsgm.sh 1 -1 100 qwen3.6:35b

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_DIR="${PROJECT_ROOT}/src"
LOG_DIR="${PROJECT_ROOT}/log"

BEGIN_IDX="${1:-31}"
END_IDX="${2:--1}"
MAX_STEPS="${3:-100}"
MODEL_NAME="${4:-qwen3.6:35b}"

TIMESTAMP="$(date '+%Y%m%d-%H%M%S')"
LOG_FILE="${LOG_DIR}/hsgm-${TIMESTAMP}.log"

mkdir -p "${LOG_DIR}"
cd "${SRC_DIR}"

NO_PROXY=127.0.0.1,localhost http_proxy= https_proxy= nohup python -u run_experiments.py \
    --task r2r \
    --config ../config/vlnce_test.yaml \
    --model_name "${MODEL_NAME}" \
    --begin_idx "${BEGIN_IDX}" \
    --end_idx "${END_IDX}" \
    --max_steps "${MAX_STEPS}" \
    > "${LOG_FILE}" 2>&1 < /dev/null &

PID=$!

echo "HSGM 实验已在后台启动"
echo "进程号: ${PID}"
echo "日志: ${LOG_FILE}"
echo "查看日志: tail -f '${LOG_FILE}'"

