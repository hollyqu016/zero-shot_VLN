#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# HSGM_ARV_v1 diagnostic pilot
#
# Purpose:
#   Run 24 selected diagnostic R2R episodes for ARV_v1.
#   Baseline is NOT rerun; compare against the existing
#   100-episode baseline results under server_results.
#
# Submit:
#   cd ~/qjx_code/HSGM_ARV_v1
#   chmod +x run_arv_diag_24.sh
#   gpu-sbatch 1 run_arv_diag_24.sh --time 50:00:00 --cpu 24 --mem 64G
#
# Optional overrides:
#   OLLAMA_MODEL=qwen3.6:35b gpu-sbatch 1 run_arv_diag_24.sh ...
#   MAX_STEPS=100 gpu-sbatch 1 run_arv_diag_24.sh ...
#
# The Ollama server is started INSIDE the Slurm allocation.
# Once gpu-sbatch returns a job ID, SSH/local-network disconnects
# do not stop this job.
# ============================================================

QJX_ROOT="${QJX_ROOT:-/home/lixiaohai/qjx_code}"
PROJECT_DIR="${QJX_ROOT}/HSGM_ARV_v1"
DIAGNOSTIC_DIR="${QJX_ROOT}/paired_diagnostic"

# Same model/settings used for the previous baseline diagnostic run.
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen3.6:35b}"
MAX_STEPS="${MAX_STEPS:-100}"
CONDA_ENV="${CONDA_ENV:-hsgm_slurm}"
TOTAL_EPISODES="${TOTAL_EPISODES:-200}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-arv_v1_diag24}"

# Selected from the existing baseline-100 failures to maximize the chance
# of observing Attribution / Revision / Validation behavior.
EPISODES=(
    0
    6
    12
    16
    18
    20
    30
    32
    41
    43
    48
    50
    54
    55
    56
    65
    66
    67
    80
    81
    82
    88
    97
    98
)

if [[ ! -d "${PROJECT_DIR}" ]]; then
    echo "[ERROR] Project directory not found: ${PROJECT_DIR}" >&2
    exit 2
fi

if [[ ! -f "${PROJECT_DIR}/scripts/batch_test.sh" ]]; then
    echo "[ERROR] Missing batch runner: ${PROJECT_DIR}/scripts/batch_test.sh" >&2
    exit 2
fi

# ------------------------------------------------------------
# Conda setup for a non-interactive Slurm job
# ------------------------------------------------------------

if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
elif [[ -x "${HOME}/miniconda3/bin/conda" ]]; then
    CONDA_BASE="${HOME}/miniconda3"
elif [[ -x "${HOME}/anaconda3/bin/conda" ]]; then
    CONDA_BASE="${HOME}/anaconda3"
elif [[ -x "${HOME}/miniforge3/bin/conda" ]]; then
    CONDA_BASE="${HOME}/miniforge3"
else
    echo "[ERROR] Cannot locate conda in this non-interactive job" >&2
    exit 2
fi

# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

if ! command -v ollama >/dev/null 2>&1; then
    echo "[ERROR] ollama is not available inside the allocated job" >&2
    exit 2
fi

# ------------------------------------------------------------
# Per-job Ollama endpoint
# ------------------------------------------------------------

JOB_TOKEN="${SLURM_JOB_ID:-$$}"
OLLAMA_PORT="${OLLAMA_PORT:-$((20000 + JOB_TOKEN % 20000))}"

export OLLAMA_HOST="127.0.0.1:${OLLAMA_PORT}"
export OPENAI_BASE_URL="http://${OLLAMA_HOST}/v1"
export OPENAI_API_KEY="${OPENAI_API_KEY:-ollama}"
export OPENAI_MODEL="${OLLAMA_MODEL}"
export VLM_TEMPERATURE="${VLM_TEMPERATURE:-0}"
export DISABLE_AUTO_PROXY=1
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,${NO_PROXY}}"
export no_proxy="127.0.0.1,localhost${no_proxy:+,${no_proxy}}"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy

export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-32768}"
export OLLAMA_FLASH_ATTENTION="${OLLAMA_FLASH_ATTENTION:-1}"

# ------------------------------------------------------------
# Logging
# ------------------------------------------------------------

RUN_ID="${EXPERIMENT_TAG}_job${JOB_TOKEN}_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${DIAGNOSTIC_DIR}/logs/${RUN_ID}"
mkdir -p "${LOG_DIR}"

OLLAMA_LOG="${LOG_DIR}/ollama.log"
PREFLIGHT_LOG="${LOG_DIR}/preflight.json"
SUMMARY_LOG="${LOG_DIR}/summary.txt"

OLLAMA_PID=""

cleanup() {
    echo "[INFO] Cleanup at $(date)"
    if [[ -n "${OLLAMA_PID}" ]] && kill -0 "${OLLAMA_PID}" 2>/dev/null; then
        kill "${OLLAMA_PID}" 2>/dev/null || true
        wait "${OLLAMA_PID}" 2>/dev/null || true
    fi
}

trap cleanup EXIT INT TERM

# ------------------------------------------------------------
# Sanity checks before starting Ollama
# ------------------------------------------------------------

echo "============================================================"
echo " HSGM ARV_v1 diagnostic pilot"
echo "   episodes     : ${EPISODES[*]}"
echo "   model        : ${OLLAMA_MODEL}"
echo "   max steps    : ${MAX_STEPS}"
echo "   conda env    : ${CONDA_ENV}"
echo "   project      : ${PROJECT_DIR}"
echo "   job token    : ${JOB_TOKEN}"
echo "   log dir      : ${LOG_DIR}"
echo "============================================================"

cd "${PROJECT_DIR}"

echo "[INFO] Python: $(python --version 2>&1)"
echo "[INFO] Running ARV smoke test..."
python scripts/smoke_arv_trace.py

echo "[INFO] Compiling changed Python files..."
python -m py_compile \
    src/agent/belief.py \
    src/agent/agent.py \
    src/run_experiments.py

# ------------------------------------------------------------
# Start Ollama inside the Slurm job
# ------------------------------------------------------------

echo "[INFO] Starting Ollama at ${OLLAMA_HOST}"
ollama serve >"${OLLAMA_LOG}" 2>&1 &
OLLAMA_PID=$!

OLLAMA_READY=0
for _ in $(seq 1 90); do
    if ollama list >/dev/null 2>&1; then
        OLLAMA_READY=1
        break
    fi
    sleep 2
done

if [[ "${OLLAMA_READY}" -ne 1 ]]; then
    echo "[ERROR] Ollama did not become ready; see ${OLLAMA_LOG}" >&2
    exit 3
fi

if ! ollama show "${OLLAMA_MODEL}" >/dev/null 2>&1; then
    echo "[ERROR] Model '${OLLAMA_MODEL}' is not installed on the server" >&2
    ollama list >&2 || true
    exit 3
fi

PREFLIGHT_PAYLOAD="$(printf '{"model":"%s","messages":[{"role":"user","content":"Reply with OK only."}],"temperature":0,"stream":false}' "${OLLAMA_MODEL}")"

PREFLIGHT_STATUS="$(
    curl --noproxy '*' --silent --show-error \
        --output "${PREFLIGHT_LOG}" \
        --write-out '%{http_code}' \
        --header 'Content-Type: application/json' \
        --data "${PREFLIGHT_PAYLOAD}" \
        "${OPENAI_BASE_URL}/chat/completions" || true
)"

if [[ "${PREFLIGHT_STATUS}" != "200" ]]; then
    echo "[ERROR] Ollama inference preflight returned HTTP ${PREFLIGHT_STATUS}" >&2
    echo "[ERROR] Response: ${PREFLIGHT_LOG}" >&2
    cat "${PREFLIGHT_LOG}" >&2 || true
    echo "[ERROR] Ollama log: ${OLLAMA_LOG}" >&2
    exit 3
fi

echo "[INFO] Ollama inference preflight passed"

# ------------------------------------------------------------
# Run selected episodes one by one
#
# batch_test.sh uses [BEGIN_EPISODE, END_EPISODE), therefore
# one episode EP is invoked as EP -> EP+1.
#
# A failure in one episode is recorded but does not abort the
# remaining diagnostic set.
# ------------------------------------------------------------

PASSED_EPISODES=()
FAILED_EPISODES=()

for EP in "${EPISODES[@]}"; do
    BEGIN_EPISODE="${EP}"
    END_EPISODE="$((EP + 1))"

    EP_TAG="${EXPERIMENT_TAG}_ep${EP}"
    EP_LOG="${LOG_DIR}/episode_${EP}.log"

    echo
    echo "============================================================"
    echo "[INFO] Episode ${EP} starting at $(date)"
    echo "[INFO] Episode log: ${EP_LOG}"
    echo "============================================================"

    set +e
    COMMENT="${EP_TAG}" \
    MAX_STEPS="${MAX_STEPS}" \
    bash "${PROJECT_DIR}/scripts/batch_test.sh" \
        r2r "${BEGIN_EPISODE}" "${END_EPISODE}" "${OLLAMA_MODEL}" \
        2>&1 | tee "${EP_LOG}"

    EP_STATUS=${PIPESTATUS[0]}
    set -e

    if [[ "${EP_STATUS}" -eq 0 ]]; then
        PASSED_EPISODES+=("${EP}")
        echo "[INFO] Episode ${EP} completed successfully at $(date)"
    else
        FAILED_EPISODES+=("${EP}")
        echo "[ERROR] Episode ${EP} failed with status ${EP_STATUS} at $(date)" >&2
        echo "[ERROR] Continuing with the remaining episodes." >&2
    fi
done

# ------------------------------------------------------------
# Final summary
# ------------------------------------------------------------

{
    echo "============================================================"
    echo "HSGM ARV_v1 diagnostic pilot summary"
    echo "Finished: $(date)"
    echo "Job token: ${JOB_TOKEN}"
    echo "Model: ${OLLAMA_MODEL}"
    echo "Max steps: ${MAX_STEPS}"
    echo
    echo "Requested episodes (${#EPISODES[@]}):"
    echo "${EPISODES[*]}"
    echo
    echo "Passed (${#PASSED_EPISODES[@]}):"
    if [[ "${#PASSED_EPISODES[@]}" -gt 0 ]]; then
        echo "${PASSED_EPISODES[*]}"
    else
        echo "none"
    fi
    echo
    echo "Failed (${#FAILED_EPISODES[@]}):"
    if [[ "${#FAILED_EPISODES[@]}" -gt 0 ]]; then
        echo "${FAILED_EPISODES[*]}"
    else
        echo "none"
    fi
    echo
    echo "Logs:"
    echo "${LOG_DIR}"
    echo "============================================================"
} | tee "${SUMMARY_LOG}"

# Return non-zero only if EVERY diagnostic episode failed.
if [[ "${#PASSED_EPISODES[@]}" -eq 0 ]]; then
    echo "[ERROR] All diagnostic episodes failed." >&2
    exit 4
fi

echo "[INFO] Diagnostic pilot finished."
