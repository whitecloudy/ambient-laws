#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

NETWORK_PKL=""
OUTPUT_DIR=""
TEST_SNR_RANGE="0,5"
STEPS="1"
TEST_SNR_STEP="1"
SOLVER="euler"
BATCH_SIZE="256"
NPROC_PER_NODE=""
GPUS=""
EXTRA_ARGS=()

usage() {
    cat <<EOF
Usage: $0 --network_pkl PATH --output DIR [OPTIONS] [EXTRA_DENOISING_TESTER_ARGS...]

Required arguments:
  --network_pkl PATH       Path to network checkpoint (.pkl)
  --output DIR             Output directory (saves <split>_<awgn|aniso>_<steps>.csv inside)

Optional arguments (defaults match reference command):
  --test_SNR_range STR     SNR range in dB (default: ${TEST_SNR_RANGE})
  --steps INT              Number of sampling steps (default: ${STEPS})
  --test_SNR_step FLOAT    SNR step size in dB (default: ${TEST_SNR_STEP})
  --solver STR             ODE solver: heun|euler (default: ${SOLVER})
  --batch_size INT         Batch size per process (default: ${BATCH_SIZE})
  --nproc_per_node INT     Number of processes per node (default: GPU count if --gpus set, else 4)
  --gpus STR               Comma-separated GPU IDs for CUDA_VISIBLE_DEVICES (e.g. "0,1,2,3")
  -h, --help               Show this help message and exit

Any additional options (e.g. --bf16, --force_no_nn_scheduler, --must_contain=...) are forwarded to denoising_tester.py.
EOF
}

require_value() {
    if [[ $# -lt 2 || -z "${2}" ]]; then
        echo "Error: Option '${1}' requires a non-empty value." >&2
        usage >&2
        exit 1
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --network_pkl=*)
            NETWORK_PKL="${1#*=}"
            shift
            ;;
        --network_pkl)
            require_value "$1" "${2:-}"
            NETWORK_PKL="$2"
            shift 2
            ;;
        --output=*)
            OUTPUT_DIR="${1#*=}"
            shift
            ;;
        --output)
            require_value "$1" "${2:-}"
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --test_SNR_range=*)
            TEST_SNR_RANGE="${1#*=}"
            shift
            ;;
        --test_SNR_range)
            require_value "$1" "${2:-}"
            TEST_SNR_RANGE="$2"
            shift 2
            ;;
        --steps=*)
            STEPS="${1#*=}"
            shift
            ;;
        --steps)
            require_value "$1" "${2:-}"
            STEPS="$2"
            shift 2
            ;;
        --test_SNR_step=*)
            TEST_SNR_STEP="${1#*=}"
            shift
            ;;
        --test_SNR_step)
            require_value "$1" "${2:-}"
            TEST_SNR_STEP="$2"
            shift 2
            ;;
        --solver=*)
            SOLVER="${1#*=}"
            shift
            ;;
        --solver)
            require_value "$1" "${2:-}"
            SOLVER="$2"
            shift 2
            ;;
        --batch_size=*)
            BATCH_SIZE="${1#*=}"
            shift
            ;;
        --batch_size)
            require_value "$1" "${2:-}"
            BATCH_SIZE="$2"
            shift 2
            ;;
        --nproc_per_node=*)
            NPROC_PER_NODE="${1#*=}"
            shift
            ;;
        --nproc_per_node)
            require_value "$1" "${2:-}"
            NPROC_PER_NODE="$2"
            shift 2
            ;;
        --gpus=*)
            GPUS="${1#*=}"
            shift
            ;;
        --gpus)
            require_value "$1" "${2:-}"
            GPUS="$2"
            shift 2
            ;;
        --allow_tf32)
            # Included by default; ignore duplicate flag in extra args.
            shift
            ;;
        --data|--data=*|--noise_type|--noise_type=*)
            echo "Error: '${1}' is managed automatically across the 4 test/valid x awgn/aniso runs." >&2
            exit 1
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -z "${NETWORK_PKL}" || -z "${OUTPUT_DIR}" ]]; then
    echo "Error: Both --network_pkl and --output are required." >&2
    usage >&2
    exit 1
fi

if [[ -n "${GPUS}" ]]; then
    export CUDA_VISIBLE_DEVICES="${GPUS}"
    if [[ -z "${NPROC_PER_NODE}" ]]; then
        IFS=',' read -ra GPU_ARRAY <<< "${GPUS}"
        NPROC_PER_NODE="${#GPU_ARRAY[@]}"
    fi
elif [[ -z "${NPROC_PER_NODE}" ]]; then
    NPROC_PER_NODE="4"
fi

if [[ "${OUTPUT_DIR}" != "/" ]]; then
    OUTPUT_DIR="${OUTPUT_DIR%/}"
fi
mkdir -p "${OUTPUT_DIR}"

if [[ "${CONDA_DEFAULT_ENV:-}" == "ambient-rf" ]]; then
    PYTHON_CMD=(python)
elif command -v conda >/dev/null 2>&1; then
    PYTHON_CMD=(conda run --no-capture-output -n ambient-rf python)
else
    echo "Error: conda environment 'ambient-rf' is not active and 'conda' command was not found." >&2
    exit 1
fi

TESTER_SCRIPT="denoising_tester.py"
TEST_DATA_REL="data/RENEW/larger_utilize_dataset2/test"
VALID_DATA_REL="data/RENEW/larger_utilize_dataset2/valid"
if [[ ! -f "${TESTER_SCRIPT}" && -f "${REPO_ROOT}/denoising_tester.py" ]]; then
    TESTER_SCRIPT="${REPO_ROOT}/denoising_tester.py"
    TEST_DATA_REL="${REPO_ROOT}/${TEST_DATA_REL}"
    VALID_DATA_REL="${REPO_ROOT}/${VALID_DATA_REL}"
fi

SPLITS=(
    "test:${TEST_DATA_REL}"
    "valid:${VALID_DATA_REL}"
)
NOISE_MODES=(
    "awgn:static"
    "aniso:boosted"
)

for split_entry in "${SPLITS[@]}"; do
    split_name="${split_entry%%:*}"
    data_path="${split_entry#*:}"

    for noise_entry in "${NOISE_MODES[@]}"; do
        noise_label="${noise_entry%%:*}"
        noise_type="${noise_entry#*:}"
        out_csv="${OUTPUT_DIR}/${split_name}_${noise_label}_${STEPS}.csv"

        echo "============================================================"
        echo "Running: split=${split_name}, noise=${noise_label} (${noise_type}), steps=${STEPS}, solver=${SOLVER}"
        echo "  data:   ${data_path}"
        echo "  output: ${out_csv}"
        echo "============================================================"

        "${PYTHON_CMD[@]}" -m torch.distributed.run \
            --standalone \
            --nproc_per_node="${NPROC_PER_NODE}" \
            "${TESTER_SCRIPT}" \
            --network_pkl="${NETWORK_PKL}" \
            --output="${out_csv}" \
            --batch_size="${BATCH_SIZE}" \
            --allow_tf32 \
            --data="${data_path}" \
            --test_SNR_range="${TEST_SNR_RANGE}" \
            --steps="${STEPS}" \
            --test_SNR_step="${TEST_SNR_STEP}" \
            --solver="${SOLVER}" \
            --noise_type="${noise_type}" \
            ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
    done
done

echo "All 4 denoising tests completed. Results saved under: ${OUTPUT_DIR}"
