#!/bin/bash
set -euo pipefail

# Unified GLUE grid runner for the current backend codebase.
# Supports BACKEND in: switch, olmoe, mistral, qwen.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND="${BACKEND:-switch}"
SCRIPT_OVERRIDE="${SCRIPT_OVERRIDE:-}"

case "${BACKEND}" in
    switch)
        SCRIPT_DEFAULT="${ROOT_DIR}/model/dp_switch_transformer.py"
        MODEL_NAME_DEFAULT="google/switch-base-16"
        TASKS_DEFAULT="mnli"
        EXPERIMENTS_DEFAULT="upper_bound upper_bound_no_lora naive_dp naive_dp_no_lora ours"
        SUPPORT_NAIVE_DP_NO_LORA=1
        SUPPORT_DPSFT=1
        SUPPORT_OURS_SEQ2SEQ=1
        SUPPORT_OURS_VER2=1
        ;;
    olmoe)
        SCRIPT_DEFAULT="${ROOT_DIR}/model/dp_olmoe.py"
        MODEL_NAME_DEFAULT="allenai/OLMoE-1B-7B-0924"
        TASKS_DEFAULT="sst2 mnli qnli qqp"
        EXPERIMENTS_DEFAULT="upper_bound upper_bound_no_lora naive_dp naive_dp_no_lora ours"
        SUPPORT_NAIVE_DP_NO_LORA=1
        SUPPORT_DPSFT=0
        SUPPORT_OURS_SEQ2SEQ=0
        SUPPORT_OURS_VER2=0
        ;;
    mistral)
        SCRIPT_DEFAULT="${ROOT_DIR}/model/dp_mistral.py"
        MODEL_NAME_DEFAULT="mistralai/Mixtral-8x7B-v0.1"
        TASKS_DEFAULT="sst2 mnli qnli qqp"
        EXPERIMENTS_DEFAULT="upper_bound upper_bound_no_lora naive_dp naive_dp_no_lora ours"
        SUPPORT_NAIVE_DP_NO_LORA=1
        SUPPORT_DPSFT=0
        SUPPORT_OURS_SEQ2SEQ=0
        SUPPORT_OURS_VER2=0
        ;;
    qwen)
        SCRIPT_DEFAULT="${ROOT_DIR}/model/dp_qwen.py"
        MODEL_NAME_DEFAULT="Qwen/Qwen1.5-MoE-A2.7B"
        TASKS_DEFAULT="sst2 mnli qnli qqp"
        EXPERIMENTS_DEFAULT="upper_bound upper_bound_no_lora naive_dp naive_dp_no_lora ours"
        SUPPORT_NAIVE_DP_NO_LORA=1
        SUPPORT_DPSFT=0
        SUPPORT_OURS_SEQ2SEQ=0
        SUPPORT_OURS_VER2=0
        ;;
    *)
        echo "[ERROR] Unsupported BACKEND='${BACKEND}'. Use one of: switch, olmoe, mistral, qwen." >&2
        exit 1
        ;;
esac

SCRIPT="${SCRIPT_OVERRIDE:-${SCRIPT_DEFAULT}}"
if [[ ! -f "${SCRIPT}" ]]; then
    echo "[ERROR] Script not found: ${SCRIPT}" >&2
    exit 1
fi

MODEL_NAME="${MODEL_NAME:-${MODEL_NAME_DEFAULT}}"
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/data}"
BASE_OUT="${BASE_OUT:-${ROOT_DIR}/results_${BACKEND}}"
LOG_DIR="${LOG_DIR:-${BASE_OUT}/logs}"
FAIL_LOG="${FAIL_LOG:-${BASE_OUT}/failed_runs.txt}"
PROBE_CACHE_DIR="${PROBE_CACHE_DIR:-${BASE_OUT}/probe_cache}"

mkdir -p "${LOG_DIR}" "${PROBE_CACHE_DIR}"
: > "${FAIL_LOG}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

TORCHRUN="${TORCHRUN:-}"
if [[ -z "${TORCHRUN}" ]]; then
    TORCHRUN="$(command -v /home/duc/.conda/envs/duc2/bin/torchrun || true)"
fi
if [[ -z "${TORCHRUN}" && -x "/home/quyennd/miniconda3/envs/duc2/bin//home/duc/.conda/envs/duc2/bin/torchrun" ]]; then
    TORCHRUN="/home/quyennd/miniconda3/envs/duc2/bin//home/duc/.conda/envs/duc2/bin/torchrun"
fi

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
if [[ "${BACKEND}" == "switch" && "${NPROC_PER_NODE}" != "1" ]]; then
    echo "[WARN] BACKEND=switch does not use DDP. Forcing NPROC_PER_NODE=1."
    NPROC_PER_NODE="1"
fi
TORCHRUN_FLAGS=(--standalone --nproc_per_node "${NPROC_PER_NODE}")
if [[ -n "${TORCHRUN_EXTRA_FLAGS:-}" ]]; then
    read -r -a TORCHRUN_EXTRA_ARR <<< "${TORCHRUN_EXTRA_FLAGS}"
    TORCHRUN_FLAGS+=("${TORCHRUN_EXTRA_ARR[@]}")
fi

RUN_WITH_TORCHRUN=1
if [[ -z "${TORCHRUN}" ]]; then
    if [[ "${BACKEND}" == "switch" ]]; then
        RUN_WITH_TORCHRUN=0
        PYTHON_BIN="${PYTHON_BIN:-}"
        if [[ -z "${PYTHON_BIN}" ]]; then
            if [[ -x "/home/quyennd/miniconda3/envs/duc2/bin/python" ]]; then
                PYTHON_BIN="/home/quyennd/miniconda3/envs/duc2/bin/python"
            else
                PYTHON_BIN="$(command -v python3 || true)"
            fi
    fi
        if [[ -z "${PYTHON_BIN}" ]]; then
            echo "[ERROR] /home/duc/.conda/envs/duc2/bin/torchrun not found and python fallback unavailable." >&2
            exit 1
        fi
        echo "[WARN] /home/duc/.conda/envs/duc2/bin/torchrun not found; using python fallback for BACKEND=switch: ${PYTHON_BIN}"
    else
        echo "[ERROR] /home/duc/.conda/envs/duc2/bin/torchrun not found in PATH (required for BACKEND=${BACKEND})." >&2
        exit 1
    fi
fi

if [[ "${RUN_WITH_TORCHRUN}" == "1" ]]; then
    LAUNCH_PREFIX=("${TORCHRUN}" "${TORCHRUN_FLAGS[@]}" "${SCRIPT}")
else
    LAUNCH_PREFIX=("${PYTHON_BIN}" "${SCRIPT}")
fi

read -r -a TASK_LIST <<< "${TASKS:-${TASKS_DEFAULT}}"
read -r -a EPS_LIST <<< "${EPS_LIST:-8}"
read -r -a EXP_LIST_RAW <<< "${EXPERIMENTS:-${EXPERIMENTS_DEFAULT}}"

SHOW_PROGRESS="${SHOW_PROGRESS:-1}"
NO_WANDB="${NO_WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-dp-${BACKEND}-glue}"
ADJACENCY="${ADJACENCY:-add_remove}"
ENABLE_IMPORTANCE_PROBING="${ENABLE_IMPORTANCE_PROBING:-0}"
FREEZE_IN_PHASE_A="${FREEZE_IN_PHASE_A:-router,classifier}"
FREEZE_ROUTER_IN_PHASE_A="${FREEZE_ROUTER_IN_PHASE_A:-0}"
EVAL_EACH_LAYER="${EVAL_EACH_LAYER:-1}"

LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
MAX_LENGTH="${MAX_LENGTH:-128}"
SEED="${SEED:-42}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-50}"
MIN_EXPERT_SIZE="${MIN_EXPERT_SIZE:-50}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
ASSIGNMENT_BATCH_SIZE="${ASSIGNMENT_BATCH_SIZE:-64}"
EVAL_STEPS="${EVAL_STEPS:-50}"
WARMUP_RATIO="${WARMUP_RATIO:-0.06}"
DILUTION_METRIC_BATCHES="${DILUTION_METRIC_BATCHES:-2}"
DILUTION_METRIC_BATCH_SIZE="${DILUTION_METRIC_BATCH_SIZE:-4}"

PHASE_A_RATIO="${PHASE_A_RATIO:-0.8}"
PROBE_METHOD="${PROBE_METHOD:-random_rademacher}"
UPDATE_RATIO="${UPDATE_RATIO:-1.0}"
RANK_MODE="${RANK_MODE:-uniform}"
OURSVER2_WINDOWS="${OURSVER2_WINDOWS:-3}"
OURSVER2_REFRESH_WINDOWS="${OURSVER2_REFRESH_WINDOWS:-all}"

UB_FINETUNE_EPOCHS="${UB_FINETUNE_EPOCHS:-8}"
UB_TRAIN_BATCH_SIZE="${UB_TRAIN_BATCH_SIZE:-1024}"
UB_MICRO_BATCH_SIZE="${UB_MICRO_BATCH_SIZE:-64}"
UB_LR="${UB_LR:-2e-4}"

UB_NOLORA_FINETUNE_EPOCHS="${UB_NOLORA_FINETUNE_EPOCHS:-3}"
UB_NOLORA_TRAIN_BATCH_SIZE="${UB_NOLORA_TRAIN_BATCH_SIZE:-64}"
UB_NOLORA_MICRO_BATCH_SIZE="${UB_NOLORA_MICRO_BATCH_SIZE:-8}"
UB_NOLORA_LR="${UB_NOLORA_LR:-2e-5}"

NAIVE_DP_FINETUNE_EPOCHS="${NAIVE_DP_FINETUNE_EPOCHS:-5}"
NAIVE_DP_TRAIN_BATCH_SIZE="${NAIVE_DP_TRAIN_BATCH_SIZE:-16}"
NAIVE_DP_MICRO_BATCH_SIZE="${NAIVE_DP_MICRO_BATCH_SIZE:-8}"
NAIVE_DP_LR="${NAIVE_DP_LR:-1e-4}"

NAIVE_DP_NOLORA_FINETUNE_EPOCHS="${NAIVE_DP_NOLORA_FINETUNE_EPOCHS:-5}"
NAIVE_DP_NOLORA_TRAIN_BATCH_SIZE="${NAIVE_DP_NOLORA_TRAIN_BATCH_SIZE:-16}"
NAIVE_DP_NOLORA_MICRO_BATCH_SIZE="${NAIVE_DP_NOLORA_MICRO_BATCH_SIZE:-8}"
NAIVE_DP_NOLORA_LR="${NAIVE_DP_NOLORA_LR:-2e-5}"

DPSFT_STAGE1_EPOCHS="${DPSFT_STAGE1_EPOCHS:-2}"
DPSFT_STAGE2_EPOCHS="${DPSFT_STAGE2_EPOCHS:-5}"
DPSFT_TRAIN_BATCH_SIZE="${DPSFT_TRAIN_BATCH_SIZE:-32}"
DPSFT_MICRO_BATCH_SIZE="${DPSFT_MICRO_BATCH_SIZE:-4}"
DPSFT_STAGE1_LR="${DPSFT_STAGE1_LR:-2e-5}"
DPSFT_STAGE2_LR="${DPSFT_STAGE2_LR:-1e-5}"
DPSFT_SUBSPACE_DIM="${DPSFT_SUBSPACE_DIM:-8}"
DPSFT_STAGE1_EPSILON_RATIO="${DPSFT_STAGE1_EPSILON_RATIO:-0.4}"
DPSFT_PARAM_SCOPE="${DPSFT_PARAM_SCOPE:-full_forward}"
DPSFT_STAGE1_CLIP_NORM="${DPSFT_STAGE1_CLIP_NORM:--1}"
DPSFT_STAGE2_CLIP_NORM="${DPSFT_STAGE2_CLIP_NORM:--1}"
DPSFT_GPU_DIRECTIONS_MAX_ELEMENTS="${DPSFT_GPU_DIRECTIONS_MAX_ELEMENTS:-20000000}"

OURS_START_LAYER="${OURS_START_LAYER:--1}"
OURS_NUM_LAYERS="${OURS_NUM_LAYERS:-1}"
OURS_EPOCHS="${OURS_EPOCHS:-3}"
OURS_ATTENTION_EPOCHS="${OURS_ATTENTION_EPOCHS:-${OURS_EPOCHS}}"
OURS_EXPERT_EPOCHS="${OURS_EXPERT_EPOCHS:-${OURS_EPOCHS}}"
OURS_TRAIN_BATCH_SIZE="${OURS_TRAIN_BATCH_SIZE:-1024}"
OURS_MICRO_BATCH_SIZE="${OURS_MICRO_BATCH_SIZE:-32}"
OURS_LR="${OURS_LR:-2e-4}"
OURS_EPSILON_SHARED_RATIO="${OURS_EPSILON_SHARED_RATIO:-0.3}"
OURS_PHASE_A_ONLY="${OURS_PHASE_A_ONLY:-0}"
OURS_EXPERT_RESIDUAL_WEIGHTING="${OURS_EXPERT_RESIDUAL_WEIGHTING:-none}"
OURS_EXPERT_RESIDUAL_MARGIN_THRESHOLD="${OURS_EXPERT_RESIDUAL_MARGIN_THRESHOLD:-0.2}"
OURS_EXPERT_RESIDUAL_SMALL_WEIGHT="${OURS_EXPERT_RESIDUAL_SMALL_WEIGHT:-0.0}"
OURS_EXPERT_RESIDUAL_MIN_WEIGHT="${OURS_EXPERT_RESIDUAL_MIN_WEIGHT:-0.0}"
OURS_EXPERT_UPDATE_SCALE_MODE="${OURS_EXPERT_UPDATE_SCALE_MODE:-batch}"
OURS_CLIP_SCOPE="${OURS_CLIP_SCOPE:-role}"
OURS_EXPERT_LR_MULTIPLIER="${OURS_EXPERT_LR_MULTIPLIER:-1.0}"
OURS_EXPERT_OBJECTIVE="${OURS_EXPERT_OBJECTIVE:-ce}"
OURS_ALTERNATING_SHARED_STEPS="${OURS_ALTERNATING_SHARED_STEPS:-0}"
OURS_ALTERNATING_EXPERT_STEPS="${OURS_ALTERNATING_EXPERT_STEPS:-0}"

BASELINE_START_LAYER="${BASELINE_START_LAYER:--1}"
BASELINE_NUM_LAYERS="${BASELINE_NUM_LAYERS:-1}"
BASELINE_EPOCHS="${BASELINE_EPOCHS:-${OURS_EPOCHS}}"
BASELINE_TRAIN_BATCH_SIZE="${BASELINE_TRAIN_BATCH_SIZE:-${OURS_TRAIN_BATCH_SIZE}}"
BASELINE_MICRO_BATCH_SIZE="${BASELINE_MICRO_BATCH_SIZE:-${OURS_MICRO_BATCH_SIZE}}"
BASELINE_LR="${BASELINE_LR:-${OURS_LR}}"
BASELINE_EPSILON_SHARED_RATIO="${BASELINE_EPSILON_SHARED_RATIO:-${OURS_EPSILON_SHARED_RATIO}}"
BASELINE_SHARED_EPS_MODE="${BASELINE_SHARED_EPS_MODE:-full}"
BASELINE_EXPERT_UPDATE_SCALE_MODE="${BASELINE_EXPERT_UPDATE_SCALE_MODE:-expected_owner}"
BASELINE_CLIP_SCOPE="${BASELINE_CLIP_SCOPE:-role}"
BASELINE_EXPERT_LR_MULTIPLIER="${BASELINE_EXPERT_LR_MULTIPLIER:-${OURS_EXPERT_LR_MULTIPLIER}}"
BASELINE_EXPERT_OBJECTIVE="${BASELINE_EXPERT_OBJECTIVE:-ce}"
BASELINE_ALTERNATING_SHARED_STEPS="${BASELINE_ALTERNATING_SHARED_STEPS:-0}"
BASELINE_ALTERNATING_EXPERT_STEPS="${BASELINE_ALTERNATING_EXPERT_STEPS:-0}"

LAST_LAYER_START_LAYER="${LAST_LAYER_START_LAYER:--1}"
LAST_LAYER_NUM_LAYERS="${LAST_LAYER_NUM_LAYERS:-1}"
LAST_LAYER_EPOCHS="${LAST_LAYER_EPOCHS:-${BASELINE_EPOCHS}}"
LAST_LAYER_TRAIN_BATCH_SIZE="${LAST_LAYER_TRAIN_BATCH_SIZE:-${BASELINE_TRAIN_BATCH_SIZE}}"
LAST_LAYER_MICRO_BATCH_SIZE="${LAST_LAYER_MICRO_BATCH_SIZE:-${BASELINE_MICRO_BATCH_SIZE}}"
LAST_LAYER_LR="${LAST_LAYER_LR:-${BASELINE_LR}}"

# Grid lists (space-separated); defaults preserve single-run behavior.
UB_FINETUNE_EPOCHS_GRID="${UB_FINETUNE_EPOCHS_GRID:-${UB_FINETUNE_EPOCHS}}"
UB_TRAIN_BATCH_SIZE_GRID="${UB_TRAIN_BATCH_SIZE_GRID:-${UB_TRAIN_BATCH_SIZE}}"
UB_MICRO_BATCH_SIZE_GRID="${UB_MICRO_BATCH_SIZE_GRID:-${UB_MICRO_BATCH_SIZE}}"
UB_LR_GRID="${UB_LR_GRID:-${UB_LR}}"

UB_NOLORA_FINETUNE_EPOCHS_GRID="${UB_NOLORA_FINETUNE_EPOCHS_GRID:-${UB_NOLORA_FINETUNE_EPOCHS}}"
UB_NOLORA_TRAIN_BATCH_SIZE_GRID="${UB_NOLORA_TRAIN_BATCH_SIZE_GRID:-${UB_NOLORA_TRAIN_BATCH_SIZE}}"
UB_NOLORA_MICRO_BATCH_SIZE_GRID="${UB_NOLORA_MICRO_BATCH_SIZE_GRID:-${UB_NOLORA_MICRO_BATCH_SIZE}}"
UB_NOLORA_LR_GRID="${UB_NOLORA_LR_GRID:-${UB_NOLORA_LR}}"

NAIVE_DP_FINETUNE_EPOCHS_GRID="${NAIVE_DP_FINETUNE_EPOCHS_GRID:-${NAIVE_DP_FINETUNE_EPOCHS}}"
NAIVE_DP_TRAIN_BATCH_SIZE_GRID="${NAIVE_DP_TRAIN_BATCH_SIZE_GRID:-${NAIVE_DP_TRAIN_BATCH_SIZE}}"
NAIVE_DP_MICRO_BATCH_SIZE_GRID="${NAIVE_DP_MICRO_BATCH_SIZE_GRID:-${NAIVE_DP_MICRO_BATCH_SIZE}}"
NAIVE_DP_LR_GRID="${NAIVE_DP_LR_GRID:-${NAIVE_DP_LR}}"

NAIVE_DP_NOLORA_FINETUNE_EPOCHS_GRID="${NAIVE_DP_NOLORA_FINETUNE_EPOCHS_GRID:-${NAIVE_DP_NOLORA_FINETUNE_EPOCHS}}"
NAIVE_DP_NOLORA_TRAIN_BATCH_SIZE_GRID="${NAIVE_DP_NOLORA_TRAIN_BATCH_SIZE_GRID:-${NAIVE_DP_NOLORA_TRAIN_BATCH_SIZE}}"
NAIVE_DP_NOLORA_MICRO_BATCH_SIZE_GRID="${NAIVE_DP_NOLORA_MICRO_BATCH_SIZE_GRID:-${NAIVE_DP_NOLORA_MICRO_BATCH_SIZE}}"
NAIVE_DP_NOLORA_LR_GRID="${NAIVE_DP_NOLORA_LR_GRID:-${NAIVE_DP_NOLORA_LR}}"

DPSFT_STAGE1_EPOCHS_GRID="${DPSFT_STAGE1_EPOCHS_GRID:-${DPSFT_STAGE1_EPOCHS}}"
DPSFT_STAGE2_EPOCHS_GRID="${DPSFT_STAGE2_EPOCHS_GRID:-${DPSFT_STAGE2_EPOCHS}}"
DPSFT_TRAIN_BATCH_SIZE_GRID="${DPSFT_TRAIN_BATCH_SIZE_GRID:-${DPSFT_TRAIN_BATCH_SIZE}}"
DPSFT_MICRO_BATCH_SIZE_GRID="${DPSFT_MICRO_BATCH_SIZE_GRID:-${DPSFT_MICRO_BATCH_SIZE}}"
DPSFT_STAGE1_LR_GRID="${DPSFT_STAGE1_LR_GRID:-${DPSFT_STAGE1_LR}}"
DPSFT_STAGE2_LR_GRID="${DPSFT_STAGE2_LR_GRID:-${DPSFT_STAGE2_LR}}"
DPSFT_SUBSPACE_DIM_GRID="${DPSFT_SUBSPACE_DIM_GRID:-${DPSFT_SUBSPACE_DIM}}"
DPSFT_STAGE1_EPSILON_RATIO_GRID="${DPSFT_STAGE1_EPSILON_RATIO_GRID:-${DPSFT_STAGE1_EPSILON_RATIO}}"
DPSFT_PARAM_SCOPE_GRID="${DPSFT_PARAM_SCOPE_GRID:-${DPSFT_PARAM_SCOPE}}"
DPSFT_STAGE1_CLIP_NORM_GRID="${DPSFT_STAGE1_CLIP_NORM_GRID:-${DPSFT_STAGE1_CLIP_NORM}}"
DPSFT_STAGE2_CLIP_NORM_GRID="${DPSFT_STAGE2_CLIP_NORM_GRID:-${DPSFT_STAGE2_CLIP_NORM}}"

OURS_NUM_LAYERS_GRID="${OURS_NUM_LAYERS_GRID:-${OURS_NUM_LAYERS}}"
OURS_START_LAYER_GRID="${OURS_START_LAYER_GRID:-${OURS_START_LAYER}}"
OURS_EPOCHS_GRID="${OURS_EPOCHS_GRID:-${OURS_EPOCHS}}"
OURS_ATTENTION_EPOCHS_GRID="${OURS_ATTENTION_EPOCHS_GRID:-${OURS_ATTENTION_EPOCHS}}"
OURS_EXPERT_EPOCHS_GRID="${OURS_EXPERT_EPOCHS_GRID:-${OURS_EXPERT_EPOCHS}}"
OURS_TRAIN_BATCH_SIZE_GRID="${OURS_TRAIN_BATCH_SIZE_GRID:-${OURS_TRAIN_BATCH_SIZE}}"
OURS_MICRO_BATCH_SIZE_GRID="${OURS_MICRO_BATCH_SIZE_GRID:-${OURS_MICRO_BATCH_SIZE}}"
OURS_LR_GRID="${OURS_LR_GRID:-${OURS_LR}}"
OURS_EPSILON_SHARED_RATIO_GRID="${OURS_EPSILON_SHARED_RATIO_GRID:-${OURS_EPSILON_SHARED_RATIO}}"
OURS_EXPERT_RESIDUAL_WEIGHTING_GRID="${OURS_EXPERT_RESIDUAL_WEIGHTING_GRID:-${OURS_EXPERT_RESIDUAL_WEIGHTING}}"
OURS_EXPERT_RESIDUAL_MARGIN_THRESHOLD_GRID="${OURS_EXPERT_RESIDUAL_MARGIN_THRESHOLD_GRID:-${OURS_EXPERT_RESIDUAL_MARGIN_THRESHOLD}}"
OURS_EXPERT_RESIDUAL_SMALL_WEIGHT_GRID="${OURS_EXPERT_RESIDUAL_SMALL_WEIGHT_GRID:-${OURS_EXPERT_RESIDUAL_SMALL_WEIGHT}}"
OURS_EXPERT_RESIDUAL_MIN_WEIGHT_GRID="${OURS_EXPERT_RESIDUAL_MIN_WEIGHT_GRID:-${OURS_EXPERT_RESIDUAL_MIN_WEIGHT}}"
OURS_EXPERT_UPDATE_SCALE_MODE_GRID="${OURS_EXPERT_UPDATE_SCALE_MODE_GRID:-${OURS_EXPERT_UPDATE_SCALE_MODE}}"
OURS_CLIP_SCOPE_GRID="${OURS_CLIP_SCOPE_GRID:-${OURS_CLIP_SCOPE}}"
OURS_EXPERT_LR_MULTIPLIER_GRID="${OURS_EXPERT_LR_MULTIPLIER_GRID:-${OURS_EXPERT_LR_MULTIPLIER}}"
OURS_EXPERT_OBJECTIVE_GRID="${OURS_EXPERT_OBJECTIVE_GRID:-${OURS_EXPERT_OBJECTIVE}}"
OURS_ALTERNATING_SHARED_STEPS_GRID="${OURS_ALTERNATING_SHARED_STEPS_GRID:-${OURS_ALTERNATING_SHARED_STEPS}}"
OURS_ALTERNATING_EXPERT_STEPS_GRID="${OURS_ALTERNATING_EXPERT_STEPS_GRID:-${OURS_ALTERNATING_EXPERT_STEPS}}"
OURS_PROBE_METHOD_GRID="${OURS_PROBE_METHOD_GRID:-${PROBE_METHOD}}"
OURS_UPDATE_RATIO_GRID="${OURS_UPDATE_RATIO_GRID:-${UPDATE_RATIO}}"
OURS_RANK_MODE_GRID="${OURS_RANK_MODE_GRID:-${RANK_MODE}}"
OURS_PHASE_A_RATIO_GRID="${OURS_PHASE_A_RATIO_GRID:-${PHASE_A_RATIO}}"

BASELINE_START_LAYER_GRID="${BASELINE_START_LAYER_GRID:-${BASELINE_START_LAYER}}"
BASELINE_NUM_LAYERS_GRID="${BASELINE_NUM_LAYERS_GRID:-${BASELINE_NUM_LAYERS}}"
BASELINE_EPOCHS_GRID="${BASELINE_EPOCHS_GRID:-${BASELINE_EPOCHS}}"
BASELINE_TRAIN_BATCH_SIZE_GRID="${BASELINE_TRAIN_BATCH_SIZE_GRID:-${BASELINE_TRAIN_BATCH_SIZE}}"
BASELINE_MICRO_BATCH_SIZE_GRID="${BASELINE_MICRO_BATCH_SIZE_GRID:-${BASELINE_MICRO_BATCH_SIZE}}"
BASELINE_LR_GRID="${BASELINE_LR_GRID:-${BASELINE_LR}}"
BASELINE_EPSILON_SHARED_RATIO_GRID="${BASELINE_EPSILON_SHARED_RATIO_GRID:-${BASELINE_EPSILON_SHARED_RATIO}}"
BASELINE_SHARED_EPS_MODE_GRID="${BASELINE_SHARED_EPS_MODE_GRID:-${BASELINE_SHARED_EPS_MODE}}"
BASELINE_EXPERT_UPDATE_SCALE_MODE_GRID="${BASELINE_EXPERT_UPDATE_SCALE_MODE_GRID:-${BASELINE_EXPERT_UPDATE_SCALE_MODE}}"
BASELINE_CLIP_SCOPE_GRID="${BASELINE_CLIP_SCOPE_GRID:-${BASELINE_CLIP_SCOPE}}"
BASELINE_EXPERT_LR_MULTIPLIER_GRID="${BASELINE_EXPERT_LR_MULTIPLIER_GRID:-${BASELINE_EXPERT_LR_MULTIPLIER}}"
BASELINE_EXPERT_OBJECTIVE_GRID="${BASELINE_EXPERT_OBJECTIVE_GRID:-${BASELINE_EXPERT_OBJECTIVE}}"
BASELINE_ALTERNATING_SHARED_STEPS_GRID="${BASELINE_ALTERNATING_SHARED_STEPS_GRID:-${BASELINE_ALTERNATING_SHARED_STEPS}}"
BASELINE_ALTERNATING_EXPERT_STEPS_GRID="${BASELINE_ALTERNATING_EXPERT_STEPS_GRID:-${BASELINE_ALTERNATING_EXPERT_STEPS}}"

LAST_LAYER_START_LAYER_GRID="${LAST_LAYER_START_LAYER_GRID:-${LAST_LAYER_START_LAYER}}"
LAST_LAYER_NUM_LAYERS_GRID="${LAST_LAYER_NUM_LAYERS_GRID:-${LAST_LAYER_NUM_LAYERS}}"
LAST_LAYER_EPOCHS_GRID="${LAST_LAYER_EPOCHS_GRID:-${LAST_LAYER_EPOCHS}}"
LAST_LAYER_TRAIN_BATCH_SIZE_GRID="${LAST_LAYER_TRAIN_BATCH_SIZE_GRID:-${LAST_LAYER_TRAIN_BATCH_SIZE}}"
LAST_LAYER_MICRO_BATCH_SIZE_GRID="${LAST_LAYER_MICRO_BATCH_SIZE_GRID:-${LAST_LAYER_MICRO_BATCH_SIZE}}"
LAST_LAYER_LR_GRID="${LAST_LAYER_LR_GRID:-${LAST_LAYER_LR}}"

read -r -a UB_FINETUNE_EPOCHS_LIST <<< "${UB_FINETUNE_EPOCHS_GRID}"
read -r -a UB_TRAIN_BATCH_SIZE_LIST <<< "${UB_TRAIN_BATCH_SIZE_GRID}"
read -r -a UB_MICRO_BATCH_SIZE_LIST <<< "${UB_MICRO_BATCH_SIZE_GRID}"
read -r -a UB_LR_LIST <<< "${UB_LR_GRID}"

read -r -a UB_NOLORA_FINETUNE_EPOCHS_LIST <<< "${UB_NOLORA_FINETUNE_EPOCHS_GRID}"
read -r -a UB_NOLORA_TRAIN_BATCH_SIZE_LIST <<< "${UB_NOLORA_TRAIN_BATCH_SIZE_GRID}"
read -r -a UB_NOLORA_MICRO_BATCH_SIZE_LIST <<< "${UB_NOLORA_MICRO_BATCH_SIZE_GRID}"
read -r -a UB_NOLORA_LR_LIST <<< "${UB_NOLORA_LR_GRID}"

read -r -a NAIVE_DP_FINETUNE_EPOCHS_LIST <<< "${NAIVE_DP_FINETUNE_EPOCHS_GRID}"
read -r -a NAIVE_DP_TRAIN_BATCH_SIZE_LIST <<< "${NAIVE_DP_TRAIN_BATCH_SIZE_GRID}"
read -r -a NAIVE_DP_MICRO_BATCH_SIZE_LIST <<< "${NAIVE_DP_MICRO_BATCH_SIZE_GRID}"
read -r -a NAIVE_DP_LR_LIST <<< "${NAIVE_DP_LR_GRID}"

read -r -a NAIVE_DP_NOLORA_FINETUNE_EPOCHS_LIST <<< "${NAIVE_DP_NOLORA_FINETUNE_EPOCHS_GRID}"
read -r -a NAIVE_DP_NOLORA_TRAIN_BATCH_SIZE_LIST <<< "${NAIVE_DP_NOLORA_TRAIN_BATCH_SIZE_GRID}"
read -r -a NAIVE_DP_NOLORA_MICRO_BATCH_SIZE_LIST <<< "${NAIVE_DP_NOLORA_MICRO_BATCH_SIZE_GRID}"
read -r -a NAIVE_DP_NOLORA_LR_LIST <<< "${NAIVE_DP_NOLORA_LR_GRID}"

read -r -a DPSFT_STAGE1_EPOCHS_LIST <<< "${DPSFT_STAGE1_EPOCHS_GRID}"
read -r -a DPSFT_STAGE2_EPOCHS_LIST <<< "${DPSFT_STAGE2_EPOCHS_GRID}"
read -r -a DPSFT_TRAIN_BATCH_SIZE_LIST <<< "${DPSFT_TRAIN_BATCH_SIZE_GRID}"
read -r -a DPSFT_MICRO_BATCH_SIZE_LIST <<< "${DPSFT_MICRO_BATCH_SIZE_GRID}"
read -r -a DPSFT_STAGE1_LR_LIST <<< "${DPSFT_STAGE1_LR_GRID}"
read -r -a DPSFT_STAGE2_LR_LIST <<< "${DPSFT_STAGE2_LR_GRID}"
read -r -a DPSFT_SUBSPACE_DIM_LIST <<< "${DPSFT_SUBSPACE_DIM_GRID}"
read -r -a DPSFT_STAGE1_EPSILON_RATIO_LIST <<< "${DPSFT_STAGE1_EPSILON_RATIO_GRID}"
read -r -a DPSFT_PARAM_SCOPE_LIST <<< "${DPSFT_PARAM_SCOPE_GRID}"
read -r -a DPSFT_STAGE1_CLIP_NORM_LIST <<< "${DPSFT_STAGE1_CLIP_NORM_GRID}"
read -r -a DPSFT_STAGE2_CLIP_NORM_LIST <<< "${DPSFT_STAGE2_CLIP_NORM_GRID}"

read -r -a OURS_NUM_LAYERS_LIST <<< "${OURS_NUM_LAYERS_GRID}"
read -r -a OURS_START_LAYER_LIST <<< "${OURS_START_LAYER_GRID}"
read -r -a OURS_EPOCHS_LIST <<< "${OURS_EPOCHS_GRID}"
read -r -a OURS_ATTENTION_EPOCHS_LIST <<< "${OURS_ATTENTION_EPOCHS_GRID}"
read -r -a OURS_EXPERT_EPOCHS_LIST <<< "${OURS_EXPERT_EPOCHS_GRID}"
read -r -a OURS_TRAIN_BATCH_SIZE_LIST <<< "${OURS_TRAIN_BATCH_SIZE_GRID}"
read -r -a OURS_MICRO_BATCH_SIZE_LIST <<< "${OURS_MICRO_BATCH_SIZE_GRID}"
read -r -a OURS_LR_LIST <<< "${OURS_LR_GRID}"
read -r -a OURS_EPSILON_SHARED_RATIO_LIST <<< "${OURS_EPSILON_SHARED_RATIO_GRID}"
read -r -a OURS_EXPERT_RESIDUAL_WEIGHTING_LIST <<< "${OURS_EXPERT_RESIDUAL_WEIGHTING_GRID}"
read -r -a OURS_EXPERT_RESIDUAL_MARGIN_THRESHOLD_LIST <<< "${OURS_EXPERT_RESIDUAL_MARGIN_THRESHOLD_GRID}"
read -r -a OURS_EXPERT_RESIDUAL_SMALL_WEIGHT_LIST <<< "${OURS_EXPERT_RESIDUAL_SMALL_WEIGHT_GRID}"
read -r -a OURS_EXPERT_RESIDUAL_MIN_WEIGHT_LIST <<< "${OURS_EXPERT_RESIDUAL_MIN_WEIGHT_GRID}"
read -r -a OURS_EXPERT_UPDATE_SCALE_MODE_LIST <<< "${OURS_EXPERT_UPDATE_SCALE_MODE_GRID}"
read -r -a OURS_CLIP_SCOPE_LIST <<< "${OURS_CLIP_SCOPE_GRID}"
read -r -a OURS_EXPERT_LR_MULTIPLIER_LIST <<< "${OURS_EXPERT_LR_MULTIPLIER_GRID}"
read -r -a OURS_EXPERT_OBJECTIVE_LIST <<< "${OURS_EXPERT_OBJECTIVE_GRID}"
read -r -a OURS_ALTERNATING_SHARED_STEPS_LIST <<< "${OURS_ALTERNATING_SHARED_STEPS_GRID}"
read -r -a OURS_ALTERNATING_EXPERT_STEPS_LIST <<< "${OURS_ALTERNATING_EXPERT_STEPS_GRID}"
read -r -a OURS_PROBE_METHOD_LIST <<< "${OURS_PROBE_METHOD_GRID}"
read -r -a OURS_UPDATE_RATIO_LIST <<< "${OURS_UPDATE_RATIO_GRID}"
read -r -a OURS_RANK_MODE_LIST <<< "${OURS_RANK_MODE_GRID}"
read -r -a OURS_PHASE_A_RATIO_LIST <<< "${OURS_PHASE_A_RATIO_GRID}"

read -r -a BASELINE_START_LAYER_LIST <<< "${BASELINE_START_LAYER_GRID}"
read -r -a BASELINE_NUM_LAYERS_LIST <<< "${BASELINE_NUM_LAYERS_GRID}"
read -r -a BASELINE_EPOCHS_LIST <<< "${BASELINE_EPOCHS_GRID}"
read -r -a BASELINE_TRAIN_BATCH_SIZE_LIST <<< "${BASELINE_TRAIN_BATCH_SIZE_GRID}"
read -r -a BASELINE_MICRO_BATCH_SIZE_LIST <<< "${BASELINE_MICRO_BATCH_SIZE_GRID}"
read -r -a BASELINE_LR_LIST <<< "${BASELINE_LR_GRID}"
read -r -a BASELINE_EPSILON_SHARED_RATIO_LIST <<< "${BASELINE_EPSILON_SHARED_RATIO_GRID}"
read -r -a BASELINE_SHARED_EPS_MODE_LIST <<< "${BASELINE_SHARED_EPS_MODE_GRID}"
read -r -a BASELINE_EXPERT_UPDATE_SCALE_MODE_LIST <<< "${BASELINE_EXPERT_UPDATE_SCALE_MODE_GRID}"
read -r -a BASELINE_CLIP_SCOPE_LIST <<< "${BASELINE_CLIP_SCOPE_GRID}"
read -r -a BASELINE_EXPERT_LR_MULTIPLIER_LIST <<< "${BASELINE_EXPERT_LR_MULTIPLIER_GRID}"
read -r -a BASELINE_EXPERT_OBJECTIVE_LIST <<< "${BASELINE_EXPERT_OBJECTIVE_GRID}"
read -r -a BASELINE_ALTERNATING_SHARED_STEPS_LIST <<< "${BASELINE_ALTERNATING_SHARED_STEPS_GRID}"
read -r -a BASELINE_ALTERNATING_EXPERT_STEPS_LIST <<< "${BASELINE_ALTERNATING_EXPERT_STEPS_GRID}"

read -r -a LAST_LAYER_START_LAYER_LIST <<< "${LAST_LAYER_START_LAYER_GRID}"
read -r -a LAST_LAYER_NUM_LAYERS_LIST <<< "${LAST_LAYER_NUM_LAYERS_GRID}"
read -r -a LAST_LAYER_EPOCHS_LIST <<< "${LAST_LAYER_EPOCHS_GRID}"
read -r -a LAST_LAYER_TRAIN_BATCH_SIZE_LIST <<< "${LAST_LAYER_TRAIN_BATCH_SIZE_GRID}"
read -r -a LAST_LAYER_MICRO_BATCH_SIZE_LIST <<< "${LAST_LAYER_MICRO_BATCH_SIZE_GRID}"
read -r -a LAST_LAYER_LR_LIST <<< "${LAST_LAYER_LR_GRID}"

has_item() {
    local needle="$1"
    shift
    local item
    for item in "$@"; do
        if [[ "${item}" == "${needle}" ]]; then
            return 0
        fi
    done
    return 1
}

is_supported_task() {
    local t="$1"
    case "${BACKEND}" in
        switch|olmoe|mistral|qwen) [[ "${t}" == "sst2" || "${t}" == "mnli" || "${t}" == "qnli" || "${t}" == "qqp" ]] ;;
        *) return 1 ;;
    esac
}

is_supported_experiment() {
    local e="$1"
    case "${e}" in
        upper_bound|upper_bound_no_lora|naive_dp|ours) return 0 ;;
        naive_dp_no_lora) [[ "${SUPPORT_NAIVE_DP_NO_LORA}" == "1" ]] ;;
        baseline_a_global_matched|baseline_b_shared_only|baseline_c_expert_only|baseline_e_selected_layer_lora|baseline_e_last_layer_lora|matched_scope_global_dp|dp_shared_only|dp_expert_only|dp_lora_last_layer|last_layer_dp_lora_global) [[ "${BACKEND}" == "switch" || "${BACKEND}" == "olmoe" ]] ;;
        dpsft) [[ "${SUPPORT_DPSFT}" == "1" ]] ;;
        oursseq2seq) [[ "${SUPPORT_OURS_SEQ2SEQ}" == "1" ]] ;;
        oursver2) [[ "${SUPPORT_OURS_VER2}" == "1" ]] ;;
        *) return 1 ;;
    esac
}

calc_eps_attention() {
    python3 -c "eps=float('${1}'); ratio=float('${2}'); print(round(eps * ratio, 7))"
}

sanitize_val() {
    local val="$1"
    val="${val//\//_}"
    val="${val// /}"
    val="${val//./p}"
    echo "${val}"
}

build_switch_ours_extra_args() {
    local -n out_ref="$1"
    out_ref=()
    if [[ "${BACKEND}" != "switch" && "${BACKEND}" != "olmoe" ]]; then
        return
    fi
    out_ref=(
        --expert_residual_weighting "${OURS_EXPERT_RESIDUAL_WEIGHTING}"
        --expert_residual_margin_threshold "${OURS_EXPERT_RESIDUAL_MARGIN_THRESHOLD}"
        --expert_residual_small_weight "${OURS_EXPERT_RESIDUAL_SMALL_WEIGHT}"
        --expert_residual_min_weight "${OURS_EXPERT_RESIDUAL_MIN_WEIGHT}"
        --expert_update_scale_mode "${OURS_EXPERT_UPDATE_SCALE_MODE}"
        --ours_clip_scope "${OURS_CLIP_SCOPE}"
        --expert_lr_multiplier "${OURS_EXPERT_LR_MULTIPLIER}"
        --expert_objective "${OURS_EXPERT_OBJECTIVE}"
        --alternating_shared_steps "${OURS_ALTERNATING_SHARED_STEPS}"
        --alternating_expert_steps "${OURS_ALTERNATING_EXPERT_STEPS}"
    )
}

build_switch_ours_extra_tag() {
    local tag=""
    if [[ "${BACKEND}" == "switch" || "${BACKEND}" == "olmoe" ]]; then
        if [[ "${OURS_EXPERT_RESIDUAL_WEIGHTING}" != "none" ]]; then
            tag="${tag}_res$(sanitize_val "${OURS_EXPERT_RESIDUAL_WEIGHTING}")"
            tag="${tag}_rthr$(sanitize_val "${OURS_EXPERT_RESIDUAL_MARGIN_THRESHOLD}")"
            tag="${tag}_rsw$(sanitize_val "${OURS_EXPERT_RESIDUAL_SMALL_WEIGHT}")"
            tag="${tag}_rmin$(sanitize_val "${OURS_EXPERT_RESIDUAL_MIN_WEIGHT}")"
        fi
        if [[ "${OURS_EXPERT_UPDATE_SCALE_MODE}" != "batch" ]]; then
            tag="${tag}_den$(sanitize_val "${OURS_EXPERT_UPDATE_SCALE_MODE}")"
        fi
        if [[ "${OURS_CLIP_SCOPE}" != "role" ]]; then
            tag="${tag}_clip$(sanitize_val "${OURS_CLIP_SCOPE}")"
        fi
        if [[ "${OURS_EXPERT_LR_MULTIPLIER}" != "1.0" && "${OURS_EXPERT_LR_MULTIPLIER}" != "1" ]]; then
            tag="${tag}_elm$(sanitize_val "${OURS_EXPERT_LR_MULTIPLIER}")"
        fi
        if [[ "${OURS_EXPERT_OBJECTIVE}" != "ce" ]]; then
            tag="${tag}_obj$(sanitize_val "${OURS_EXPERT_OBJECTIVE}")"
        fi
        if [[ "${OURS_ALTERNATING_EXPERT_STEPS}" != "0" ]]; then
            tag="${tag}_altS$(sanitize_val "${OURS_ALTERNATING_SHARED_STEPS}")E$(sanitize_val "${OURS_ALTERNATING_EXPERT_STEPS}")"
        fi
    fi
    echo "${tag}"
}

build_common_args() {
    local task="$1"
    local -n out_ref="$2"

    out_ref=(
        --model_name "${MODEL_NAME}"
        --task "${task}"
        --glue_data_root "${DATA_ROOT}"
        --lora_r "${LORA_R}"
        --lora_alpha "${LORA_ALPHA}"
        --lora_dropout "${LORA_DROPOUT}"
        --max_length "${MAX_LENGTH}"
        --max_grad_norm 1.0
        --delta -1
        --adjacency "${ADJACENCY}"
        --seed "${SEED}"
        --beta1 0.9
        --beta2 0.999
        --weight_decay 0.01
        --early_stop_patience "${EARLY_STOP_PATIENCE}"
        --min_expert_size "${MIN_EXPERT_SIZE}"
        --clipping_mode MixOpt
        --clipping_fn automatic
        --clipping_style layer-wise
        --eval_batch_size "${EVAL_BATCH_SIZE}"
        --assignment_batch_size "${ASSIGNMENT_BATCH_SIZE}"
        --eval_steps "${EVAL_STEPS}"
        --warmup_ratio "${WARMUP_RATIO}"
        --dilution_metric_batches "${DILUTION_METRIC_BATCHES}"
        --dilution_metric_batch_size "${DILUTION_METRIC_BATCH_SIZE}"
    )

    if [[ "${SHOW_PROGRESS}" == "1" ]]; then
        out_ref+=(--show_progress)
    fi

    if [[ "${NO_WANDB}" == "1" ]]; then
        out_ref+=(--no_wandb)
    else
        out_ref+=(--wandb_project "${WANDB_PROJECT}")
    fi
}

build_phase_a_freeze_args() {
    local -n out_ref="$1"
    out_ref=()
    local items=()
    local spec="${FREEZE_IN_PHASE_A//,/ }"
    if [[ -n "${spec// }" ]]; then
        read -r -a items <<< "${spec}"
    fi
    if [[ "${FREEZE_ROUTER_IN_PHASE_A}" == "1" ]] && ! has_item router "${items[@]}"; then
        items+=(router)
    fi
    if [[ "${#items[@]}" -gt 0 ]]; then
        out_ref=(--freeze_in_phase_a "${items[@]}")
    fi
}

run_exp() {
    local tag="$1"
    local log="${LOG_DIR}/${tag//\//_}.log"
    shift

    echo ""
    echo "============================================================"
    echo "BACKEND: ${BACKEND}"
    echo "START:   ${tag}"
    echo "LOG:     ${log}"
    echo "============================================================"

    if "$@" > "${log}" 2>&1; then
        echo "DONE:    ${tag}"
    else
        echo "FAILED:  ${tag}  (see ${log})"
        echo "${tag}" >> "${FAIL_LOG}"
    fi
}

# Filter invalid tasks/experiments up front.
VALID_TASK_LIST=()
for t in "${TASK_LIST[@]}"; do
    if is_supported_task "${t}"; then
        VALID_TASK_LIST+=("${t}")
    else
        echo "[WARN] Skipping unsupported task '${t}' for BACKEND='${BACKEND}'"
    fi
done
if [[ "${#VALID_TASK_LIST[@]}" -eq 0 ]]; then
    echo "[ERROR] No valid tasks left after filtering for BACKEND='${BACKEND}'." >&2
    exit 1
fi

VALID_EXP_LIST=()
for e in "${EXP_LIST_RAW[@]}"; do
    if is_supported_experiment "${e}"; then
        VALID_EXP_LIST+=("${e}")
    else
        echo "[WARN] Skipping unsupported experiment '${e}' for BACKEND='${BACKEND}'"
    fi
done
if [[ "${#VALID_EXP_LIST[@]}" -eq 0 ]]; then
    echo "[ERROR] No valid experiments left after filtering for BACKEND='${BACKEND}'." >&2
    exit 1
fi

echo "============================================================"
echo "Runner:    ${0}"
echo "Backend:   ${BACKEND}"
echo "Script:    ${SCRIPT}"
echo "Model:     ${MODEL_NAME}"
echo "Tasks:     ${VALID_TASK_LIST[*]}"
echo "Exps:      ${VALID_EXP_LIST[*]}"
echo "Output:    ${BASE_OUT}"
if [[ "${RUN_WITH_TORCHRUN}" == "1" ]]; then
    echo "Launch:    ${TORCHRUN} ${TORCHRUN_FLAGS[*]} ${SCRIPT}"
else
    echo "Launch:    ${PYTHON_BIN} ${SCRIPT}"
fi
echo "============================================================"

for TASK in "${VALID_TASK_LIST[@]}"; do
    echo ""
    echo "############################################################"
    echo "TASK: ${TASK}"
    echo "############################################################"

    OUT="${BASE_OUT}/${TASK}"
    mkdir -p "${OUT}"

    build_common_args "${TASK}" COMMON_ARGS

    if has_item upper_bound "${VALID_EXP_LIST[@]}"; then
        for UB_EPOCHS_CUR in "${UB_FINETUNE_EPOCHS_LIST[@]}"; do
            for UB_BATCH_CUR in "${UB_TRAIN_BATCH_SIZE_LIST[@]}"; do
                for UB_MICRO_CUR in "${UB_MICRO_BATCH_SIZE_LIST[@]}"; do
                    for UB_LR_CUR in "${UB_LR_LIST[@]}"; do
                        UB_TAG="${TASK}/upper_bound_ep${UB_EPOCHS_CUR}_bs${UB_BATCH_CUR}_mb${UB_MICRO_CUR}_lr$(sanitize_val "${UB_LR_CUR}")"
                        run_exp "${UB_TAG}" \
                            "${LAUNCH_PREFIX[@]}" \
                            "${COMMON_ARGS[@]}" \
                            --experiment upper_bound \
                            --output_dir "${OUT}/upper_bound/ep${UB_EPOCHS_CUR}_bs${UB_BATCH_CUR}_mb${UB_MICRO_CUR}_lr$(sanitize_val "${UB_LR_CUR}")" \
                            --finetune_epochs "${UB_EPOCHS_CUR}" \
                            --train_batch_size "${UB_BATCH_CUR}" \
                            --micro_batch_size "${UB_MICRO_CUR}" \
                            --lr "${UB_LR_CUR}" \
                            --wandb_name "${BACKEND}_${TASK}_upper_bound_ep${UB_EPOCHS_CUR}_bs${UB_BATCH_CUR}_mb${UB_MICRO_CUR}_lr$(sanitize_val "${UB_LR_CUR}")"
                    done
                done
            done
        done
        fi

    if has_item upper_bound_no_lora "${VALID_EXP_LIST[@]}"; then
        for UB_NL_EPOCHS_CUR in "${UB_NOLORA_FINETUNE_EPOCHS_LIST[@]}"; do
            for UB_NL_BATCH_CUR in "${UB_NOLORA_TRAIN_BATCH_SIZE_LIST[@]}"; do
                for UB_NL_MICRO_CUR in "${UB_NOLORA_MICRO_BATCH_SIZE_LIST[@]}"; do
                    for UB_NL_LR_CUR in "${UB_NOLORA_LR_LIST[@]}"; do
                        UB_NL_TAG="${TASK}/upper_bound_no_lora_ep${UB_NL_EPOCHS_CUR}_bs${UB_NL_BATCH_CUR}_mb${UB_NL_MICRO_CUR}_lr$(sanitize_val "${UB_NL_LR_CUR}")"
                        run_exp "${UB_NL_TAG}" \
                            "${LAUNCH_PREFIX[@]}" \
                            "${COMMON_ARGS[@]}" \
                            --experiment upper_bound_no_lora \
                            --output_dir "${OUT}/upper_bound_no_lora/ep${UB_NL_EPOCHS_CUR}_bs${UB_NL_BATCH_CUR}_mb${UB_NL_MICRO_CUR}_lr$(sanitize_val "${UB_NL_LR_CUR}")" \
                            --finetune_epochs "${UB_NL_EPOCHS_CUR}" \
                            --train_batch_size "${UB_NL_BATCH_CUR}" \
                            --micro_batch_size "${UB_NL_MICRO_CUR}" \
                            --lr_full_finetune "${UB_NL_LR_CUR}" \
                            --wandb_name "${BACKEND}_${TASK}_upper_bound_no_lora_ep${UB_NL_EPOCHS_CUR}_bs${UB_NL_BATCH_CUR}_mb${UB_NL_MICRO_CUR}_lr$(sanitize_val "${UB_NL_LR_CUR}")"
                    done
                done
            done
        done
    fi

    for EPS in "${EPS_LIST[@]}"; do
        if has_item naive_dp "${VALID_EXP_LIST[@]}"; then
            for NAIVE_DP_EPOCHS_CUR in "${NAIVE_DP_FINETUNE_EPOCHS_LIST[@]}"; do
                for NAIVE_DP_BATCH_CUR in "${NAIVE_DP_TRAIN_BATCH_SIZE_LIST[@]}"; do
                    for NAIVE_DP_MICRO_CUR in "${NAIVE_DP_MICRO_BATCH_SIZE_LIST[@]}"; do
                        for NAIVE_DP_LR_CUR in "${NAIVE_DP_LR_LIST[@]}"; do
                            NAIVE_DP_TAG="${TASK}/naive_dp_eps${EPS}_ep${NAIVE_DP_EPOCHS_CUR}_bs${NAIVE_DP_BATCH_CUR}_mb${NAIVE_DP_MICRO_CUR}_lr$(sanitize_val "${NAIVE_DP_LR_CUR}")"
                            run_exp "${NAIVE_DP_TAG}" \
                                "${LAUNCH_PREFIX[@]}" \
                                "${COMMON_ARGS[@]}" \
                                --experiment naive_dp \
                                --output_dir "${OUT}/naive_dp_eps${EPS}/ep${NAIVE_DP_EPOCHS_CUR}_bs${NAIVE_DP_BATCH_CUR}_mb${NAIVE_DP_MICRO_CUR}_lr$(sanitize_val "${NAIVE_DP_LR_CUR}")" \
                                --epsilon_total "${EPS}" \
                                --finetune_epochs "${NAIVE_DP_EPOCHS_CUR}" \
                                --train_batch_size "${NAIVE_DP_BATCH_CUR}" \
                                --micro_batch_size "${NAIVE_DP_MICRO_CUR}" \
                                --lr "${NAIVE_DP_LR_CUR}" \
                                --wandb_name "${BACKEND}_${TASK}_naive_dp_eps${EPS}_ep${NAIVE_DP_EPOCHS_CUR}_bs${NAIVE_DP_BATCH_CUR}_mb${NAIVE_DP_MICRO_CUR}_lr$(sanitize_val "${NAIVE_DP_LR_CUR}")"
                        done
                    done
                done
            done
        fi

        if has_item naive_dp_no_lora "${VALID_EXP_LIST[@]}"; then
            for NAIVE_DP_NL_EPOCHS_CUR in "${NAIVE_DP_NOLORA_FINETUNE_EPOCHS_LIST[@]}"; do
                for NAIVE_DP_NL_BATCH_CUR in "${NAIVE_DP_NOLORA_TRAIN_BATCH_SIZE_LIST[@]}"; do
                    for NAIVE_DP_NL_MICRO_CUR in "${NAIVE_DP_NOLORA_MICRO_BATCH_SIZE_LIST[@]}"; do
                        for NAIVE_DP_NL_LR_CUR in "${NAIVE_DP_NOLORA_LR_LIST[@]}"; do
                            NAIVE_DP_NL_TAG="${TASK}/naive_dp_no_lora_eps${EPS}_ep${NAIVE_DP_NL_EPOCHS_CUR}_bs${NAIVE_DP_NL_BATCH_CUR}_mb${NAIVE_DP_NL_MICRO_CUR}_lr$(sanitize_val "${NAIVE_DP_NL_LR_CUR}")"
                            run_exp "${NAIVE_DP_NL_TAG}" \
                                "${LAUNCH_PREFIX[@]}" \
                                "${COMMON_ARGS[@]}" \
                                --experiment naive_dp_no_lora \
                                --output_dir "${OUT}/naive_dp_no_lora_eps${EPS}/ep${NAIVE_DP_NL_EPOCHS_CUR}_bs${NAIVE_DP_NL_BATCH_CUR}_mb${NAIVE_DP_NL_MICRO_CUR}_lr$(sanitize_val "${NAIVE_DP_NL_LR_CUR}")" \
                                --epsilon_total "${EPS}" \
                                --finetune_epochs "${NAIVE_DP_NL_EPOCHS_CUR}" \
                                --train_batch_size "${NAIVE_DP_NL_BATCH_CUR}" \
                                --micro_batch_size "${NAIVE_DP_NL_MICRO_CUR}" \
                                --lr_full_finetune "${NAIVE_DP_NL_LR_CUR}" \
                                --wandb_name "${BACKEND}_${TASK}_naive_dp_no_lora_eps${EPS}_ep${NAIVE_DP_NL_EPOCHS_CUR}_bs${NAIVE_DP_NL_BATCH_CUR}_mb${NAIVE_DP_NL_MICRO_CUR}_lr$(sanitize_val "${NAIVE_DP_NL_LR_CUR}")"
                        done
                    done
                done
            done
        fi

        if has_item dpsft "${VALID_EXP_LIST[@]}"; then
            for DPSFT_S1_EPOCHS_CUR in "${DPSFT_STAGE1_EPOCHS_LIST[@]}"; do
                for DPSFT_S2_EPOCHS_CUR in "${DPSFT_STAGE2_EPOCHS_LIST[@]}"; do
                    for DPSFT_BATCH_CUR in "${DPSFT_TRAIN_BATCH_SIZE_LIST[@]}"; do
                        for DPSFT_MICRO_CUR in "${DPSFT_MICRO_BATCH_SIZE_LIST[@]}"; do
                            for DPSFT_S1_LR_CUR in "${DPSFT_STAGE1_LR_LIST[@]}"; do
                                for DPSFT_S2_LR_CUR in "${DPSFT_STAGE2_LR_LIST[@]}"; do
                                    for DPSFT_DIM_CUR in "${DPSFT_SUBSPACE_DIM_LIST[@]}"; do
                                        for DPSFT_S1_RATIO_CUR in "${DPSFT_STAGE1_EPSILON_RATIO_LIST[@]}"; do
                                            for DPSFT_SCOPE_CUR in "${DPSFT_PARAM_SCOPE_LIST[@]}"; do
                                                for DPSFT_S1_CLIP_CUR in "${DPSFT_STAGE1_CLIP_NORM_LIST[@]}"; do
                                                    for DPSFT_S2_CLIP_CUR in "${DPSFT_STAGE2_CLIP_NORM_LIST[@]}"; do
                                                        DPSFT_TAG="${TASK}/dpsft_eps${EPS}_s1e${DPSFT_S1_EPOCHS_CUR}_s2e${DPSFT_S2_EPOCHS_CUR}_bs${DPSFT_BATCH_CUR}_mb${DPSFT_MICRO_CUR}_d${DPSFT_DIM_CUR}_s1lr$(sanitize_val "${DPSFT_S1_LR_CUR}")_s2lr$(sanitize_val "${DPSFT_S2_LR_CUR}")_s1r$(sanitize_val "${DPSFT_S1_RATIO_CUR}")_scope$(sanitize_val "${DPSFT_SCOPE_CUR}")"
                                                        run_exp "${DPSFT_TAG}" \
                                                            "${LAUNCH_PREFIX[@]}" \
                                                            "${COMMON_ARGS[@]}" \
                                                            --experiment dpsft \
                                                            --output_dir "${OUT}/dpsft_eps${EPS}/s1e${DPSFT_S1_EPOCHS_CUR}_s2e${DPSFT_S2_EPOCHS_CUR}_bs${DPSFT_BATCH_CUR}_mb${DPSFT_MICRO_CUR}_d${DPSFT_DIM_CUR}_s1lr$(sanitize_val "${DPSFT_S1_LR_CUR}")_s2lr$(sanitize_val "${DPSFT_S2_LR_CUR}")_s1r$(sanitize_val "${DPSFT_S1_RATIO_CUR}")_scope$(sanitize_val "${DPSFT_SCOPE_CUR}")" \
                                                            --epsilon_total "${EPS}" \
                                                            --train_batch_size "${DPSFT_BATCH_CUR}" \
                                                            --micro_batch_size "${DPSFT_MICRO_CUR}" \
                                                            --dpsft_stage1_epochs "${DPSFT_S1_EPOCHS_CUR}" \
                                                            --dpsft_stage2_epochs "${DPSFT_S2_EPOCHS_CUR}" \
                                                            --dpsft_stage1_lr "${DPSFT_S1_LR_CUR}" \
                                                            --dpsft_stage2_lr "${DPSFT_S2_LR_CUR}" \
                                                            --dpsft_subspace_dim "${DPSFT_DIM_CUR}" \
                                                            --dpsft_stage1_epsilon_ratio "${DPSFT_S1_RATIO_CUR}" \
                                                            --dpsft_param_scope "${DPSFT_SCOPE_CUR}" \
                                                            --dpsft_stage1_clip_norm "${DPSFT_S1_CLIP_CUR}" \
                                                            --dpsft_stage2_clip_norm "${DPSFT_S2_CLIP_CUR}" \
                                                            --dpsft_gpu_directions_max_elements "${DPSFT_GPU_DIRECTIONS_MAX_ELEMENTS}" \
                                                            --wandb_name "${BACKEND}_${TASK}_dpsft_eps${EPS}_s1e${DPSFT_S1_EPOCHS_CUR}_s2e${DPSFT_S2_EPOCHS_CUR}_bs${DPSFT_BATCH_CUR}_mb${DPSFT_MICRO_CUR}_d${DPSFT_DIM_CUR}_s1lr$(sanitize_val "${DPSFT_S1_LR_CUR}")_s2lr$(sanitize_val "${DPSFT_S2_LR_CUR}")_s1r$(sanitize_val "${DPSFT_S1_RATIO_CUR}")_scope$(sanitize_val "${DPSFT_SCOPE_CUR}")"
                                                    done
                                                done
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        fi

        if has_item baseline_a_global_matched "${VALID_EXP_LIST[@]}" || has_item matched_scope_global_dp "${VALID_EXP_LIST[@]}" || has_item baseline_b_shared_only "${VALID_EXP_LIST[@]}" || has_item dp_shared_only "${VALID_EXP_LIST[@]}" || has_item baseline_c_expert_only "${VALID_EXP_LIST[@]}" || has_item dp_expert_only "${VALID_EXP_LIST[@]}"; then
            for BASELINE_START_LAYER_CUR in "${BASELINE_START_LAYER_LIST[@]}"; do
                for BASELINE_LAYERS_CUR in "${BASELINE_NUM_LAYERS_LIST[@]}"; do
                    for BASELINE_EPOCHS_CUR in "${BASELINE_EPOCHS_LIST[@]}"; do
                        for BASELINE_BATCH_CUR in "${BASELINE_TRAIN_BATCH_SIZE_LIST[@]}"; do
                            for BASELINE_MICRO_CUR in "${BASELINE_MICRO_BATCH_SIZE_LIST[@]}"; do
                                for BASELINE_LR_CUR in "${BASELINE_LR_LIST[@]}"; do
                                    build_phase_a_freeze_args PHASE_A_FREEZE_FLAG
                                    BASELINE_COMMON_TAG="S${BASELINE_START_LAYER_CUR}_L${BASELINE_LAYERS_CUR}_ep${BASELINE_EPOCHS_CUR}_bs${BASELINE_BATCH_CUR}_mb${BASELINE_MICRO_CUR}_lr$(sanitize_val "${BASELINE_LR_CUR}")"

                                    if has_item baseline_a_global_matched "${VALID_EXP_LIST[@]}" || has_item matched_scope_global_dp "${VALID_EXP_LIST[@]}"; then
                                        MATCHED_TAG_BODY="${BASELINE_COMMON_TAG}_global"
                                        MATCHED_TAG="${TASK}/baseline_a_global_matched_eps${EPS}_${MATCHED_TAG_BODY}"
                                        run_exp "${MATCHED_TAG}" \
                                            "${LAUNCH_PREFIX[@]}" \
                                            "${COMMON_ARGS[@]}" \
                                            --experiment baseline_a_global_matched \
                                            --output_dir "${OUT}/baseline_a_global_matched_eps${EPS}/${MATCHED_TAG_BODY}" \
                                            --epsilon_total "${EPS}" \
                                            --ours_epochs "${BASELINE_EPOCHS_CUR}" \
                                            --start_layer "${BASELINE_START_LAYER_CUR}" \
                                            --num_layers_to_train "${BASELINE_LAYERS_CUR}" \
                                            --train_batch_size "${BASELINE_BATCH_CUR}" \
                                            --micro_batch_size "${BASELINE_MICRO_CUR}" \
                                            --lr "${BASELINE_LR_CUR}" \
                                            "${PHASE_A_FREEZE_FLAG[@]}" \
                                            --wandb_name "${BACKEND}_${TASK}_baseline_a_global_matched_eps${EPS}_${MATCHED_TAG_BODY}"
                                    fi

                                    if has_item baseline_b_shared_only "${VALID_EXP_LIST[@]}" || has_item dp_shared_only "${VALID_EXP_LIST[@]}"; then
                                        for BASELINE_EPS_RATIO_CUR in "${BASELINE_EPSILON_SHARED_RATIO_LIST[@]}"; do
                                            for BASELINE_EPS_MODE_CUR in "${BASELINE_SHARED_EPS_MODE_LIST[@]}"; do
                                                SHARED_TAG_BODY="${BASELINE_COMMON_TAG}_eps$(sanitize_val "${BASELINE_EPS_MODE_CUR}")"
                                                if [[ "${BASELINE_EPS_MODE_CUR}" == "shared" ]]; then
                                                    SHARED_TAG_BODY="${SHARED_TAG_BODY}_esr$(sanitize_val "${BASELINE_EPS_RATIO_CUR}")"
                                                fi
                                                SHARED_TAG="${TASK}/baseline_b_shared_only_eps${EPS}_${SHARED_TAG_BODY}"
                                                run_exp "${SHARED_TAG}" \
                                                    "${LAUNCH_PREFIX[@]}" \
                                                    "${COMMON_ARGS[@]}" \
                                                    --experiment baseline_b_shared_only \
                                                    --output_dir "${OUT}/baseline_b_shared_only_eps${EPS}/${SHARED_TAG_BODY}" \
                                                    --epsilon_total "${EPS}" \
                                                    --epsilon_shared_ratio "${BASELINE_EPS_RATIO_CUR}" \
                                                    --baseline_epsilon_mode "${BASELINE_EPS_MODE_CUR}" \
                                                    --ours_epochs "${BASELINE_EPOCHS_CUR}" \
                                                    --start_layer "${BASELINE_START_LAYER_CUR}" \
                                                    --num_layers_to_train "${BASELINE_LAYERS_CUR}" \
                                                    --train_batch_size "${BASELINE_BATCH_CUR}" \
                                                    --micro_batch_size "${BASELINE_MICRO_CUR}" \
                                                    --lr "${BASELINE_LR_CUR}" \
                                                    "${PHASE_A_FREEZE_FLAG[@]}" \
                                                    --wandb_name "${BACKEND}_${TASK}_baseline_b_shared_only_eps${EPS}_${SHARED_TAG_BODY}"
                                            done
                                        done
                                    fi

                                    if has_item baseline_c_expert_only "${VALID_EXP_LIST[@]}" || has_item dp_expert_only "${VALID_EXP_LIST[@]}"; then
                                        for BASELINE_EXPERT_UPDATE_SCALE_MODE_CUR in "${BASELINE_EXPERT_UPDATE_SCALE_MODE_LIST[@]}"; do
                                            for BASELINE_CLIP_SCOPE_CUR in "${BASELINE_CLIP_SCOPE_LIST[@]}"; do
                                                for BASELINE_EXPERT_LR_MULTIPLIER_CUR in "${BASELINE_EXPERT_LR_MULTIPLIER_LIST[@]}"; do
                                                    for BASELINE_EXPERT_OBJECTIVE_CUR in "${BASELINE_EXPERT_OBJECTIVE_LIST[@]}"; do
                                                        for BASELINE_ALTERNATING_SHARED_STEPS_CUR in "${BASELINE_ALTERNATING_SHARED_STEPS_LIST[@]}"; do
                                                            for BASELINE_ALTERNATING_EXPERT_STEPS_CUR in "${BASELINE_ALTERNATING_EXPERT_STEPS_LIST[@]}"; do
                                                                EXPERT_TAG_BODY="${BASELINE_COMMON_TAG}_den$(sanitize_val "${BASELINE_EXPERT_UPDATE_SCALE_MODE_CUR}")_clip$(sanitize_val "${BASELINE_CLIP_SCOPE_CUR}")_elm$(sanitize_val "${BASELINE_EXPERT_LR_MULTIPLIER_CUR}")_obj$(sanitize_val "${BASELINE_EXPERT_OBJECTIVE_CUR}")"
                                                                if [[ "${BASELINE_ALTERNATING_EXPERT_STEPS_CUR}" != "0" ]]; then
                                                                    EXPERT_TAG_BODY="${EXPERT_TAG_BODY}_altS$(sanitize_val "${BASELINE_ALTERNATING_SHARED_STEPS_CUR}")E$(sanitize_val "${BASELINE_ALTERNATING_EXPERT_STEPS_CUR}")"
                                                                fi
                                                                EXPERT_TAG="${TASK}/baseline_c_expert_only_eps${EPS}_${EXPERT_TAG_BODY}"
                                                                run_exp "${EXPERT_TAG}" \
                                                                    "${LAUNCH_PREFIX[@]}" \
                                                                    "${COMMON_ARGS[@]}" \
                                                                    --experiment baseline_c_expert_only \
                                                                    --output_dir "${OUT}/baseline_c_expert_only_eps${EPS}/${EXPERT_TAG_BODY}" \
                                                                    --epsilon_total "${EPS}" \
                                                                    --ours_epochs "${BASELINE_EPOCHS_CUR}" \
                                                                    --start_layer "${BASELINE_START_LAYER_CUR}" \
                                                                    --num_layers_to_train "${BASELINE_LAYERS_CUR}" \
                                                                    --train_batch_size "${BASELINE_BATCH_CUR}" \
                                                                    --micro_batch_size "${BASELINE_MICRO_CUR}" \
                                                                    --lr "${BASELINE_LR_CUR}" \
                                                                    --expert_update_scale_mode "${BASELINE_EXPERT_UPDATE_SCALE_MODE_CUR}" \
                                                                    --ours_clip_scope "${BASELINE_CLIP_SCOPE_CUR}" \
                                                                    --expert_lr_multiplier "${BASELINE_EXPERT_LR_MULTIPLIER_CUR}" \
                                                                    --expert_objective "${BASELINE_EXPERT_OBJECTIVE_CUR}" \
                                                                    --alternating_shared_steps "${BASELINE_ALTERNATING_SHARED_STEPS_CUR}" \
                                                                    --alternating_expert_steps "${BASELINE_ALTERNATING_EXPERT_STEPS_CUR}" \
                                                                    "${PHASE_A_FREEZE_FLAG[@]}" \
                                                                    --wandb_name "${BACKEND}_${TASK}_baseline_c_expert_only_eps${EPS}_${EXPERT_TAG_BODY}"
                                                            done
                                                        done
                                                    done
                                                done
                                            done
                                        done
                                    fi
                                done
                            done
                        done
                    done
                done
            done
        fi

        if has_item baseline_e_selected_layer_lora "${VALID_EXP_LIST[@]}" || has_item baseline_e_last_layer_lora "${VALID_EXP_LIST[@]}" || has_item last_layer_dp_lora_global "${VALID_EXP_LIST[@]}" || has_item dp_lora_last_layer "${VALID_EXP_LIST[@]}"; then
            for LAST_START_CUR in "${LAST_LAYER_START_LAYER_LIST[@]}"; do
                for LAST_LAYERS_CUR in "${LAST_LAYER_NUM_LAYERS_LIST[@]}"; do
                    for LAST_EPOCHS_CUR in "${LAST_LAYER_EPOCHS_LIST[@]}"; do
                        for LAST_BATCH_CUR in "${LAST_LAYER_TRAIN_BATCH_SIZE_LIST[@]}"; do
                            for LAST_MICRO_CUR in "${LAST_LAYER_MICRO_BATCH_SIZE_LIST[@]}"; do
                                for LAST_LR_CUR in "${LAST_LAYER_LR_LIST[@]}"; do
                                    build_phase_a_freeze_args PHASE_A_FREEZE_FLAG
                                    LAST_TAG_BODY="S${LAST_START_CUR}_L${LAST_LAYERS_CUR}_ep${LAST_EPOCHS_CUR}_bs${LAST_BATCH_CUR}_mb${LAST_MICRO_CUR}_lr$(sanitize_val "${LAST_LR_CUR}")_global"
                                    LAST_TAG="${TASK}/baseline_e_selected_layer_lora_eps${EPS}_${LAST_TAG_BODY}"
                                    run_exp "${LAST_TAG}" \
                                        "${LAUNCH_PREFIX[@]}" \
                                        "${COMMON_ARGS[@]}" \
                                        --experiment baseline_e_selected_layer_lora \
                                        --output_dir "${OUT}/baseline_e_selected_layer_lora_eps${EPS}/${LAST_TAG_BODY}" \
                                        --epsilon_total "${EPS}" \
                                        --ours_epochs "${LAST_EPOCHS_CUR}" \
                                        --start_layer "${LAST_START_CUR}" \
                                        --num_layers_to_train "${LAST_LAYERS_CUR}" \
                                        --train_batch_size "${LAST_BATCH_CUR}" \
                                        --micro_batch_size "${LAST_MICRO_CUR}" \
                                        --lr "${LAST_LR_CUR}" \
                                        "${PHASE_A_FREEZE_FLAG[@]}" \
                                        --wandb_name "${BACKEND}_${TASK}_baseline_e_selected_layer_lora_eps${EPS}_${LAST_TAG_BODY}"
                                done
                            done
                        done
                    done
                done
            done
        fi

        if has_item ours "${VALID_EXP_LIST[@]}"; then
            for OURS_LAYERS_CUR in "${OURS_NUM_LAYERS_LIST[@]}"; do
                OURS_START_LAYER_CUR="${OURS_START_LAYER_LIST[0]}"
                OURS_LAYER_TAG="S$(sanitize_val "${OURS_START_LAYER_CUR}")_L${OURS_LAYERS_CUR}"
                for OURS_EPOCHS_CUR in "${OURS_EPOCHS_LIST[@]}"; do
                    for OURS_BATCH_CUR in "${OURS_TRAIN_BATCH_SIZE_LIST[@]}"; do
                        for OURS_MICRO_CUR in "${OURS_MICRO_BATCH_SIZE_LIST[@]}"; do
                            for OURS_LR_CUR in "${OURS_LR_LIST[@]}"; do
                                for OURS_EPS_RATIO_CUR in "${OURS_EPSILON_SHARED_RATIO_LIST[@]}"; do
                                    for OURS_EXPERT_RESIDUAL_WEIGHTING_CUR in "${OURS_EXPERT_RESIDUAL_WEIGHTING_LIST[@]}"; do
                                        for OURS_EXPERT_RESIDUAL_MARGIN_THRESHOLD_CUR in "${OURS_EXPERT_RESIDUAL_MARGIN_THRESHOLD_LIST[@]}"; do
                                            for OURS_EXPERT_RESIDUAL_SMALL_WEIGHT_CUR in "${OURS_EXPERT_RESIDUAL_SMALL_WEIGHT_LIST[@]}"; do
                                                for OURS_EXPERT_RESIDUAL_MIN_WEIGHT_CUR in "${OURS_EXPERT_RESIDUAL_MIN_WEIGHT_LIST[@]}"; do
                                                    for OURS_EXPERT_UPDATE_SCALE_MODE_CUR in "${OURS_EXPERT_UPDATE_SCALE_MODE_LIST[@]}"; do
                                                        for OURS_CLIP_SCOPE_CUR in "${OURS_CLIP_SCOPE_LIST[@]}"; do
                                                        for OURS_EXPERT_LR_MULTIPLIER_CUR in "${OURS_EXPERT_LR_MULTIPLIER_LIST[@]}"; do
                                                            for OURS_EXPERT_OBJECTIVE_CUR in "${OURS_EXPERT_OBJECTIVE_LIST[@]}"; do
                                                                for OURS_ALTERNATING_SHARED_STEPS_CUR in "${OURS_ALTERNATING_SHARED_STEPS_LIST[@]}"; do
                                                                    for OURS_ALTERNATING_EXPERT_STEPS_CUR in "${OURS_ALTERNATING_EXPERT_STEPS_LIST[@]}"; do
                                    OURS_EXPERT_RESIDUAL_WEIGHTING="${OURS_EXPERT_RESIDUAL_WEIGHTING_CUR}"
                                    OURS_EXPERT_RESIDUAL_MARGIN_THRESHOLD="${OURS_EXPERT_RESIDUAL_MARGIN_THRESHOLD_CUR}"
                                    OURS_EXPERT_RESIDUAL_SMALL_WEIGHT="${OURS_EXPERT_RESIDUAL_SMALL_WEIGHT_CUR}"
                                    OURS_EXPERT_RESIDUAL_MIN_WEIGHT="${OURS_EXPERT_RESIDUAL_MIN_WEIGHT_CUR}"
                                    OURS_EXPERT_UPDATE_SCALE_MODE="${OURS_EXPERT_UPDATE_SCALE_MODE_CUR}"
                                    OURS_CLIP_SCOPE="${OURS_CLIP_SCOPE_CUR}"
                                    OURS_EXPERT_LR_MULTIPLIER="${OURS_EXPERT_LR_MULTIPLIER_CUR}"
                                    OURS_EXPERT_OBJECTIVE="${OURS_EXPERT_OBJECTIVE_CUR}"
                                    OURS_ALTERNATING_SHARED_STEPS="${OURS_ALTERNATING_SHARED_STEPS_CUR}"
                                    OURS_ALTERNATING_EXPERT_STEPS="${OURS_ALTERNATING_EXPERT_STEPS_CUR}"
                                    OURS_EPS_RATIO_TAG="esr$(sanitize_val "${OURS_EPS_RATIO_CUR}")"
                                    OURS_PHASE_A_ONLY_TAG=""
                                    OURS_PHASE_A_ONLY_FLAG=()
                                    if [[ "${OURS_PHASE_A_ONLY}" == "1" ]]; then
                                        OURS_PHASE_A_ONLY_TAG="_phaseAonly"
                                        OURS_PHASE_A_ONLY_FLAG=(--phase_a_only)
                                    fi
                                    OURS_EXPERT_TAG="$(build_switch_ours_extra_tag)"
                                    OURS_TAG="${TASK}/ours_eps${EPS}_${OURS_LAYER_TAG}_ep${OURS_EPOCHS_CUR}_bs${OURS_BATCH_CUR}_mb${OURS_MICRO_CUR}_lr$(sanitize_val "${OURS_LR_CUR}")_${OURS_EPS_RATIO_TAG}${OURS_PHASE_A_ONLY_TAG}${OURS_EXPERT_TAG}"

                                PHASE_A_FREEZE_ITEMS=()
                                PHASE_A_FREEZE_SPEC="${FREEZE_IN_PHASE_A//,/ }"
                                if [[ -n "${PHASE_A_FREEZE_SPEC// }" ]]; then
                                    read -r -a PHASE_A_FREEZE_ITEMS <<< "${PHASE_A_FREEZE_SPEC}"
                                fi
                                if [[ "${FREEZE_ROUTER_IN_PHASE_A}" == "1" ]] && ! has_item router "${PHASE_A_FREEZE_ITEMS[@]}"; then
                                    PHASE_A_FREEZE_ITEMS+=(router)
                                fi
                                PHASE_A_FREEZE_FLAG=()
                                if [[ "${#PHASE_A_FREEZE_ITEMS[@]}" -gt 0 ]]; then
                                    PHASE_A_FREEZE_FLAG=(--freeze_in_phase_a "${PHASE_A_FREEZE_ITEMS[@]}")
                                fi
                                build_switch_ours_extra_args OURS_EXTRA_FLAG

                                run_exp "${OURS_TAG}" \
                                    "${LAUNCH_PREFIX[@]}" \
                                    "${COMMON_ARGS[@]}" \
                                    --experiment ours \
                                    --output_dir "${OUT}/ours_eps${EPS}/${OURS_LAYER_TAG}_ep${OURS_EPOCHS_CUR}_bs${OURS_BATCH_CUR}_mb${OURS_MICRO_CUR}_lr$(sanitize_val "${OURS_LR_CUR}")_${OURS_EPS_RATIO_TAG}${OURS_PHASE_A_ONLY_TAG}${OURS_EXPERT_TAG}" \
                                    --epsilon_total "${EPS}" \
                                    --ours_epochs "${OURS_EPOCHS_CUR}" \
                                    --start_layer "${OURS_START_LAYER_CUR}" \
                                    --num_layers_to_train "${OURS_LAYERS_CUR}" \
                                    --train_batch_size "${OURS_BATCH_CUR}" \
                                    --micro_batch_size "${OURS_MICRO_CUR}" \
                                    --lr "${OURS_LR_CUR}" \
                                    --epsilon_shared_ratio "${OURS_EPS_RATIO_CUR}" \
                                    "${OURS_PHASE_A_ONLY_FLAG[@]}" \
                                    "${OURS_EXTRA_FLAG[@]}" \
                                    "${PHASE_A_FREEZE_FLAG[@]}" \
                                    --wandb_name "${BACKEND}_${TASK}_ours_eps${EPS}_${OURS_LAYER_TAG}_ep${OURS_EPOCHS_CUR}_bs${OURS_BATCH_CUR}_mb${OURS_MICRO_CUR}_lr$(sanitize_val "${OURS_LR_CUR}")_${OURS_EPS_RATIO_TAG}${OURS_PHASE_A_ONLY_TAG}${OURS_EXPERT_TAG}"
                                                                    done
                                                                done
                                                            done
                                                        done
                                                        done
                                                    done
                                                done
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        fi


        if has_item oursver2 "${VALID_EXP_LIST[@]}"; then
            for OURS_LAYERS_CUR in "${OURS_NUM_LAYERS_LIST[@]}"; do
                OURS_START_LAYER_CUR="${OURS_START_LAYER_LIST[0]}"
                OURS_LAYER_TAG="S$(sanitize_val "${OURS_START_LAYER_CUR}")_L${OURS_LAYERS_CUR}"
                for OURS_AE_CUR in "${OURS_ATTENTION_EPOCHS_LIST[@]}"; do
                    for OURS_EE_CUR in "${OURS_EXPERT_EPOCHS_LIST[@]}"; do
                        for OURS_BATCH_CUR in "${OURS_TRAIN_BATCH_SIZE_LIST[@]}"; do
                            for OURS_MICRO_CUR in "${OURS_MICRO_BATCH_SIZE_LIST[@]}"; do
                                for OURS_LR_CUR in "${OURS_LR_LIST[@]}"; do
                                    for OURS_PROBE_METHOD_CUR in "${OURS_PROBE_METHOD_LIST[@]}"; do
                                        for OURS_UPDATE_RATIO_CUR in "${OURS_UPDATE_RATIO_LIST[@]}"; do
                                            for OURS_RANK_MODE_CUR in "${OURS_RANK_MODE_LIST[@]}"; do
                                                for OURS_PHASE_A_RATIO_CUR in "${OURS_PHASE_A_RATIO_LIST[@]}"; do
                                                    EPS_ATTN="$(calc_eps_attention "${EPS}" "${OURS_PHASE_A_RATIO_CUR}")"
                                                    PROBE_CACHE="${PROBE_CACHE_DIR}/${TASK}_${OURS_PROBE_METHOD_CUR}_u${OURS_UPDATE_RATIO_CUR}_r${OURS_RANK_MODE_CUR}.pt"
                                                    OURSVER2_TAG="${TASK}/oursver2_eps${EPS}_${OURS_LAYER_TAG}_ae${OURS_AE_CUR}_ee${OURS_EE_CUR}_bs${OURS_BATCH_CUR}_mb${OURS_MICRO_CUR}_lr$(sanitize_val "${OURS_LR_CUR}")_p$(sanitize_val "${OURS_PROBE_METHOD_CUR}")_u$(sanitize_val "${OURS_UPDATE_RATIO_CUR}")_r$(sanitize_val "${OURS_RANK_MODE_CUR}")_par$(sanitize_val "${OURS_PHASE_A_RATIO_CUR}")"
                                                    OURS_PROBE_FLAG=()
                                                    if [[ "${ENABLE_IMPORTANCE_PROBING}" == "1" ]]; then
                                                        OURS_PROBE_FLAG=(--enable_importance_probing)
                                                    fi

                                                    PHASE_A_FREEZE_ITEMS=()
                                                    PHASE_A_FREEZE_SPEC="${FREEZE_IN_PHASE_A//,/ }"
                                                    if [[ -n "${PHASE_A_FREEZE_SPEC// }" ]]; then
                                                        read -r -a PHASE_A_FREEZE_ITEMS <<< "${PHASE_A_FREEZE_SPEC}"
                                                    fi
                                                    if [[ "${FREEZE_ROUTER_IN_PHASE_A}" == "1" ]] && ! has_item router "${PHASE_A_FREEZE_ITEMS[@]}"; then
                                                        PHASE_A_FREEZE_ITEMS+=(router)
                                                    fi
                                                    PHASE_A_FREEZE_FLAG=()
                                                    if [[ "${#PHASE_A_FREEZE_ITEMS[@]}" -gt 0 ]]; then
                                                        PHASE_A_FREEZE_FLAG=(--freeze_in_phase_a "${PHASE_A_FREEZE_ITEMS[@]}")
                                                    fi

                                                    EVAL_LAYER_FLAG=()
                                                    if [[ "${EVAL_EACH_LAYER}" == "1" ]]; then
                                                        EVAL_LAYER_FLAG=(--eval_each_layer)
                                                    fi

                                                    run_exp "${OURSVER2_TAG}" \
                                                        "${LAUNCH_PREFIX[@]}" \
                                                        "${COMMON_ARGS[@]}" \
                                                        --experiment oursver2 \
                                                        --output_dir "${OUT}/oursver2_eps${EPS}/${OURS_LAYER_TAG}_ae${OURS_AE_CUR}_ee${OURS_EE_CUR}_bs${OURS_BATCH_CUR}_mb${OURS_MICRO_CUR}_lr$(sanitize_val "${OURS_LR_CUR}")_p$(sanitize_val "${OURS_PROBE_METHOD_CUR}")_u$(sanitize_val "${OURS_UPDATE_RATIO_CUR}")_r$(sanitize_val "${OURS_RANK_MODE_CUR}")_par$(sanitize_val "${OURS_PHASE_A_RATIO_CUR}")" \
                                                        --epsilon_total "${EPS}" \
                                                        --epsilon_attention "${EPS_ATTN}" \
                                                        --start_layer "${OURS_START_LAYER_CUR}" \
                                                        --num_layers_to_train "${OURS_LAYERS_CUR}" \
                                                        --attention_epochs "${OURS_AE_CUR}" \
                                                        --expert_epochs "${OURS_EE_CUR}" \
                                                        --oursver2_shared_total_epochs "${OURS_AE_CUR}" \
                                                        --oursver2_expert_total_epochs "${OURS_EE_CUR}" \
                                                        --oursver2_windows "${OURSVER2_WINDOWS}" \
                                                        --oursver2_refresh_windows "${OURSVER2_REFRESH_WINDOWS}" \
                                                        --train_batch_size "${OURS_BATCH_CUR}" \
                                                        --micro_batch_size "${OURS_MICRO_CUR}" \
                                                        --lr "${OURS_LR_CUR}" \
                                                        "${PHASE_A_FREEZE_FLAG[@]}" \
                                                        "${EVAL_LAYER_FLAG[@]}" \
                                                        "${OURS_PROBE_FLAG[@]}" \
                                                        --probe_method "${OURS_PROBE_METHOD_CUR}" \
                                                        --update_ratio "${OURS_UPDATE_RATIO_CUR}" \
                                                        --rank_mode "${OURS_RANK_MODE_CUR}" \
                                                        --probe_cache "${PROBE_CACHE}" \
                                                        --wandb_name "${BACKEND}_${TASK}_oursver2_eps${EPS}_${OURS_LAYER_TAG}_ae${OURS_AE_CUR}_ee${OURS_EE_CUR}_bs${OURS_BATCH_CUR}_mb${OURS_MICRO_CUR}_lr$(sanitize_val "${OURS_LR_CUR}")_p$(sanitize_val "${OURS_PROBE_METHOD_CUR}")_u$(sanitize_val "${OURS_UPDATE_RATIO_CUR}")_r$(sanitize_val "${OURS_RANK_MODE_CUR}")_par$(sanitize_val "${OURS_PHASE_A_RATIO_CUR}")"
                                                done
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        fi

        if has_item oursseq2seq "${VALID_EXP_LIST[@]}"; then
            for OURS_LAYERS_CUR in "${OURS_NUM_LAYERS_LIST[@]}"; do
                OURS_START_LAYER_CUR="${OURS_START_LAYER_LIST[0]}"
                OURS_LAYER_TAG="S$(sanitize_val "${OURS_START_LAYER_CUR}")_L${OURS_LAYERS_CUR}"
                for OURS_EPOCHS_CUR in "${OURS_EPOCHS_LIST[@]}"; do
                    for OURS_BATCH_CUR in "${OURS_TRAIN_BATCH_SIZE_LIST[@]}"; do
                        for OURS_MICRO_CUR in "${OURS_MICRO_BATCH_SIZE_LIST[@]}"; do
                            for OURS_LR_CUR in "${OURS_LR_LIST[@]}"; do
                                for OURS_EPS_RATIO_CUR in "${OURS_EPSILON_SHARED_RATIO_LIST[@]}"; do
                                    for OURS_EXPERT_RESIDUAL_WEIGHTING_CUR in "${OURS_EXPERT_RESIDUAL_WEIGHTING_LIST[@]}"; do
                                        for OURS_EXPERT_RESIDUAL_MARGIN_THRESHOLD_CUR in "${OURS_EXPERT_RESIDUAL_MARGIN_THRESHOLD_LIST[@]}"; do
                                            for OURS_EXPERT_RESIDUAL_SMALL_WEIGHT_CUR in "${OURS_EXPERT_RESIDUAL_SMALL_WEIGHT_LIST[@]}"; do
                                                for OURS_EXPERT_RESIDUAL_MIN_WEIGHT_CUR in "${OURS_EXPERT_RESIDUAL_MIN_WEIGHT_LIST[@]}"; do
                                                    for OURS_EXPERT_UPDATE_SCALE_MODE_CUR in "${OURS_EXPERT_UPDATE_SCALE_MODE_LIST[@]}"; do
                                                        for OURS_CLIP_SCOPE_CUR in "${OURS_CLIP_SCOPE_LIST[@]}"; do
                                                        for OURS_EXPERT_LR_MULTIPLIER_CUR in "${OURS_EXPERT_LR_MULTIPLIER_LIST[@]}"; do
                                                            for OURS_EXPERT_OBJECTIVE_CUR in "${OURS_EXPERT_OBJECTIVE_LIST[@]}"; do
                                                                for OURS_ALTERNATING_SHARED_STEPS_CUR in "${OURS_ALTERNATING_SHARED_STEPS_LIST[@]}"; do
                                                                    for OURS_ALTERNATING_EXPERT_STEPS_CUR in "${OURS_ALTERNATING_EXPERT_STEPS_LIST[@]}"; do
                                    OURS_EXPERT_RESIDUAL_WEIGHTING="${OURS_EXPERT_RESIDUAL_WEIGHTING_CUR}"
                                    OURS_EXPERT_RESIDUAL_MARGIN_THRESHOLD="${OURS_EXPERT_RESIDUAL_MARGIN_THRESHOLD_CUR}"
                                    OURS_EXPERT_RESIDUAL_SMALL_WEIGHT="${OURS_EXPERT_RESIDUAL_SMALL_WEIGHT_CUR}"
                                    OURS_EXPERT_RESIDUAL_MIN_WEIGHT="${OURS_EXPERT_RESIDUAL_MIN_WEIGHT_CUR}"
                                    OURS_EXPERT_UPDATE_SCALE_MODE="${OURS_EXPERT_UPDATE_SCALE_MODE_CUR}"
                                    OURS_CLIP_SCOPE="${OURS_CLIP_SCOPE_CUR}"
                                    OURS_EXPERT_LR_MULTIPLIER="${OURS_EXPERT_LR_MULTIPLIER_CUR}"
                                    OURS_EXPERT_OBJECTIVE="${OURS_EXPERT_OBJECTIVE_CUR}"
                                    OURS_ALTERNATING_SHARED_STEPS="${OURS_ALTERNATING_SHARED_STEPS_CUR}"
                                    OURS_ALTERNATING_EXPERT_STEPS="${OURS_ALTERNATING_EXPERT_STEPS_CUR}"
                                    OURS_EPS_RATIO_TAG="esr$(sanitize_val "${OURS_EPS_RATIO_CUR}")"
                                    OURS_PHASE_A_ONLY_TAG=""
                                    OURS_PHASE_A_ONLY_FLAG=()
                                    if [[ "${OURS_PHASE_A_ONLY}" == "1" ]]; then
                                        OURS_PHASE_A_ONLY_TAG="_phaseAonly"
                                        OURS_PHASE_A_ONLY_FLAG=(--phase_a_only)
                                    fi
                                    OURS_EXPERT_TAG="$(build_switch_ours_extra_tag)"
                                    OURSSEQ2SEQ_TAG="${TASK}/oursseq2seq_eps${EPS}_${OURS_LAYER_TAG}_ep${OURS_EPOCHS_CUR}_bs${OURS_BATCH_CUR}_mb${OURS_MICRO_CUR}_lr$(sanitize_val "${OURS_LR_CUR}")_${OURS_EPS_RATIO_TAG}${OURS_PHASE_A_ONLY_TAG}${OURS_EXPERT_TAG}"

                                PHASE_A_FREEZE_ITEMS=()
                                PHASE_A_FREEZE_SPEC="${FREEZE_IN_PHASE_A//,/ }"
                                if [[ -n "${PHASE_A_FREEZE_SPEC// }" ]]; then
                                    read -r -a PHASE_A_FREEZE_ITEMS <<< "${PHASE_A_FREEZE_SPEC}"
                                fi
                                if [[ "${FREEZE_ROUTER_IN_PHASE_A}" == "1" ]] && ! has_item router "${PHASE_A_FREEZE_ITEMS[@]}"; then
                                    PHASE_A_FREEZE_ITEMS+=(router)
                                fi
                                PHASE_A_FREEZE_FLAG=()
                                if [[ "${#PHASE_A_FREEZE_ITEMS[@]}" -gt 0 ]]; then
                                    PHASE_A_FREEZE_FLAG=(--freeze_in_phase_a "${PHASE_A_FREEZE_ITEMS[@]}")
                                fi
                                build_switch_ours_extra_args OURS_EXTRA_FLAG

                                run_exp "${OURSSEQ2SEQ_TAG}" \
                                    "${LAUNCH_PREFIX[@]}" \
                                    "${COMMON_ARGS[@]}" \
                                    --experiment oursseq2seq \
                                    --output_dir "${OUT}/oursseq2seq_eps${EPS}/${OURS_LAYER_TAG}_ep${OURS_EPOCHS_CUR}_bs${OURS_BATCH_CUR}_mb${OURS_MICRO_CUR}_lr$(sanitize_val "${OURS_LR_CUR}")_${OURS_EPS_RATIO_TAG}${OURS_PHASE_A_ONLY_TAG}${OURS_EXPERT_TAG}" \
                                    --epsilon_total "${EPS}" \
                                    --ours_epochs "${OURS_EPOCHS_CUR}" \
                                    --start_layer "${OURS_START_LAYER_CUR}" \
                                    --num_layers_to_train "${OURS_LAYERS_CUR}" \
                                    --train_batch_size "${OURS_BATCH_CUR}" \
                                    --micro_batch_size "${OURS_MICRO_CUR}" \
                                    --lr "${OURS_LR_CUR}" \
                                    --epsilon_shared_ratio "${OURS_EPS_RATIO_CUR}" \
                                    "${OURS_PHASE_A_ONLY_FLAG[@]}" \
                                    "${OURS_EXTRA_FLAG[@]}" \
                                    "${PHASE_A_FREEZE_FLAG[@]}" \
                                    --wandb_name "${BACKEND}_${TASK}_oursseq2seq_eps${EPS}_${OURS_LAYER_TAG}_ep${OURS_EPOCHS_CUR}_bs${OURS_BATCH_CUR}_mb${OURS_MICRO_CUR}_lr$(sanitize_val "${OURS_LR_CUR}")_${OURS_EPS_RATIO_TAG}${OURS_PHASE_A_ONLY_TAG}${OURS_EXPERT_TAG}"
                                                                    done
                                                                done
                                                            done
                                                        done
                                                        done
                                                    done
                                                done
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        fi
    done
done

echo ""
echo "============================================================"
echo "ALL RUNS COMPLETE -> ${BASE_OUT}"
echo "============================================================"

if [[ -s "${FAIL_LOG}" ]]; then
    echo "FAILED RUNS:"
    cat "${FAIL_LOG}"
else
    echo "All runs succeeded."
fi
