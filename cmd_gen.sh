#!/usr/bin/env bash
# Generate independent execution-provenance R-GCN ablation commands.
# Usage: ./cmd_gen.sh <prgcn-wo|prgcn-wo-multiseed>

set -euo pipefail

readonly VERSION_TAG="v2"
readonly DATASET="twowiki_provenance"
readonly DATASET_SHORT_NAME="twp"
readonly PROFILE="provenance_full"
readonly METHOD="execution_provenance_rgcn_retriever"
readonly METHOD_SHORT_NAME="prgcn"
readonly SINGLE_SEED=13
readonly -a MULTI_SEEDS=(13 17 29 43 57)
readonly -a VARIANTS=(
  full_rgcn
  wo_graph
  wo_edge_type
  wo_edge_weight
  wo_hard_negatives
  wo_edge_rerank
)

usage() {
  printf 'Usage: %s <prgcn-wo|prgcn-wo-multiseed>\n' "${0##*/}" >&2
}

emit_command() {
  local seed="$1"
  local variant="$2"
  local command_index="$3"
  local device="cuda:$((command_index % 8))"
  local name="${METHOD_SHORT_NAME}-${DATASET_SHORT_NAME}-sd${seed}-${VERSION_TAG}"

  if [[ "$variant" != "full_rgcn" ]]; then
    name="${METHOD_SHORT_NAME}-${DATASET_SHORT_NAME}-${variant}-sd${seed}-${VERSION_TAG}"
  fi

  printf '%s\n' \
    "python experiment/run.py name=${name} dataset=${DATASET} profile=${PROFILE} device=${device} seed=${seed} method=${METHOD} method.variant=${variant}"
}

emit_wo_task() {
  local seed="$1"
  local command_index=0
  local variant

  for variant in "${VARIANTS[@]}"; do
    emit_command "$seed" "$variant" "$command_index"
    ((command_index += 1))
  done
}

emit_wo_multiseed_task() {
  local command_index=0
  local seed
  local variant

  for seed in "${MULTI_SEEDS[@]}"; do
    for variant in "${VARIANTS[@]}"; do
      emit_command "$seed" "$variant" "$command_index"
      ((command_index += 1))
    done
  done
}

if [[ $# -ne 1 ]]; then
  usage
  exit 2
fi

case "$1" in
  prgcn-wo)
    emit_wo_task "$SINGLE_SEED"
    ;;
  prgcn-wo-multiseed)
    emit_wo_multiseed_task
    ;;
  *)
    usage
    exit 2
    ;;
esac
