#!/usr/bin/env bash
# Generate evidence-retriever multiseed and execution-provenance R-GCN ablation commands.
# Usage: ./cmd_gen.sh [--session NAME] [--prefix SHELL_CODE] <task>

set -euo pipefail

usage() {
  printf '%s\n' \
    "Usage: ${0##*/} [--session NAME] [--prefix SHELL_CODE] <task>" \
    "" \
    "Tasks:" \
    "  prgcn-wo" \
    "  prgcn-wo-multiseed" \
    "  evidence-hotpotqa-multiseed" \
    "  evidence-twowiki-multiseed" \
    "  evidence-musique-multiseed" \
    "" \
    "Options:" \
    "  --session NAME       Run panes in the named Zellij session" \
    "  --sesion NAME        Alias for --session (kept for compatibility)" \
    "  --prefix SHELL_CODE  Shell code inserted before 'python'" \
    "                       Examples: 'uv run' or 'source cloud_setup/setup.bash;'" \
    "  -h, --help           Show this help" \
    >&2
}

emit_zellij_command() {
  local pane_name="$1"
  shift

  local -a zellij_command=(zellij)
  if [[ -n "$SESSION_NAME" ]]; then
    zellij_command+=(--session "$SESSION_NAME")
  fi
  if [[ -n "$COMMAND_PREFIX" ]]; then
    local -a python_parts=(python experiment/run.py "$@")
    local python_command
    local part
    printf -v python_command '%q' "${python_parts[0]}"
    for part in "${python_parts[@]:1}"; do
      printf -v python_command '%s %q' "$python_command" "$part"
    done
    zellij_command+=(run --name "$pane_name" -- bash -c "${COMMAND_PREFIX} ${python_command}")
  else
    zellij_command+=(run --name "$pane_name" -- python experiment/run.py "$@")
  fi

  printf '%q' "${zellij_command[0]}"
  printf ' %q' "${zellij_command[@]:1}"
  printf '\n'
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

  emit_zellij_command "$name" \
    "name=${name}" \
    "dataset=${DATASET}" \
    "profile=${PROFILE}" \
    "device=${device}" \
    "seed=${seed}" \
    "method=${METHOD}" \
    "method.variant=${variant}"
}

emit_evidence_command() {
  local dataset="$1"
  local dataset_short_name="$2"
  local method="$3"
  local seed="$4"
  local command_index="$5"
  local device="cuda:$((command_index % 8))"
  local name="evidence-${dataset_short_name}-${method}-sd${seed}-${VERSION_TAG}"

  emit_zellij_command "$name" \
    "name=${name}" \
    "dataset=${dataset}" \
    "profile=${EVIDENCE_PROFILE}" \
    "device=${device}" \
    "seed=${seed}" \
    "method=${method}"
}

emit_evidence_multiseed_task() {
  local dataset="$1"
  local dataset_short_name="$2"
  local command_index=0
  local seed
  local method

  for method in "${EVIDENCE_BASELINE_METHODS[@]}"; do
    emit_evidence_command "$dataset" "$dataset_short_name" "$method" "$SINGLE_SEED" "$command_index"
    ((command_index += 1))
  done

  for seed in "${MULTI_SEEDS[@]}"; do
    for method in "${EVIDENCE_TRAINABLE_METHODS[@]}"; do
      emit_evidence_command "$dataset" "$dataset_short_name" "$method" "$seed" "$command_index"
      ((command_index += 1))
    done
  done
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

main() {
  SESSION_NAME=""
  COMMAND_PREFIX=""

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --session|--sesion)
        if [[ $# -lt 2 || -z "$2" ]]; then
          printf 'Error: %s requires a value.\n' "$1" >&2
          usage
          exit 2
        fi
        SESSION_NAME="$2"
        shift 2
        ;;
      --session=*|--sesion=*)
        SESSION_NAME="${1#*=}"
        if [[ -z "$SESSION_NAME" ]]; then
          printf 'Error: --session requires a value.\n' >&2
          usage
          exit 2
        fi
        shift
        ;;
      --prefix)
        if [[ $# -lt 2 || -z "$2" ]]; then
          printf 'Error: --prefix requires a value.\n' >&2
          usage
          exit 2
        fi
        COMMAND_PREFIX="$2"
        shift 2
        ;;
      --prefix=*)
        COMMAND_PREFIX="${1#*=}"
        if [[ -z "$COMMAND_PREFIX" ]]; then
          printf 'Error: --prefix requires a value.\n' >&2
          usage
          exit 2
        fi
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      --)
        shift
        break
        ;;
      -*)
        printf 'Error: unknown option: %s\n' "$1" >&2
        usage
        exit 2
        ;;
      *)
        break
        ;;
    esac
  done

  readonly SESSION_NAME
  readonly COMMAND_PREFIX
  readonly VERSION_TAG="v2"
  readonly DATASET="twowiki_provenance"
  readonly DATASET_SHORT_NAME="twp"
  readonly PROFILE="provenance_full"
  readonly METHOD="execution_provenance_rgcn_retriever"
  readonly METHOD_SHORT_NAME="prgcn"
  readonly SINGLE_SEED=13
  readonly -a MULTI_SEEDS=(13 17 29 43 57)
  readonly EVIDENCE_PROFILE="full"
  readonly -a EVIDENCE_BASELINE_METHODS=(bm25 dense graphrag)
  readonly -a EVIDENCE_TRAINABLE_METHODS=(dense_ft dense_rgcn_graph_retriever dense_ft_rgcn_graph_retriever)
  readonly -a VARIANTS=(
    full_rgcn
    wo_graph
    wo_edge_type
    wo_edge_weight
    wo_hard_negatives
    wo_edge_rerank
  )

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
    evidence-hotpotqa-multiseed)
      emit_evidence_multiseed_task "hotpotqa" "hp"
      ;;
    evidence-twowiki-multiseed)
      emit_evidence_multiseed_task "2wiki" "tw"
      ;;
    evidence-musique-multiseed)
      emit_evidence_multiseed_task "musique" "mq"
      ;;
    *)
      usage
      exit 2
      ;;
  esac
}

main "$@"
