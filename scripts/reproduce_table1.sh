#!/usr/bin/env bash
# =============================================================================
# Reproduce the DualPrompt track of Table I (Vision Class-Incremental Learning)
# from "Soft-TransFormers for Continual Learning".
#
# Backbone: ViT-B/16 (Sup-21K, timm vit_base_patch16_224).
# Soft-TF is toggled by --subnet:
#     dense    -> plain baseline (no mask)          [--subnet defaults to dense]
#     soft     -> + Soft-TF  (real-valued masks)    <-- our method
#     adapter  -> + Adapter  (PEFT baseline)
#     lora     -> + LoRA     (PEFT baseline)
# Soft-TF masks are placed on the E-Prompt attention layers, i.e. the last
# three blocks L[10,11,12] (0-indexed [9,10,11], the config default).
#
# Usage:   ./scripts/reproduce_table1.sh <GPU_ID> [dataset] [variant]
#   dataset : c100_10 | c100_20 | imr | all   (default: all)
#   variant : dense | soft | adapter | lora | all   (default: soft)
#
# Examples:
#   ./scripts/reproduce_table1.sh 0                 # Soft-TF on all 3 datasets
#   ./scripts/reproduce_table1.sh 0 c100_10 all     # every variant, 10-CIFAR100
#   ./scripts/reproduce_table1.sh 0 imr soft        # Soft-TF, 10-Split-ImageNet-R
#
# Each run prints Avg. Acc / Forgetting and the trainable-parameter count that
# populate the corresponding Table I cell.
# =============================================================================
set -uo pipefail

GPU="${1:?usage: reproduce_table1.sh <GPU_ID> [dataset] [variant]}"
DATASET="${2:-all}"
VARIANT="${3:-soft}"

export CUDA_VISIBLE_DEVICES="$GPU"
export WANDB_MODE="${WANDB_MODE:-offline}"   # run without a wandb account by default
PY="${PY:-python}"
OUT="${OUT:-./output}"

# dataset -> (config, epochs)   [epochs follow the paper / configs]
declare -A CFG=(  [c100_10]=10cifar100_dualprompt_pgp
                  [c100_20]=20cifar100_dualprompt_pgp
                  [imr]=imr_dualprompt_pgp )
declare -A EP=(   [c100_10]=20  [c100_20]=20  [imr]=50 )

run_one () {   # run_one <dataset_key> <variant>
  local d="$1" v="$2"
  local cfg="${CFG[$d]}" ep="${EP[$d]}"
  local extra=()
  [ "$v" = dense ] || extra=(--subnet "$v")     # dense = no --subnet flag
  echo "=== Table I | ${cfg} | variant=${v} | epochs=${ep} | $(date '+%F %T') ==="
  $PY main.py "$cfg" \
      --model vit_base_patch16_224 \
      --output_dir "$OUT" \
      --epochs "$ep" \
      --no_pgp \
      "${extra[@]}"
}

datasets=(c100_10 c100_20 imr); [ "$DATASET" != all ] && datasets=("$DATASET")
variants=(dense soft adapter lora); [ "$VARIANT" != all ] && variants=("$VARIANT")

for d in "${datasets[@]}"; do
  for v in "${variants[@]}"; do
    run_one "$d" "$v"
  done
done
