#!/usr/bin/env bash
# Phase 6, step 3 on a rented GPU: LoRA SFT, merge, and router eval.
#
# Copy the repo to ~/poc on the instance, then run from ~/poc:
#   bash training/brev/run_sft.sh train    # LoRA SFT; rerun to resume from the latest checkpoint
#   bash training/brev/run_sft.sh merge    # best adapter (lowest val loss) + base -> checkpoints/router-sft/
#   bash training/brev/run_sft.sh eval     # serve merged model with vLLM, score dev and stress
#   bash training/brev/run_sft.sh all
set -euo pipefail
cd "$(dirname "$0")/../.."

AUTOMODEL_IMAGE=nvcr.io/nvidia/nemo-automodel:26.10.00
BASE=nvidia/NVIDIA-Nemotron-Nano-9B-v2
HF_CACHE=$HOME/hf-cache
mkdir -p "$HF_CACHE" checkpoints logs

automodel_run() {
  docker run --rm --gpus all --ipc=host --shm-size=16g \
    -v "$PWD":/poc -v "$HF_CACHE":/root/.cache/huggingface -w /poc "$AUTOMODEL_IMAGE" bash -lc "$1"
}

train() {
  python3 training/build_sft_data.py
  # AutoModel requires the full model snapshot (not just the weights) in the cache.
  automodel_run "hf download $BASE >/dev/null" 2>&1 | tail -n 3
  automodel_run "automodel training/configs/router_peft.yaml --nproc-per-node 1" 2>&1 | tee -a logs/sft_train.log
}

merge() {
  local adapter="" p
  for p in LOWEST_VAL LATEST; do   # best validation loss if recorded, else the last checkpoint
    [ -e "checkpoints/router-sft-train/$p" ] && { adapter=$(readlink -f "checkpoints/router-sft-train/$p"); break; }
  done
  [ -n "$adapter" ] || { echo "No checkpoint found in checkpoints/router-sft-train"; exit 1; }
  echo "Merging adapter: $adapter"
  automodel_run "python training/merge_lora.py \
    --base \$(ls -d /root/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-Nano-9B-v2/snapshots/*) \
    --adapter ${adapter#$PWD/}/model --out checkpoints/router-sft" 2>&1 | tee -a logs/sft_merge.log
}

evaluate() {
  bash training/brev/eval_router.sh router-sft
}

case "${1:-all}" in
  train) train ;;
  merge) merge ;;
  eval) evaluate ;;
  all) train; merge; evaluate ;;
  *) echo "usage: $0 train|merge|eval|all"; exit 1 ;;
esac
