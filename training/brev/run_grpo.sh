#!/usr/bin/env bash
# Phase 6, step 4 on a rented GPU: GRPO on the SFT router with the store-state reward.
#
# Needs checkpoints/router-sft from run_sft.sh. From ~/poc on the instance:
#   bash training/brev/run_grpo.sh train   # rerun to resume from the latest checkpoint
#   bash training/brev/run_grpo.sh eval    # after the best checkpoint is exported to checkpoints/router-grpo/
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs

train() {
  python3 training/grpo/build_grpo_data.py
  # Loading the 9B policy briefly needs about 85 GB of host RAM (float32 load plus the vLLM copy),
  # just under the A100 instance's 94 GB. A swap file absorbs that peak, and Ray's memory monitor
  # is turned off so it does not kill the worker at 95% before the kernel can swap.
  if ! swapon --show | grep -q /swapfile; then
    sudo fallocate -l 64G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile >/dev/null && sudo swapon /swapfile
  fi
  docker run --rm --gpus all --ipc=host --shm-size=32g -e RAY_memory_monitor_refresh_ms=0 \
    -v "$PWD":/poc -v "$HOME/hf-cache":/root/.cache/huggingface -w /poc nvcr.io/nvidia/nemo-rl:v0.7.0 \
    bash -lc "python training/grpo/run_grpo.py --config training/configs/router_grpo.yaml" 2>&1 | tee -a logs/grpo_train.log
}

case "${1:-train}" in
  train) train ;;
  eval) bash training/brev/eval_router.sh router-grpo ;;
  *) echo "usage: $0 train|eval"; exit 1 ;;
esac
