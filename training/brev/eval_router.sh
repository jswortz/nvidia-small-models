#!/usr/bin/env bash
# Serve checkpoints/<model> with vLLM (same image as the 3090) and score the router on dev and stress.
#   bash training/brev/eval_router.sh router-sft
set -euo pipefail
cd "$(dirname "$0")/../.."
MODEL=${1:?usage: eval_router.sh <folder under checkpoints/>}
mkdir -p logs
docker rm -f router-vllm >/dev/null 2>&1 || true
docker run -d --name router-vllm --gpus all --ipc=host -p 8000:8000 \
  -v "$PWD/checkpoints":/ckpt -v "$HOME/hf-cache":/root/.cache/huggingface vllm/vllm-openai:v0.10.1 \
  --model "/ckpt/$MODEL" --served-model-name "$MODEL" --trust-remote-code \
  --max-model-len 8192 --max-num-seqs 8 --max-num-batched-tokens 8192
echo "Waiting for vLLM..."
until curl -sf localhost:8000/v1/models >/dev/null; do
  docker ps -q -f name=router-vllm | grep -q . || { docker logs router-vllm 2>&1 | tail -30; exit 1; }
  sleep 10
done
docker run --rm --network host -v "$PWD":/poc -w /poc python:3.12 bash -c "
  pip install -q openai==3.37.0    # the router eval needs only openai; same version as poc-dev
  for split in router_dev router_stress_test; do
    python evals/run_router_eval.py --cases training/data/\$split.jsonl --model $MODEL \
      --base-url http://localhost:8000/v1 --prompt short
  done" 2>&1 | tee -a "logs/eval_$MODEL.log"
docker rm -f router-vllm >/dev/null
