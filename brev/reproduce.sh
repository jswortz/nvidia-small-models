#!/usr/bin/env bash
# Reproduce the README results. Run on the host after brev/setup.sh (it calls "all" for you):
#   bash brev/reproduce.sh all                 # tests and evals; writes results/SUMMARY.md
#   bash brev/reproduce.sh start-nim <model> [extra vLLM args]
#
# Environment: NVIDIA_API_KEY (adds hosted-planner runs), PHASE6=yes (fine-tunes the router),
# SKIP_SINGLE=yes (skips the 7 minute single-agent baseline), PLANNER_MODEL (hosted planner id).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

VLLM_IMAGE=vllm/vllm-openai:v0.10.1
BASE=nvidia/NVIDIA-Nemotron-Nano-9B-v2
HF_CACHE="$HOME/hf-cache"
PLUGINS="$HOME/nim-plugins"
HOSTED_PLANNER=${PLANNER_MODEL:-nvidia/nemotron-3.5-super-vl-120b-a12b}
SUMMARY=results/SUMMARY.md

# Same serving flags as the desktop 3090. --served-model-name keeps the base name working for the
# writer and planner when a merged fine-tuned model is served instead.
start_nim() {
  local model=$1; shift
  echo "== $(date -u +%T) Starting nim with $model (vLLM captures CUDA graphs, about 3 minutes)"
  docker rm -f nim >/dev/null 2>&1 || true
  docker run -d --name nim --network poc-net --gpus all -p 8000:8000 --ipc=host \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -v "$HF_CACHE":/root/.cache/huggingface -v "$PLUGINS":/plugins -v "$REPO/checkpoints":/adapters \
    "$VLLM_IMAGE" --model "$model" "$@" --trust-remote-code --max-model-len 8192 --max-num-seqs 8 \
    --max-num-batched-tokens 8192 --gpu-memory-utilization 0.93 --enable-auto-tool-choice \
    --tool-parser-plugin /plugins/nemotron_toolcall_parser_no_streaming.py --tool-call-parser nemotron_json >/dev/null
  for _ in $(seq 1 120); do
    curl -sf localhost:8000/health >/dev/null && { echo "nim is up"; return 0; }
    docker ps -q -f name=^nim$ | grep -q . || break
    sleep 10
  done
  docker logs nim 2>&1 | tail -n 30
  echo "nim did not start"; exit 1
}

# Run a command in poc-dev, keep the full output in results/<name>.txt, and copy the summary lines.
run() {
  local name=$1 title=$2 cmd=$3
  echo "== $(date -u +%T) $title"
  docker exec poc-dev bash -c "$cmd" > "results/$name.txt" 2>&1 || echo "(exited with an error, see results/$name.txt)" >> "results/$name.txt"
  { echo; echo "### $title"; echo '```'
    # Failing case lines only when there are few of them (the single agent fails most cases).
    [ "$(grep -ac 'FAIL  ' "results/$name.txt")" -le 10 ] && grep -a 'FAIL  ' "results/$name.txt" || true
    grep -aE "^Agent:|^Router accuracy|^Model calls|^Model: |checks passed|Traceback|exited with an error" "results/$name.txt" || true
    echo '```'; } >> "$SUMMARY"
  tail -n 4 "results/$name.txt"
}

LOCAL="env -u PLANNER_BASE_URL -u PLANNER_MODEL"
# NVIDIA_API_KEY is set inside poc-dev by setup.sh, so it never appears on a command line.
HOSTED="env PLANNER_BASE_URL=https://integrate.api.nvidia.com/v1 PLANNER_MODEL=$HOSTED_PLANNER"
TRAINED="ROUTER_MODEL=router-sft ROUTER_PROMPT=short"

evals() {   # $1: label suffix, $2: extra env for the routed agent (empty for the base router)
  local tag=$1 router_env=$2
  run "routed_62$tag" "Routed agent, 62 cases, local planner$3" \
    "$LOCAL $router_env python evals/run_eval.py --agent routed --cases evals/cases_phase5.jsonl --profile"
  run "routed_holdout$tag" "Routed agent, 11 held-out cases, local planner$3" \
    "$LOCAL $router_env python evals/run_eval.py --agent routed --cases evals/cases_holdout.jsonl"
  if [ -n "${NVIDIA_API_KEY:-}" ]; then
    run "routed_62_hosted$tag" "Routed agent, 62 cases, hosted Super planner$3" \
      "$HOSTED $router_env python evals/run_eval.py --agent routed --cases evals/cases_phase5.jsonl"
    run "routed_holdout_hosted$tag" "Routed agent, 11 held-out cases, hosted Super planner$3" \
      "$HOSTED $router_env python evals/run_eval.py --agent routed --cases evals/cases_holdout.jsonl"
  fi
}

phase6() {
  for p in full short; do
    run "router_base_$p" "Router alone, base model, $p prompt (dev, then stress)" \
      "for s in router_dev router_stress_test; do $LOCAL python evals/run_router_eval.py --cases training/data/\$s.jsonl --prompt $p; done"
  done
  echo "== $(date -u +%T) Fine-tuning the router (LoRA SFT with NeMo AutoModel); nim is stopped to free the GPU"
  docker rm -f nim >/dev/null
  bash training/brev/run_sft.sh train > results/sft_train.txt 2>&1
  sudo rm -rf checkpoints/router-sft 2>/dev/null || rm -rf checkpoints/router-sft
  bash training/brev/run_sft.sh merge > results/sft_merge.txt 2>&1
  { echo; echo "### Router fine-tuning"; echo '```'
    grep -ahE "val_loss|Merged [0-9]+ LoRA" results/sft_train.txt results/sft_merge.txt | tail -n 3; echo '```'; } >> "$SUMMARY"
  start_nim /adapters/router-sft --served-model-name "$BASE" router-sft
  run router_sft "Router alone, fine-tuned (dev, then stress)" \
    "for s in router_dev router_stress_test; do $LOCAL python evals/run_router_eval.py --cases training/data/\$s.jsonl --model router-sft --prompt short; done"
  evals _sft "$TRAINED" ", fine-tuned router"
}

all() {
  local start=$SECONDS
  { echo "# Reproduction results"; echo
    echo "Started $(date -u '+%F %T') UTC on $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)."
    echo "Expected numbers are in brev/README.md. Full output for each run is in results/."; } > "$SUMMARY"
  run tests "Offline tests (fake model clients)" \
    'for t in tests/smoke_test.py tests/test_policy_engine.py tests/test_router.py tests/test_multi_order.py tests/test_planner_fallback.py tests/test_nat_profiling.py tests/test_reward.py tests/test_eval_scoring.py; do python $t | tail -n 1; done'
  docker exec poc-dev python data/generate.py >/dev/null
  evals "" "" ""
  if [ "${SKIP_SINGLE:-no}" != "yes" ]; then
    run single_62 "Single agent baseline, 62 cases" \
      "$LOCAL python evals/run_eval.py --agent single --cases evals/cases_phase5.jsonl"
  fi
  [ "${PHASE6:-no}" = "yes" ] && phase6
  echo; echo "Finished in $(( (SECONDS - start) / 60 )) minutes." | tee -a "$SUMMARY"
  echo "REPRODUCE_DONE"
}

case "${1:-all}" in
  start-nim) shift; start_nim "$@" ;;
  all) all ;;
  *) echo "usage: $0 all | start-nim <model> [vLLM args]"; exit 1 ;;
esac
