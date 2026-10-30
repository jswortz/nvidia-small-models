#!/usr/bin/env bash
# One-click reproduction on a Brev GPU VM (Launchable in VM mode).
#
# Sets up the same two containers as the desktop: "nim" (vLLM v0.10.1 serving Nemotron Nano 9B v2)
# and "poc-dev" (Python 3.12 with this repo mounted), joined by the "poc-net" network. Then it starts
# brev/reproduce.sh in the background. Progress: tail -f ~/poc-reproduce.log. Results: results/SUMMARY.md.
#
# Launch parameters (environment variables, all optional):
#   NVIDIA_API_KEY   also run the evals with hosted Nemotron Super as the planner
#   PHASE6           "yes" to also fine-tune the router (LoRA SFT, about 15 extra minutes, 200 GB disk)
#   SKIP_SINGLE      "yes" to skip the single-agent baseline (saves about 7 minutes)
#
# Run by hand from a checkout:  bash brev/setup.sh
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

VLLM_IMAGE=vllm/vllm-openai:v0.10.1   # pinned: the latest image broke the Nemotron tool parser
BASE=nvidia/NVIDIA-Nemotron-Nano-9B-v2
HF_CACHE="$HOME/hf-cache"
PLUGINS="$HOME/nim-plugins"
mkdir -p "$HF_CACHE" "$PLUGINS" checkpoints results

echo "== $(date -u +%T) Pulling images"
docker network create poc-net >/dev/null 2>&1 || true
docker pull -q "$VLLM_IMAGE"
docker pull -q python:3.12

echo "== $(date -u +%T) Downloading $BASE (about 17 GB)"
snapshot=$(docker run --rm -v "$HF_CACHE":/root/.cache/huggingface --entrypoint python3 "$VLLM_IMAGE" -c \
  "from huggingface_hub import snapshot_download; print(snapshot_download('$BASE'))" | tail -n 1)
# The tool call parser plugin ships in the model's Hugging Face repo.
sudo cp -L "${snapshot/\/root\/.cache\/huggingface/$HF_CACHE}/nemotron_toolcall_parser_no_streaming.py" "$PLUGINS/" 2>/dev/null \
  || cp -L "${snapshot/\/root\/.cache\/huggingface/$HF_CACHE}/nemotron_toolcall_parser_no_streaming.py" "$PLUGINS/"

bash brev/reproduce.sh start-nim "$BASE"

echo "== $(date -u +%T) Starting the dev container"
docker rm -f poc-dev >/dev/null 2>&1 || true
docker run -dit --name poc-dev --network poc-net -v "$REPO":/workspace -w /workspace \
  -e OPENAI_BASE_URL=http://nim:8000/v1 -e OPENAI_API_KEY=local -e NVIDIA_API_KEY="${NVIDIA_API_KEY:-}" \
  python:3.12 bash >/dev/null
docker exec poc-dev pip install -q --root-user-action=ignore -r requirements.txt

echo "== $(date -u +%T) Setup done. Reproduction runs in the background: tail -f ~/poc-reproduce.log"
# Launch parameters reach this script but not later SSH sessions, so pass them on explicitly.
export NVIDIA_API_KEY="${NVIDIA_API_KEY:-}" PHASE6="${PHASE6:-no}" SKIP_SINGLE="${SKIP_SINGLE:-no}"
nohup bash brev/reproduce.sh all > "$HOME/poc-reproduce.log" 2>&1 < /dev/null &
