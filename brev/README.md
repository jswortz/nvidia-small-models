# One-click reproduction on Brev

This folder turns the repo into a Brev Launchable. One click gives a GPU VM that serves Nemotron Nano 9B v2 with vLLM, runs the offline tests, and runs the evals behind the numbers in the main README. Results land in `results/SUMMARY.md` on the instance.

- `setup.sh` is the Launchable's setup script. It builds the same two containers as the desktop setup (`nim` serving the model, `poc-dev` running the code, both on `poc-net`), then starts `reproduce.sh` in the background.
- `reproduce.sh` runs the tests and evals and writes the summary. It can also be run by hand: `bash brev/reproduce.sh all`.

## Creating the Launchable (Brev Console, Launchables tab)

| Field | Value |
| --- | --- |
| Name | Retail support agent: routed Nemotron Nano |
| Source | Git repository: `https://github.com/jswortz/workspace` (must be public) |
| Runtime | VM mode, with Docker and the NVIDIA container toolkit |
| GPU | One GPU with 40 GB or more (L40S, A100, or H100). Phase 6 needs an A100 80 GB or H100 |
| Disk | 100 GB, or 200 GB with `PHASE6=yes` (the NeMo AutoModel image alone is 51 GB) |
| Setup script | `bash brev/setup.sh` |
| Ports | None needed. vLLM listens on 8000 inside the VM |
| Launch parameters | `NVIDIA_API_KEY` (Text, optional), `PHASE6` (Choice: no, yes), `SKIP_SINGLE` (Choice: no, yes) |

Launch parameters reach the setup script only, so `setup.sh` passes them on to the background run. The API key goes into the `poc-dev` container's environment and never into a file or a command line.

## What runs

| Step | Time (measured on an A6000, except Phase 6) | Needs |
| --- | --- | --- |
| Pull images, download the model (17 GB), start vLLM | 14 min | |
| Offline tests (79 checks, fake clients) | under 1 min | |
| Routed agent, 62 cases with NAT profiling, then 11 held-out cases, local planner | 4 min | |
| Single-agent baseline, 62 cases | 5 min | `SKIP_SINGLE=no` |
| Routed agent again with hosted Nemotron Super as the planner | 5 min | `NVIDIA_API_KEY` |
| Router alone (base, both prompts), LoRA SFT, merge, restart vLLM on the merged model, router and routed evals again | 30 to 40 min | `PHASE6=yes` |

Watch progress with `tail -f ~/poc-reproduce.log`. The log ends with `REPRODUCE_DONE`. Full output of every step is in `results/<step>.txt`, and eval traces are in `evals/results/`.

Stop or delete the instance when the summary is written. The setup has no reason to keep running.

## Tested run

Tested 2026-10-27 on a Brev A6000 48 GB (massedcompute, $0.68/hr) with `NVIDIA_API_KEY` set and `PHASE6=no`. Setup took 14 minutes and the evals 13, so the whole run cost about $0.39 including boot. Every result below matched the 3090 except the single agent (12/62, 6 harmful) and the hosted-planner 62-case run (62/62). Phase 6 was not run on Brev through this script; its steps are the ones used in the Phase 6 sessions.

## Expected results

From the desktop RTX 3090 runs on this code (run 16 in CLAUDE.md for the base model). Accuracy should match closely; time per case will differ by GPU.

| Run | Expected |
| --- | --- |
| Offline tests | All checks pass |
| Routed agent, 62 cases, local planner | 60/62, 0 harmful, 0 false promises, router 62/62. Both misses are ungrounded citations (v005 from the writer, o002 from the planner) |
| Routed agent, held-out 11, local planner | 10/11, 0 harmful, 0 false promises, router 11/11. The miss is h010, an ungrounded planner citation |
| Single agent, 62 cases | 11/62 to 12/62, 4 to 6 harmful actions, 7 to 8 false promises. The single agent varies more from run to run |
| Routed agent, hosted Super planner (base router) | 61/62 to 62/62 and 11/11, 0 harmful, 0 false promises |
| Fine-tuned router + hosted Super planner | 61/62 and 11/11, 0 harmful, 0 false promises |
| Router alone, fine-tuned (dev, stress) | 95/95 and 165/165 on the 3090 |

Model output at temperature 0 can still vary slightly across GPUs and vLLM batch sizes, so a case or two moving is expected. Harmful actions should stay at 0 for the routed agent on any hardware, because the policy engine, not the model, decides every store change.
