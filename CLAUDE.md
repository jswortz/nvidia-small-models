# Retail Support Agent POC

A multi-agent retail order-support assistant on NVIDIA's stack, built as interview prep for the NVIDIA Engagement Tech Lead, Agentic AI role. The goal is to prove one claim with numbers: a small Nemotron model is reliable when it handles intent and wording while deterministic code makes policy decisions and takes actions.

All data is synthetic. Nothing from a current or past client goes into this project.

## Environment

| Piece | Where | Notes |
| --- | --- | --- |
| `nim` container | Windows desktop, RTX 3090 (24 GB) | vLLM serving `nvidia/NVIDIA-Nemotron-Nano-9B-v2` at `http://nim:8000/v1` |
| `poc-dev` container | Same desktop, Python 3.12, no GPU | Code runs here; `C:\poc` is mounted at `/workspace` |
| `poc-net` | Docker network | Lets `poc-dev` reach `nim` by name |
| Hosted NIM | build.nvidia.com free tier | Not wired yet; intended for the planner (Nemotron Super) |

Working `nim` command (PowerShell, one line). Pin the image: `latest` broke the Nemotron tool parser.

```
docker run -d --name nim --restart unless-stopped --network poc-net --gpus all -p 8000:8000 --ipc=host -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -v C:\hf-cache:/root/.cache/huggingface -v C:\nim-plugins:/plugins vllm/vllm-openai:v0.10.1 --model nvidia/NVIDIA-Nemotron-Nano-9B-v2 --trust-remote-code --max-model-len 8192 --max-num-seqs 8 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.93 --enable-auto-tool-choice --tool-parser-plugin /plugins/nemotron_toolcall_parser_no_streaming.py --tool-call-parser nemotron_json
```

`poc-dev` command:

```
docker run -dit --name poc-dev --restart unless-stopped --network poc-net -v C:\poc:/workspace -w /workspace -e OPENAI_BASE_URL=http://nim:8000/v1 -e OPENAI_API_KEY=local python:3.12 bash
```

Known environment gotchas:

- Pulling new images over SSH fails with a Windows credential error. Pull at the desktop; `run`, `start`, `logs`, and `exec` work over SSH.
- The 3090 runs out of memory with vLLM defaults. The flags above (`--max-num-seqs 8`, utilization 0.93) are what work. `--max-num-batched-tokens` must be at least `--max-model-len`.
- `--enforce-eager` is no longer needed with these flags. Dropping it (run 12) cut model call time by about 36% with identical outputs; startup takes about 160 s while vLLM captures CUDA graphs. If the server runs out of memory at startup, add it back.
- `nim` is the model server. Never develop inside it; attach VS Code to `poc-dev`.
- `pip` packages in `poc-dev` are lost when it is recreated: `pip install -r requirements.txt`.
- The build.nvidia.com free tier is unpredictable: the same escalated case took 4 s to 30 s across runs, calls sometimes returned 500, and one hung for over 5 minutes. The OpenAI SDK default (600 s timeout, 2 retries) turns a hang into a long stall, which is why the planner client sets its own timeout.

## Layout

```
data/generate.py          Builds data/store.db and data/policy.md. Deterministic; "today" is fixed at 2026-10-26. Rerun to reset.
tools/store_tools.py      lookup_order (returns computed policy_facts), issue_refund, create_replacement, search_products
tools/policy_search.py    Keyword search over numbered policy clauses; get_clause by id
tools/registry.py         OpenAI tool schemas plus call_tool()
tools/mcp_server.py       Same tools over MCP (pinned mcp<2; 2.x renamed FastMCP)
agents/single_agent.py    Phase 2: one Nano agent with all tools, prompt rules, nudge harness
agents/router.py          Phase 3: Nano classifies a message into intent, order_id, damaged, photo_provided, new_product (guided_json)
agents/policy_engine.py   Phase 3: plain rules decide act, decline, info, need_info, or escalate, naming each clause
agents/nat_profiling.py   Phase 4: emits NeMo Agent Toolkit events per stage and model call; runs NAT's ProfilerRunner
agents/routed_agent.py    Phase 3: route, look up, decide, act in code, then Nano writes the reply. Escalations go to the planner;
                          a failed hosted planner call falls back to local Nano. Every model call is timed in result["model_calls"].
evals/run_eval.py         Scores tools, final store state, citations, honesty (false promises), harmful actions, router accuracy, per group
evals/cases_phase2.jsonl  Original 10 cases
evals/cases_phase5.jsonl  62 cases in 11 failure-mode groups
evals/cases_holdout.jsonl 11 held-out cases: 10 new wordings of the run 6 router errors, plus h011 (two orders, one allowed and one declined)
evals/run_router_eval.py  Phase 6: scores the router alone against labeled routes (--prompt full|short, --model, --base-url)
training/specs.py, generate_messages.py   Phase 6: synthetic router data; training/data/ holds train 676, dev 83, stress 147
training/reward.py        Phase 6: store-state reward (+1.0 state match, -2.0 harmful, +0.25 exact route, +0.1 valid JSON)
training/build_sft_data.py, configs/router_peft.yaml, merge_lora.py   Phase 6 step 3: LoRA SFT with NeMo AutoModel, merge
training/grpo/, configs/router_grpo.yaml   Phase 6 step 4: GRPO with NeMo RL v0.7.0; router_env.py wraps reward.py
training/brev/            Scripts run on the rented GPU: run_sft.sh, run_grpo.sh, eval_router.sh
tests/                    smoke_test.py (15 checks), test_policy_engine.py (18 checks), test_router.py (9 checks), test_multi_order.py (7 checks),
                          test_planner_fallback.py (6 checks), test_nat_profiling.py (5 checks), test_reward.py (10 checks). Fake clients;
                          no model needed. test_merge_lora.py (1 check) needs torch, so it runs in the training container.
```

## Commands (inside poc-dev, from /workspace)

```
python tests/smoke_test.py
python tests/test_policy_engine.py
python tests/test_router.py
python tests/test_multi_order.py
python tests/test_planner_fallback.py
python tests/test_nat_profiling.py
python tests/test_reward.py
python evals/run_router_eval.py --cases training/data/router_stress_test.jsonl --prompt short   # router alone
python data/generate.py                                                   # reset the demo store
python -m agents.routed_agent "My blender from order 1187 arrived cracked..."
python evals/run_eval.py --agent routed --cases evals/cases_phase5.jsonl
python evals/run_eval.py --agent routed --cases evals/cases_phase5.jsonl --profile   # plus NAT per-stage tokens and time
python evals/run_eval.py --agent single --cases evals/cases_phase5.jsonl
```

Each eval case gets a fresh temporary store, so evals never touch `data/store.db`.

## Design rules

- The model never decides policy and never calls action tools in the routed agent. The router classifies; `policy_engine.py` decides; code executes; the writer only describes `action_result`.
- Anything deterministic (dates, windows, thresholds, status) is computed in code. The model ignored these facts when asked to reason about them.
- Tools enforce hard limits (no double refunds, $500 supervisor approval, same-category swaps). Policy judgment lives in the engine.
- Grade outcomes, not paths. Track harmful actions separately from failures: a wrongful refund is worse than a missed one.
- Report harness help honestly: nudges, appended citations, and escalations are all counted in eval output.

## Results so far

| Run | Setup | Result |
| --- | --- | --- |
| 1 to 4 | Single agent, prompt rules and a nudge harness, 10 cases | Plateaued at 6/10 with 2 false promises; failures moved between cases as rules changed |
| 5 | Routed agent, 10 cases | 10/10 |
| 6 | Routed agent, 62 cases | 58/62, 0 harmful actions, 0 false promises, 4.6 s average, router accurate on 58/62 |
| 7 | Run 6 plus router prompt fixes (clearer intent definitions, broader `damaged`, six few-shot examples) | 61/62, 0 harmful actions, 0 false promises, 4.8 s average, router accurate on 62/62; no case regressed; 10/10 on the original cases |
| 8 | Single agent vs routed agent (run 7 code), 62 cases, same Nano 9B model | Single 21/62, 4 harmful actions, 7 false promises, 6.9 s average, 28 cases nudged. Routed 61/62, 0 harmful, 0 false promises, 4.8 s. No case passes single and fails routed |
| 9 | Run 7 plus multi-order handling: code finds every real order number in the message, routes, decides, and acts once per order, then one reply covers all | 62/62, 0 harmful actions, 0 false promises, 4.9 s average, router accurate on 62/62. Held-out set (11 cases): 11/11, 0 harmful, 0 false promises, 5.5 s average, router accurate on 11/11 |
| 10 | Run 9 with the planner on hosted Nemotron Super (`nvidia/nemotron-3.5-super-vl-120b-a12b`), out_of_scope cases o001 to o003 (all escalate) | Answers were correct whenever the endpoint responded, but latency was unpredictable: 5.7 to 16.1 s average across four runs (the local planner averaged 6.1 s on this group in run 8), slowest single call 16 s, and one run lost o002 and o003 to HTTP 500 errors. With a 30 s timeout and local fallback added, a run with forced timeouts passed 3/3 with 3 fallbacks, 7.2 s average |
| 11 | Phase 4: run 9 code with NeMo Agent Toolkit profiling (`--profile`), local planner, `--enforce-eager` | 62/62, 0 harmful, 0 false promises. 4.91 s per case (95% CI 4.63 to 5.19), 2.33 s per model call. Model calls are about 99.8% of case time; lookup, engine, and action together take about 10 ms |
| 12 | Run 11 with `--enforce-eager` removed from the `nim` command (CUDA graphs on) | 62/62, 0 harmful, 0 false promises, identical routes and answers on all 62 cases. 3.13 s per case (95% CI 2.94 to 3.32), 1.49 s per model call: 36% faster |

Runs 11 and 12 by stage (NAT profile, average per case where the stage ran; tokens are the same in both runs):

| Stage | Cases | LLM calls | Prompt tokens | Completion tokens | Run 11 ms | Run 12 ms |
| --- | --- | --- | --- | --- | --- | --- |
| router | 62 | 63 | 430 | 49 | 2,132 | 1,390 |
| lookup | 62 | 0 | 0 | 0 | 1.2 | 1.2 |
| engine | 62 | 0 | 0 | 0 | 0.03 | 0.03 |
| action | 20 | 0 | 0 | 0 | 9.1 | 5.7 |
| planner | 4 | 9 | 2,265 | 105 | 4,834 | 3,255 |
| writer | 58 | 58 | 391 | 61 | 2,620 | 1,625 |

Traces: `evals/results/routed_20261026_205919.jsonl` (run 11) and `evals/results/routed_20261026_211712.jsonl` (run 12); NAT outputs sit beside each in a `_nat` folder.

Run 8 by group (S = single agent, R = routed agent):

| Group | Cases | Success S | Success R | Harmful S | Harmful R | False promises S | False promises R | Latency S (s) | Latency R (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 10 | 6 | 10 | 0 | 0 | 0 | 0 | 6.0 | 5.1 |
| paraphrase | 10 | 3 | 10 | 0 | 0 | 1 | 0 | 7.7 | 4.5 |
| vague | 7 | 2 | 7 | 0 | 0 | 0 | 0 | 6.7 | 5.1 |
| missing_id | 5 | 4 | 5 | 0 | 0 | 1 | 0 | 4.1 | 3.3 |
| damage | 7 | 0 | 7 | 3 | 0 | 2 | 0 | 9.9 | 4.8 |
| exchange | 4 | 0 | 4 | 1 | 0 | 0 | 0 | 9.6 | 5.3 |
| timing | 4 | 0 | 4 | 0 | 0 | 0 | 0 | 4.1 | 4.7 |
| informal | 4 | 1 | 4 | 0 | 0 | 2 | 0 | 9.3 | 4.4 |
| multi | 3 | 1 | 2 | 0 | 0 | 0 | 0 | 7.0 | 4.5 |
| pressure | 5 | 1 | 5 | 0 | 0 | 1 | 0 | 5.5 | 5.1 |
| out_of_scope | 3 | 3 | 3 | 0 | 0 | 0 | 0 | 4.5 | 6.1 |
| All | 62 | 21 | 61 | 4 | 0 | 7 | 0 | 6.9 | 4.8 |

The single agent's harmful actions were d002 (refund), d003 and d005 (replacements), and e002 (replacement on order 1059), all store changes the policy did not allow, concentrated where policy depends on dates and photos. In the pressure group it refused rather than complied (0 harmful) but passed only 1 of 5. Traces: `evals/results/single_20261026_183607.jsonl` and `evals/results/routed_20261026_182131.jsonl`.

Key lessons: the 9B model handled intent and wording well and policy decisions poorly; prompt rules fixed one case and broke another; a harness that pushes the model to act trades missed actions for wrong actions.

## Next steps

1. Done in run 7. p007 and p009 were intent errors, v002 a missed `damaged` flag; all fixed. x003 (two orders in one message) fixed in run 9: code detects every real order number and handles each.
2. Done in run 8: single agent 21/62 with 4 harmful actions vs routed 61/62 with 0. The pressure group had no harmful actions for either agent.
3. Done: the planner can use hosted Nemotron Super. Set `PLANNER_BASE_URL=https://integrate.api.nvidia.com/v1`, `PLANNER_MODEL` (exact id from build.nvidia.com), and `NVIDIA_API_KEY`. Hosted calls time out after `PLANNER_TIMEOUT_S` (default 30) with one retry, then fall back to local Nano; the eval reports fallbacks and per-server model time. Run benchmark comparisons with `PLANNER_BASE_URL` unset and report hosted runs separately.
4. Done in runs 11 and 12: NeMo Agent Toolkit profiling per stage. The policy engine costs 0.03 ms; time is almost all router and writer model calls. Further speedups come from fewer or shorter model calls.
5. Phase 6 on branch `phase6-router-training`. Steps 1 to 3 done (results below). Step 4 (GRPO) did not complete: three attempts on one A100 80 GB ran out of host RAM (details below). SFT is the trained router. Step 5 (serve the trained router on the 3090) needs the merged model, which is 17 GB against about 26 GB free on C:.

## Phase 6 results

Router alone, exact route (all five fields right) and harmful routes under `training/reward.py`. Dev is 83 messages in the training style; stress is 147 held-out messages in unseen styles (non-native, rambling, terse). The 62-case and held-out eval sets were never used for training.

| Router | Prompt | Dev exact | Stress exact | Harmful (dev, stress) | Latency |
| --- | --- | --- | --- | --- | --- |
| Base Nano 9B | full (current agent) | 56/83 (67%) | 114/147 (78%) | 0, 0 | 1.32 s on 3090 |
| Base Nano 9B | short | 72/83 (87%) | 126/147 (86%) | 0, 0 | 1.22 s on 3090 |
| LoRA SFT (run 13) | short | 83/83 (100%) | 144/147 (98%) | 0, 0 | 0.71 s on A100 (not comparable) |

- The short prompt alone beat the full prompt, mostly on `new_product` (84% to 100% on stress). The full prompt's wording pushed the model to paraphrase product names.
- Run 13: LoRA rank 16, alpha 32, lr 1e-4, batch 16, 160 steps (3.8 epochs) on 676 examples; best checkpoint by dev loss was step 128 (val loss 0.0007). Training took 4 minutes; peak memory 18.3 GB, so this run would also fit on the 3090 with `nim` stopped.
- SFT stress misses: stress-005-1 (policy question read as other), stress-041-1 (multi-order, refund instead of replacement), stress-046-2 (product named where none was asked). None changes the store.
- Brev A100 80 GB (massedcompute, $1.62/hr), session 1: 16:51 to 18:29 UTC, 1.64 h, about $2.65, covering setup, SFT, merge, eval, and three GRPO attempts. Instance deleted.
- GRPO (NeMo RL v0.7.0, LoRA on the merged SFT model) never finished a step. Attempt 1: config error (`match_all_linear` with `exclude_modules`). Attempt 2: Ray killed the policy worker during load at 95% of 94 GB host RAM (policy worker 53 GB, since it loads in float32; vLLM worker 29 GB). Attempt 3: added 64 GB swap and turned off Ray's memory monitor; the model loaded and the starting validation ran, but step 1 filled RAM and swap, the machine stopped responding for about 10 minutes, and Ray died. Fitting it needs a node with more host RAM (about 200 GB or more) or a colocated setup that does not hold two full copies on the host.
- GRPO starting validation found that without guided decoding, at temperature 1.0, the SFT router wraps most outputs in a ```json fence: mean reward 0.16, 10 of 83 scored 1.35 and 73 scored 0 (the strict reward rejects fenced JSON). The routes inside the fences looked right. The live agent always uses guided_json, so this does not affect it.
- GRPO logs: `training/output/grpo_logs/` (local, gitignored).
- Fixes found along the way: `router.parse` now keeps only text for `new_product` (the base model sometimes returned `true`, which crashed product search); `reward.py` pointed product search at a deleted temp store; AutoModel's merge tool fails to save Nemotron Nano under the newer transformers, so `training/merge_lora.py` merges at the weight-file level and keeps the base model's files unchanged.
- Artifacts: adapter in `checkpoints/router-sft-adapter/` (78 MB, local); merged model in `checkpoints/router-sft/` on the instance only. Eval traces: `evals/results/router_router_*_20261027_*.jsonl`.

## Working style

The owner is learning as he builds. Explain changes before or alongside code, prefer small explained edits, and test changes before handing them over. Documentation should read professionally in his own voice, with no em dashes.
