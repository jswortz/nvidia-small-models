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
docker run -d --name nim --restart unless-stopped --network poc-net --gpus all -p 8000:8000 --ipc=host -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -v C:\hf-cache:/root/.cache/huggingface -v C:\nim-plugins:/plugins vllm/vllm-openai:v0.10.1 --model nvidia/NVIDIA-Nemotron-Nano-9B-v2 --trust-remote-code --max-model-len 8192 --max-num-seqs 8 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.93 --enforce-eager --enable-auto-tool-choice --tool-parser-plugin /plugins/nemotron_toolcall_parser_no_streaming.py --tool-call-parser nemotron_json
```

`poc-dev` command:

```
docker run -dit --name poc-dev --restart unless-stopped --network poc-net -v C:\poc:/workspace -w /workspace -e OPENAI_BASE_URL=http://nim:8000/v1 -e OPENAI_API_KEY=local python:3.12 bash
```

Known environment gotchas:

- Pulling new images over SSH fails with a Windows credential error. Pull at the desktop; `run`, `start`, `logs`, and `exec` work over SSH.
- The 3090 runs out of memory with vLLM defaults. The flags above (`--max-num-seqs 8`, `--enforce-eager`, utilization 0.93) are what work. `--max-num-batched-tokens` must be at least `--max-model-len`.
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
agents/routed_agent.py    Phase 3: route, look up, decide, act in code, then Nano writes the reply. Escalations go to the planner;
                          a failed hosted planner call falls back to local Nano. Every model call is timed in result["model_calls"].
evals/run_eval.py         Scores tools, final store state, citations, honesty (false promises), harmful actions, router accuracy, per group
evals/cases_phase2.jsonl  Original 10 cases
evals/cases_phase5.jsonl  62 cases in 11 failure-mode groups
tests/                    smoke_test.py (15 checks), test_policy_engine.py (18 checks), test_router.py (7 checks), test_multi_order.py (7 checks),
                          test_planner_fallback.py (6 checks). Fake clients; no model needed.
```

## Commands (inside poc-dev, from /workspace)

```
python tests/smoke_test.py
python tests/test_policy_engine.py
python tests/test_router.py
python tests/test_multi_order.py
python tests/test_planner_fallback.py
python data/generate.py                                                   # reset the demo store
python -m agents.routed_agent "My blender from order 1187 arrived cracked..."
python evals/run_eval.py --agent routed --cases evals/cases_phase5.jsonl
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

Known gap: x003 sends two requests in one message; the routed agent handles one request per message.

## Next steps

1. Done in run 7. p007 and p009 were intent errors, v002 a missed `damaged` flag; all fixed. x003 remains (one request per message): detect two or more order numbers in code and handle or ask about each.
2. Done in run 8: single agent 21/62 with 4 harmful actions vs routed 61/62 with 0. The pressure group had no harmful actions for either agent.
3. Done: the planner can use hosted Nemotron Super. Set `PLANNER_BASE_URL=https://integrate.api.nvidia.com/v1`, `PLANNER_MODEL` (exact id from build.nvidia.com), and `NVIDIA_API_KEY`. Hosted calls time out after `PLANNER_TIMEOUT_S` (default 30) with one retry, then fall back to local Nano; the eval reports fallbacks and per-server model time. Run benchmark comparisons with `PLANNER_BASE_URL` unset and report hosted runs separately.
4. Phase 4: wrap the routed agent with NeMo Agent Toolkit (`nvidia-nat`) and profile tokens and latency per step.
5. Phase 6 (written plan): LoRA on router errors; RL from verifiable rewards using final store state as the reward.

## Working style

The owner is learning as he builds. Explain changes before or alongside code, prefer small explained edits, and test changes before handing them over. Documentation should read professionally in his own voice, with no em dashes.
