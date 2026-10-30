"""Phase 4: NeMo Agent Toolkit (NAT) profiling for the routed agent.

NAT's profiler works on "intermediate steps": START and END events for each
function, LLM call, and workflow run. NAT emits them itself for workflows built
from its components. The routed agent is plain Python, so this module emits the
same events by hand:

    step("router")    marks one agent stage as a NAT function. Any LLM call made
                      inside it is recorded under that stage.
    NatLLMClient      wraps an OpenAI client and records each chat call as an
                      LLM_START / LLM_END pair with token usage.
    Collector         captures one case's events, framed as a workflow run.
    profile(...)      hands every case's events to NAT's ProfilerRunner, then
                      sums tokens and time per stage from NAT's own CSV output.

If NAT is not installed, step() and NatLLMClient do nothing, so the agent and
its tests run the same without it.
"""
from __future__ import annotations

import asyncio
import csv
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

try:
    from nat.builder.context import Context
    from nat.data_models.intermediate_step import IntermediateStepPayload, StreamEventData, UsageInfo
    from nat.data_models.intermediate_step import IntermediateStepType as Event
    from nat.data_models.token_usage import TokenUsageBaseModel
    NAT_AVAILABLE = True
except ImportError:
    NAT_AVAILABLE = False

# Report order. "lookup" covers order detection, order and product lookups, and policy clause reads.
STAGES = ["router", "lookup", "engine", "action", "planner", "writer"]


@contextmanager
def step(name: str):
    """Record the enclosed block as one NAT function, so its LLM calls and time are grouped under it."""
    if not NAT_AVAILABLE:
        yield
        return
    with Context.get().push_active_function(name, input_data=None):
        yield


class NatLLMClient:
    """Wraps a client and emits a NAT LLM_START / LLM_END pair, with token usage, for every chat call."""

    def __init__(self, client):
        self._client = client
        self.base_url = getattr(client, "base_url", None)  # TimedClient reads this to name the server
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        if not NAT_AVAILABLE:
            return self._client.chat.completions.create(**kwargs)
        model = kwargs.get("model")
        # One manager for both events: NAT pairs START and END inside a manager, and
        # Context.get() returns a new one on every call.
        manager = Context.get().intermediate_step_manager
        start = IntermediateStepPayload(event_type=Event.LLM_START, name=model, data=StreamEventData())
        manager.push_intermediate_step(start)
        usage, output = TokenUsageBaseModel(), None
        try:
            resp = self._client.chat.completions.create(**kwargs)
            if resp.usage:
                usage = TokenUsageBaseModel(prompt_tokens=resp.usage.prompt_tokens,
                                            completion_tokens=resp.usage.completion_tokens,
                                            total_tokens=resp.usage.prompt_tokens + resp.usage.completion_tokens)
            output = resp.choices[0].message.content
            return resp
        except Exception as exc:
            output = f"ERROR {type(exc).__name__}"
            raise
        finally:
            # Same UUID as the start event, so NAT pairs them. Always sent, even when the call failed.
            manager.push_intermediate_step(IntermediateStepPayload(
                UUID=start.UUID, event_type=Event.LLM_END, name=model, data=StreamEventData(output=output),
                usage_info=UsageInfo(token_usage=usage, num_llm_calls=1)))


class Collector:
    """Captures the NAT events for one case: `with Collector(case_id) as c: ...`, then read c.steps."""

    def __init__(self, case_id: str):
        self.case_id, self.steps = case_id, []

    def __enter__(self):
        if NAT_AVAILABLE:
            self._manager = Context.get().intermediate_step_manager
            self._sub = self._manager.subscribe(self.steps.append)
            self._start = IntermediateStepPayload(event_type=Event.WORKFLOW_START, name="routed_agent",
                                                  data=StreamEventData(input=self.case_id))
            self._manager.push_intermediate_step(self._start)
        return self

    def __exit__(self, *exc):
        if NAT_AVAILABLE:
            self._manager.push_intermediate_step(IntermediateStepPayload(
                UUID=self._start.UUID, event_type=Event.WORKFLOW_END, name="routed_agent",
                data=StreamEventData(input=self.case_id)))
            self._sub.unsubscribe()
        return False


def profile(all_steps: list[list], out_dir: Path) -> list[dict]:
    """Run NAT's ProfilerRunner over every case, then return one summary row per stage.

    NAT writes its standard outputs to out_dir: standardized_data_all.csv (one row per event),
    workflow_profiling_metrics.json, and workflow_profiling_report.txt (bottlenecks). The
    per-stage table is built from that CSV, so it uses the same numbers NAT reports.
    """
    from nat.data_models.profiler import BottleneckConfig, ProfilerConfig
    from nat.plugins.profiler.profile_runner import ProfilerRunner

    config = ProfilerConfig(base_metrics=True, compute_llm_metrics=True, csv_exclude_io_text=True,
                            bottleneck_analysis=BottleneckConfig(enable_simple_stack=True))
    asyncio.run(ProfilerRunner(config, out_dir).run(all_steps))

    with (out_dir / "standardized_data_all.csv").open() as f:
        rows = list(csv.DictReader(f))

    # Per case and stage: time from FUNCTION_START to FUNCTION_END, tokens and calls from LLM_END events.
    per_case: dict[tuple[str, str], dict] = {}
    starts: dict[str, float] = {}
    for r in rows:
        name, case, ts = r["function_name"], r["example_number"], float(r["event_timestamp"])
        if name not in STAGES:
            continue
        cell = per_case.setdefault((case, name), {"seconds": 0.0, "prompt": 0, "completion": 0, "llm_calls": 0})
        if r["event_type"] == "FUNCTION_START":
            starts[r["function_id"]] = ts
        elif r["event_type"] == "FUNCTION_END":
            cell["seconds"] += ts - starts.pop(r["function_id"])
        elif r["event_type"] == "LLM_END":
            cell["prompt"] += int(float(r["prompt_tokens"] or 0))
            cell["completion"] += int(float(r["completion_tokens"] or 0))
            cell["llm_calls"] += 1

    n_cases = len(all_steps)
    summary = []
    for stage in STAGES:
        cells = [c for (_, s), c in per_case.items() if s == stage]
        if not cells:
            continue
        secs = sorted(c["seconds"] for c in cells)
        n = len(cells)
        summary.append({
            "stage": stage, "cases": n, "of_cases": n_cases,
            "llm_calls": sum(c["llm_calls"] for c in cells),
            "avg_prompt_tokens": round(sum(c["prompt"] for c in cells) / n, 1),
            "avg_completion_tokens": round(sum(c["completion"] for c in cells) / n, 1),
            "avg_ms": round(1000 * sum(secs) / n, 2),
            "p95_ms": round(1000 * secs[min(n - 1, int(0.95 * n))], 2),
            "total_s": round(sum(secs), 2),
        })
    (out_dir / "per_stage_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
