"""Phase 6, step 1: write customer messages from specs, check them, and split them.

Each spec (training/specs.py) is sent to a writer model with a brief. The model
returns three messages in three styles. The label is the spec's route, so labels
are correct by construction, provided the message follows the brief. Automatic
checks drop messages that clearly do not.

Train and dev use one style prompt; the stress test uses a different one, so the
stress test also measures robustness to wording the router was never trained on.

Every model call is cached in training/output/cache/, so reruns cost nothing.

Run from /workspace (uses NVIDIA_API_KEY for hosted Nemotron Super by default):
    python training/generate_messages.py
    python training/generate_messages.py --local          # use local Nano instead
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from training.specs import EVAL_FILES, OUT as SPECS_PATH  # noqa: E402

DATA = ROOT / "training" / "data"
CACHE = ROOT / "training" / "output" / "cache"
HOSTED_URL = "https://integrate.api.nvidia.com/v1"
HOSTED_MODEL = os.getenv("GENERATOR_MODEL", "nvidia/nemotron-3.5-super-vl-120b-a12b")
LOCAL_MODEL = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"

COMMON_RULES = """Rules:
- Write as the customer, in first person, to an online store's support chat.
- Express exactly what the brief says: no extra requests, and nothing that changes the facts.
- When the brief names an order number, include it exactly (for example 1077, #1077, or order no. 1077). Never include any other four-digit number.
- Do not mention prices.
- Reply with JSON only: {"messages": ["...", "...", "..."]}"""

TRAIN_PROMPT = """/no_think You write realistic customer support messages for testing a support system.
Write three different messages for the brief, one in each style:
1. short and polite, one or two sentences
2. casual chat: lowercase, contractions, little punctuation
3. detailed and a bit frustrated, three to five sentences

""" + COMMON_RULES

STRESS_PROMPT = """/no_think You are helping stress-test a retail support classifier with messy, real-world customer writing.
Write three messages for the brief, each in a different voice:
1. a non-native English speaker: simple grammar, a few spelling mistakes and typos, missing articles
2. a long rambling message that wanders through unrelated personal details before and after the actual request
3. terse fragments with abbreviations (pls, thx, asap, w/, b/c), as if typed on a phone

""" + COMMON_RULES

BANNED_INDIRECT = re.compile(r"\b(damaged?|broken|defective|refunds?|returns?|returning|replace|replacement)\b", re.I)
_FOUR_DIGITS = re.compile(r"(?<!\d)(\d{4})(?!\d)")


def _cached_call(client: OpenAI, model: str, system: str, user: str) -> str:
    key = hashlib.sha256(json.dumps([model, system, user]).encode()).hexdigest()
    path = CACHE / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())["content"]
    # Hosted Super ignores /no_think and can spend the whole budget reasoning; this switch turns reasoning off.
    extra = {} if model == LOCAL_MODEL else {"chat_template_kwargs": {"enable_thinking": False}}
    resp = client.chat.completions.create(model=model, temperature=0.8, top_p=0.95, max_tokens=1500,
                                          extra_body=extra,
                                          messages=[{"role": "system", "content": system},
                                                    {"role": "user", "content": user}])
    content = resp.choices[0].message.content or ""
    if not content.strip():
        raise RuntimeError(f"empty reply (finish_reason={resp.choices[0].finish_reason})")  # never cache an empty reply
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"model": model, "system": system, "user": user, "content": content}))
    return content


def _parse(content: str) -> list[str]:
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    match = re.search(r"\{.*\}", content, re.DOTALL)
    try:
        messages = json.loads(match.group(0))["messages"] if match else []
    except (json.JSONDecodeError, KeyError, TypeError):
        return []
    return [m.strip() for m in messages if isinstance(m, str) and m.strip()]


def check(spec: dict, message: str) -> str | None:
    """Return a reason to drop the message, or None if it follows the brief."""
    expected = {o for o in [spec["route"]["order_id"], *spec.get("orders", [])] if o}
    found = set(_FOUR_DIGITS.findall(message))
    if found != expected:
        return "order numbers"
    if spec["group"] == "indirect" and BANNED_INDIRECT.search(message):
        return "banned word"
    new = spec["route"]["new_product"]
    if new:
        words = [w for w in re.findall(r"[a-z0-9]+", new.lower()) if len(w) > 1]
        hits = sum(w in message.lower() for w in words)
        if hits < max(1, len(words) - 1):  # allow one word to be dropped, such as the brand
            return "product not named"
    if spec["group"] == "injection_flags" and not re.search(r"damaged|override|ignore|SYSTEM|intent", message, re.I):
        return "injection missing"
    return None


def _ngrams(text: str, n: int = 6) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z0-9']+", text.lower())
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def eval_ngrams() -> set[tuple[str, ...]]:
    grams: set[tuple[str, ...]] = set()
    for path in EVAL_FILES:
        for line in path.read_text().splitlines():
            if line.strip():
                grams |= _ngrams(json.loads(line)["message"])
    return grams


def router_input(spec: dict, message: str) -> str:
    """The text the router sees. Multi-order messages get the same note run_multi adds."""
    if spec.get("focus_order"):
        return f"{message}\n\n(Classify only the request about order {spec['focus_order']}.)"
    return message


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="write messages with local Nano instead of hosted Super")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    if args.local:
        client = OpenAI(base_url=os.getenv("OPENAI_BASE_URL", "http://nim:8000/v1"), api_key="local", timeout=120)
        model = LOCAL_MODEL
    else:
        client = OpenAI(base_url=HOSTED_URL, api_key=os.environ["NVIDIA_API_KEY"], timeout=45, max_retries=6)
        model = HOSTED_MODEL

    specs = [json.loads(line) for line in SPECS_PATH.read_text().splitlines() if line.strip()]

    done = [0]

    def write(spec: dict) -> tuple[dict, list[str], str | None]:
        system = STRESS_PROMPT if spec["split"] == "stress" else TRAIN_PROMPT
        try:
            result = spec, _parse(_cached_call(client, model, system, f"Brief: {spec['brief']}")), None
        except Exception as exc:  # keep going; failed specs are reported and can be rerun from cache
            result = spec, [], f"{type(exc).__name__}: {exc}"[:200]
        done[0] += 1
        if done[0] % 25 == 0:
            print(f"  {done[0]}/{len(specs)} specs written", flush=True)
        return result

    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(write, specs))

    banned = eval_ngrams()
    seen: set[str] = set()
    dropped: Counter = Counter()
    rows: dict[str, list[dict]] = {"train": [], "dev": [], "stress": []}
    errors = [(s["spec_id"], e) for s, _, e in results if e]
    styles = ["polite", "casual", "frustrated"]
    stress_styles = ["non_native", "rambling", "terse"]
    for spec, messages, _ in results:
        if not messages:
            dropped["no output"] += 1
            continue
        for i, message in enumerate(messages[:3]):
            norm = " ".join(message.lower().split())
            reason = check(spec, message)
            if reason is None and _ngrams(message) & banned:
                reason = "overlaps an eval message"
            if reason is None and norm in seen:
                reason = "duplicate"
            if reason:
                dropped[reason] += 1
                continue
            seen.add(norm)
            rows[spec["split"]].append({
                "id": f"{spec['spec_id']}-{i}", "spec_id": spec["spec_id"], "group": spec["group"],
                "style": (stress_styles if spec["split"] == "stress" else styles)[i] if i < 3 else "extra",
                "message": message, "router_input": router_input(spec, message),
                "expected_route": spec["route"], "generator": model,
            })

    names = {"train": "router_train.jsonl", "dev": "router_dev.jsonl", "stress": "router_stress_test.jsonl"}
    for split, name in names.items():
        (DATA / name).write_text("".join(json.dumps(r) + "\n" for r in rows[split]))
    print(f"Writer: {model}")
    print("Kept:", {s: len(r) for s, r in rows.items()})
    print("Dropped:", dict(dropped))
    if errors:
        print(f"{len(errors)} specs failed (rerun to retry; finished calls are cached):", errors[:3])


if __name__ == "__main__":
    main()
