"""Keyword search over the numbered policy clauses in data/policy.md.

Deliberately simple (TF-IDF style scoring, no vector DB) so Phase 1 has no
extra dependencies. Swap in NeMo Retriever or a vector DB later and compare.
"""
from __future__ import annotations

import math
import os
import re
from functools import lru_cache
from pathlib import Path

DEFAULT_POLICY = Path(__file__).resolve().parent.parent / "data" / "policy.md"
_HEADING = re.compile(r"^###\s+(\d+\.\d+)\s+(.*)$")
_WORD = re.compile(r"[a-z0-9$]+")
_STOP = {"a", "an", "and", "are", "be", "can", "do", "for", "how", "i", "if", "in", "is",
         "it", "my", "of", "on", "or", "the", "they", "this", "to", "we", "what", "with", "you"}


def _stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def _tokens(text: str) -> list[str]:
    return [_stem(w) for w in _WORD.findall(text.lower()) if w not in _STOP]


def _policy_path() -> Path:
    return Path(os.getenv("STORE_POLICY", DEFAULT_POLICY))


@lru_cache(maxsize=4)
def _load(path_str: str) -> tuple[dict, ...]:
    clauses: list[dict] = []
    for line in Path(path_str).read_text().splitlines():
        line = line.strip()
        match = _HEADING.match(line)
        if match:
            clauses.append({"clause_id": match.group(1), "title": match.group(2).strip(), "lines": []})
        elif clauses and line and not line.startswith("#"):
            clauses[-1]["lines"].append(line)
    return tuple({"clause_id": c["clause_id"], "title": c["title"], "text": " ".join(c["lines"])}
                 for c in clauses)


def search_policy(query: str, k: int = 3) -> dict:
    """Return the k policy clauses most relevant to the query, with clause ids for citation."""
    clauses = _load(str(_policy_path()))
    docs = [_tokens(c["title"]) * 2 + _tokens(c["text"]) for c in clauses]
    df: dict[str, int] = {}
    for doc in docs:
        for term in set(doc):
            df[term] = df.get(term, 0) + 1
    n = len(docs)
    scored = []
    for clause, doc in zip(clauses, docs):
        score = sum(doc.count(t) * math.log(1 + n / df[t]) for t in set(_tokens(query)) if t in df)
        if score > 0:
            scored.append((score, clause))
    scored.sort(key=lambda x: -x[0])
    return {"query": query,
            "results": [{**c, "score": round(s, 3)} for s, c in scored[:k]]}


def get_clause(clause_id: str) -> dict:
    """Return one clause by id, for example '4.2'."""
    for clause in _load(str(_policy_path())):
        if clause["clause_id"] == clause_id.strip():
            return {"found": True, **clause}
    return {"found": False, "error": f"Clause {clause_id} not found"}
