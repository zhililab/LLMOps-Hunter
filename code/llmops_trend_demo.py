#!/usr/bin/env python3
"""
LLMOps Trend Demo
-----------------
This script simulates a tiny retrieval‑augmented generation (RAG) evaluation.  It
answers a set of predefined questions using a toy corpus, measures how well the
answers align with the retrieved context (groundedness), and computes simple
latency and cost proxies.  The results are written to a markdown report in
``reports/<date>-ragops-eval.md`` relative to the project root.

Usage::

    python code/llmops_trend_demo.py

The script has no external dependencies besides the Python standard library.
"""

import os
import re
import time
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple


# -----------------------------------------------------------------------------
# Toy corpus and questions
# -----------------------------------------------------------------------------
CORPUS: List[Dict[str, str]] = [
    {
        "id": "doc1",
        "title": "LLM Evaluation Best Practices",
        "text": "Groundedness measures how well an LLM answer relies on the provided context. "
                "Toxicity, latency and cost are also common evaluation metrics."
    },
    {
        "id": "doc2",
        "title": "Cost Optimization",
        # Use a single literal so that the string is not split across lines.
        "text": (
            "To reduce inference cost, use smaller models and caching. "
            "Selective routing can pick the right model size on demand."
        )
    },
    {
        "id": "doc3",
        "title": "Agentic RAG",
        "text": (
            "Agentic RAG adds planning and tool use to retrieval‑augmented generation. "
            "It often re‑queries or verifies intermediate results."
        )
    },
]

QUESTIONS: List[Dict[str, str]] = [
    {
        "id": "q1",
        "question": "What does groundedness measure in LLM evaluation?",
        "expected": "It measures how much the answer relies on the provided context."
    },
    {
        "id": "q2",
        "question": "Name two methods to reduce inference cost.",
        "expected": "Use smaller models and caching."
    },
    {
        "id": "q3",
        "question": "What does Agentic RAG add to standard RAG?",
        "expected": "It adds planning and tool use."
    },
]


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric tokens."""
    return re.findall(r"[a-z0-9]+", text.lower())


def retrieve(question: str) -> Dict[str, str]:
    """Retrieve the most relevant document from the toy corpus based on keyword overlap."""
    q_tokens = set(tokenize(question))
    scored: List[Tuple[int, Dict[str, str]]] = []
    for doc in CORPUS:
        d_tokens = set(tokenize(doc["text"] + " " + doc["title"]))
        score = len(q_tokens & d_tokens)
        scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def answer_question(question: str, doc: Dict[str, str]) -> str:
    """Generate a simple answer based on keyword heuristics."""
    q = question.lower()
    if "groundedness" in q:
        return "It measures how much the answer relies on the provided context."
    if "reduce inference cost" in q or "reduce inference cost" in q or "reduce cost" in q:
        return "Use smaller models and caching."
    if "agentic rag" in q:
        return "It adds planning and tool use."
    # fallback: return the first sentence of the retrieved document
    return doc["text"].split(".")[0] + "."


def compute_groundedness(answer: str, context: str) -> float:
    """Compute a simple groundedness metric: fraction of answer tokens that appear in context."""
    ans_tokens = tokenize(answer)
    ctx_tokens = set(tokenize(context))
    if not ans_tokens:
        return 0.0
    overlap = sum(1 for t in ans_tokens if t in ctx_tokens)
    return overlap / len(ans_tokens)


def simulate_latency_and_cost(answer: str) -> Tuple[int, float]:
    """Simulate latency (ms) and cost (USD) based on answer length."""
    tokens = max(1, len(tokenize(answer)))
    latency = int(50 + 2.0 * tokens + random.randint(0, 20))
    cost = round(0.00001 * tokens, 6)
    return latency, cost


# -----------------------------------------------------------------------------
# Main evaluation function
# -----------------------------------------------------------------------------
def evaluate() -> List[Dict[str, str]]:
    """Evaluate all questions and return results as a list of dictionaries."""
    results: List[Dict[str, str]] = []
    for q in QUESTIONS:
        doc = retrieve(q["question"])
        ans = answer_question(q["question"], doc)
        g = compute_groundedness(ans, doc["text"])
        latency, cost = simulate_latency_and_cost(ans)
        results.append({
            "query_id": q["id"],
            "question": q["question"],
            "doc_id": doc["id"],
            "answer": ans,
            "groundedness": round(g, 3),
            "latency_ms": latency,
            "cost_usd": cost,
        })
    return results


def write_report(results: List[Dict[str, str]], report_dir: Path) -> None:
    """Write the evaluation results to a markdown file in the specified directory."""
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    report_path = report_dir / f"{date_str}-ragops-eval.md"
    # Compute averages
    avg_g = sum(r["groundedness"] for r in results) / len(results)
    avg_lat = sum(r["latency_ms"] for r in results) / len(results)
    avg_cost = sum(r["cost_usd"] for r in results) / len(results)
    # Build markdown
    lines = [
        f"# RAGOps Evaluation – {date_str}",
        "",
        "## Summary",
        f"- Average groundedness: **{avg_g:.3f}**",
        f"- Average latency (ms): **{avg_lat:.1f}**",
        f"- Average cost (USD): **{avg_cost:.6f}**",
        "",
        "## Detailed Results",
        "|Query|Doc|Groundedness|Latency (ms)|Cost (USD)|",
        "|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"|{r['query_id']}|{r['doc_id']}|{r['groundedness']:.3f}|{r['latency_ms']}|{r['cost_usd']:.6f}|"
        )
    lines.append("")
    # Write to file
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Report written to {report_path}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    report_dir = project_root / "reports"
    results = evaluate()
    write_report(results, report_dir)


if __name__ == "__main__":
    main()
