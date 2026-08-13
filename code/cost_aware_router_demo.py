#!/usr/bin/env python3
"""
Cost-aware LLM routing demo.

Demonstrates a simple LLMOps pattern:
request classification -> model routing -> cost optimization.
"""


def classify_request(prompt: str) -> str:
    """Classify request complexity."""
    complex_keywords = ["architecture", "debug", "design", "analysis"]
    text = prompt.lower()
    return "large" if any(k in text for k in complex_keywords) else "small"


def route_model(prompt: str) -> dict:
    """Route simple requests to cheaper models."""
    level = classify_request(prompt)
    if level == "large":
        return {"model": "reasoning-model", "tier": "large"}
    return {"model": "fast-model", "tier": "small"}


if __name__ == "__main__":
    samples = [
        "Translate this sentence",
        "Design a Kubernetes build platform architecture",
    ]
    for sample in samples:
        print(sample, "=>", route_model(sample))
