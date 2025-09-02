# coding: utf-8
"""
Demo script to evaluate simple RAG retrieval and generation performance.
This script defines a small dataset of queries, expected answers, retrieved contexts,
and predicted answers, then computes RAG metrics:
  - contextual_recall: fraction of relevant contexts retrieved (capped at 1.0)
  - contextual_precision: fraction of retrieved contexts that are relevant
  - answer_relevancy: proportion of predicted answers containing expected answer
  - faithfulness: proportion of predicted answers that appear in a retrieved context

Run this script directly to see evaluation results.
"""

from typing import List, Dict

# Example dataset
DATASET: List[Dict[str, object]] = [
    {
        "query": "What is the capital of France?",
        "expected_answer": "Paris",
        "retrieved_contexts": [
            "Paris is the capital of France.",
            "France is located in Europe.",
            "The population of the city is about 2 million.",
        ],
        "predicted_answer": "Paris is the capital of France.",
    },
    {
        "query": "Who wrote the novel 1984?",
        "expected_answer": "George Orwell",
        "retrieved_contexts": [
            "1984 is a dystopian novel by George Orwell.",
            "It was published in 1949.",
            "Animal Farm is another book by the same author.",
        ],
        "predicted_answer": "George Orwell wrote 1984.",
    },
    {
        "query": "What is the largest planet in our solar system?",
        "expected_answer": "Jupiter",
        "retrieved_contexts": [
            "Jupiter is the largest planet in the solar system.",
            "Mars is smaller than Earth.",
            "Saturn has large rings.",
        ],
        "predicted_answer": "Jupiter",
    },
    {
        "query": "When did the Second World War end?",
        "expected_answer": "1945",
        "retrieved_contexts": [
            "World War II ended in 1945.",
            "The war started in 1939.",
            "It involved many nations around the globe.",
        ],
        "predicted_answer": "It ended in 1945.",
    },
    {
        "query": "Who discovered penicillin?",
        "expected_answer": "Alexander Fleming",
        "retrieved_contexts": [
            "Alexander Fleming discovered penicillin in 1928.",
            "Penicillin was the first true antibiotic.",
            "The discovery revolutionized medicine.",
        ],
        "predicted_answer": "Alexander Fleming",
    },
]

def evaluate(dataset: List[Dict[str, object]]) -> Dict[str, float]:
    """Compute simple RAG evaluation metrics over a dataset."""
    total_recall = 0.0
    total_precision = 0.0
    total_answer_rel = 0.0
    total_faithfulness = 0.0

    for item in dataset:
        expected = str(item["expected_answer"]).lower()
        contexts: List[str] = item["retrieved_contexts"]
        # Count the number of contexts containing the expected answer
        relevant_contexts = sum(1 for c in contexts if expected in c.lower())
        # Assuming one relevant context should be retrieved, recall is capped at 1
        recall = min(relevant_contexts / 1.0, 1.0)
        # Precision: fraction of retrieved contexts that are relevant
        precision = relevant_contexts / len(contexts) if contexts else 0.0
        predicted = str(item["predicted_answer"]).lower()
        # Answer relevancy: check if expected answer appears in the predicted answer
        answer_rel = 1.0 if expected in predicted else 0.0
        # Faithfulness: predicted answer should appear in one of the retrieved contexts
        faithfulness = 1.0 if any(
            predicted in c.lower() or c.lower() in predicted for c in contexts
        ) else 0.0

        total_recall += recall
        total_precision += precision
        total_answer_rel += answer_rel
        total_faithfulness += faithfulness

    n = float(len(dataset)) or 1.0
    return {
        "contextual_recall": round(total_recall / n, 2),
        "contextual_precision": round(total_precision / n, 2),
        "answer_relevancy": round(total_answer_rel / n, 2),
        "faithfulness": round(total_faithfulness / n, 2),
    }


def main() -> None:
    """Print evaluation results for the example dataset."""
    metrics = evaluate(DATASET)
    print("Evaluation metrics:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value}")


if __name__ == "__main__":
    main()
