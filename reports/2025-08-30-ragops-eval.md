# RAGOps 最新评测报告（2025‑08‑30）

## RAG 评测背景与指标

RAG（Retrieval‑Augmented Generation）通过在大模型生成答案时检索外部知识库来增加上下文，从而提升个性化和时效性【312886005649155†L74-L88】。RAG 系统通常由两部分组成：检索器负责向知识库查询相关内容，生成器根据检索结果构建提示并生成最终答案【312886005649155†L101-L114】。因此，评测 RAG 系统的关键是分别衡量检索器和生成器的表现【312886005649155†L109-L114】。

一些通用的 RAG 评测指标包括【312886005649155†L118-L156】：

- **Contextual Recall**：检索到的上下文中包含多少与期望答案相关的内容。需要将期望答案作为标签，衡量检索器是否覆盖了正确的信息【312886005649155†L125-L130】。
- **Contextual Precision**：检索到的上下文中相关内容的占比，反映检索器对相关性排序的能力【312886005649155†L134-L140】。
- **Answer Relevancy**：生成的答案与期望答案的相关度，通常与检索器的质量直接相关【312886005649155†L142-L149】。
- **Faithfulness**：生成的答案是否忠实于检索到的上下文，衡量生成器的事实正确性【312886005649155†L151-L156】。

这些指标不仅适用于通用场景，还可根据业务需求扩展。例如在金融助手场景可能需要增加“偏见”等定制指标【312886005649155†L163-L169】。

## 测评方法

本次测试使用了 5 条简单的问答对，数据集每条包含**查询**、**期望答案**、**检索到的上下文**（3 条）以及**模型生成的答案**。通过自编脚本对每条数据计算上述四个指标，最后取平均值。为了简单起见，我们假设每个问题只有一个相关上下文。

计算方法：

- **Contextual Recall**：每条问题检索到的相关上下文数量 ÷ 理论相关上下文数量（1）。为避免分数超过 1，取结果的最小值为 1。
- **Contextual Precision**：每条问题检索到的相关上下文数量 ÷ 检索上下文总数（3）。
- **Answer Relevancy**：如果生成答案包含期望答案则记为 1，否则为 0。
- **Faithfulness**：如果生成答案可以在检索到的某条上下文中找到则记为 1，否则为 0。

## 评测结果

| 指标 | 平均值 |
| --- | --- |
| Contextual Recall | 1.00 |
| Contextual Precision | 0.33 |
| Answer Relevancy | 1.00 |
| Faithfulness | 0.60 |

**分析：**

- **高召回、低精度**：所有问题都能检索到包含正确答案的上下文，因此召回率为 1.00；但每次检索 3 条上下文时只有 1 条是相关的，导致精度只有 0.33。这表明检索器会返回一些无关文档，影响了后续生成质量。
- **答案相关度高**：生成的答案都包含期望答案，说明生成器能够使用检索到的内容提供准确答案。
- **忠实度一般**：只有 60% 的生成答案在检索到的原文中完全出现。这意味着生成器在部分情况下进行了改写或扩展，可能引入了未在上下文中存在的信息，需要在后续迭代中加强检索约束或加大人工验证。

## Python 演示代码

以下示例脚本展示如何构建数据集并计算上述指标。可保存为 `rag_evaluation_demo.py` 执行。

```python
# coding: utf-8
# 示例数据集与评测脚本

dataset = [
    {
        "query": "What is the capital of France?",
        "expected_answer": "Paris",
        "retrieved_contexts": [
            "Paris is the capital of France.",
            "France is located in Europe.",
            "The population of the city is about 2 million."
        ],
        "predicted_answer": "Paris is the capital of France."
    },
    {
        "query": "Who wrote the novel 1984?",
        "expected_answer": "George Orwell",
        "retrieved_contexts": [
            "1984 is a dystopian novel by George Orwell.",
            "It was published in 1949.",
            "Animal Farm is another book by the same author."
        ],
        "predicted_answer": "George Orwell wrote 1984."
    },
    {
        "query": "What is the largest planet in our solar system?",
        "expected_answer": "Jupiter",
        "retrieved_contexts": [
            "Jupiter is the largest planet in the solar system.",
            "Mars is smaller than Earth.",
            "Saturn has large rings."
        ],
        "predicted_answer": "Jupiter"
    },
    {
        "query": "When did the Second World War end?",
        "expected_answer": "1945",
        "retrieved_contexts": [
            "World War II ended in 1945.",
            "The war started in 1939.",
            "It involved many nations around the globe."
        ],
        "predicted_answer": "It ended in 1945."
    },
    {
        "query": "Who discovered penicillin?",
        "expected_answer": "Alexander Fleming",
        "retrieved_contexts": [
            "Alexander Fleming discovered penicillin in 1928.",
            "Penicillin was the first true antibiotic.",
            "The discovery revolutionized medicine."
        ],
        "predicted_answer": "Alexander Fleming"
    }
]


def evaluate(dataset):
    total_recall = 0
    total_precision = 0
    total_answer_rel = 0
    total_faithfulness = 0
    for item in dataset:
        expected = item["expected_answer"].lower()
        contexts = item["retrieved_contexts"]
        # 统计包含期望答案的上下文数量
        relevant_contexts = sum(1 for c in contexts if expected in c.lower())
        # 按一次只有一个相关上下文假设，召回率最大为 1
        recall = min(relevant_contexts / 1, 1)
        precision = relevant_contexts / len(contexts)
        predicted = item["predicted_answer"].lower()
        answer_rel = 1 if expected in predicted else 0
        faithfulness = 1 if any(
            predicted in c.lower() or c.lower() in predicted for c in contexts
        ) else 0
        total_recall += recall
        total_precision += precision
        total_answer_rel += answer_rel
        total_faithfulness += faithfulness
    n = len(dataset)
    return {
        "contextual_recall": round(total_recall / n, 2),
        "contextual_precision": round(total_precision / n, 2),
        "answer_relevancy": round(total_answer_rel / n, 2),
        "faithfulness": round(total_faithfulness / n, 2)
    }


if __name__ == "__main__":
    metrics = evaluate(dataset)
    print("Evaluation metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")
```

## 下一步优化建议

1. **提高检索精度**：目前检索器返回的上下文中只有三分之一是相关的，可尝试使用更强的向量模型或引入重排序器提高相关性。
2. **提升忠实度**：在生成时采用基于检索上下文的约束（如 `chain-of-thought + retrieval` 或 `answer citing`）减少模型自由发挥的部分。
3. **扩展评测指标**：根据具体场景引入偏见检测、结构化输出验证等指标【312886005649155†L163-L169】。

该报告将随每周迭代更新，为改进 RAG 系统提供数据支持。
