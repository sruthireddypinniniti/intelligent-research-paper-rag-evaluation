"""Evaluate the retriever and generator end-to-end with DeepEval."""

from __future__ import annotations

import json
import os
from pathlib import Path

from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase

from src.rag_pipeline import ResearchRAG, load_documents

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "evals" / "dataset.json"
THRESHOLD = float(os.getenv("DEEPEVAL_THRESHOLD", "0.70"))


def load_dataset() -> list[dict]:
    with DATASET.open(encoding="utf-8") as file:
        return json.load(file)


def build_test_cases() -> list[LLMTestCase]:
    rag = ResearchRAG()
    rag.index(load_documents(ROOT / "data"))

    cases = []
    for row in load_dataset():
        answer, context = rag.answer(row["input"], k=4)
        cases.append(
            LLMTestCase(
                input=row["input"],
                actual_output=answer,
                expected_output=row["expected_output"],
                retrieval_context=context,
            )
        )
    return cases


def main() -> None:
    if not os.getenv("GEMINI_API_KEY"):
        raise SystemExit("GEMINI_API_KEY is required for the LLM evaluation.")

    metrics = [
        AnswerRelevancyMetric(threshold=THRESHOLD, include_reason=True),
        FaithfulnessMetric(threshold=THRESHOLD, include_reason=True),
        ContextualRelevancyMetric(threshold=THRESHOLD, include_reason=True),
        ContextualPrecisionMetric(threshold=THRESHOLD, include_reason=True),
        ContextualRecallMetric(threshold=THRESHOLD, include_reason=True),
    ]

    evaluate(build_test_cases(), metrics)


if __name__ == "__main__":
    main()
