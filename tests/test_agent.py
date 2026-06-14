"""Tests for the bounded agentic retrieval workflow."""

from __future__ import annotations

import unittest

from rag.agent import AgenticRAGOrchestrator


class FakeRetriever:
    def __init__(self, responses: list[list[dict]]) -> None:
        self.responses = responses
        self.last_blocked_resources: list[str] = []
        self.last_unauthorized_sources_blocked = 0
        self.last_top_result_blocked = False

    def retrieve(self, query: str, user_id: str, rbac_manager: object, top_k: int = 5) -> list[dict]:
        return self.responses.pop(0) if self.responses else []


def chunk(doc_id: str, resource: str, score: float) -> dict:
    return {
        "content": "Grounded enterprise evidence.",
        "metadata": {"doc_id": doc_id, "resource": resource, "source": f"{resource}.txt", "chunk": 1},
        "score": score,
    }


class AgenticRAGOrchestratorTests(unittest.TestCase):
    def test_plans_compound_question_and_builds_diverse_context(self) -> None:
        retriever = FakeRetriever(
            [
                [chunk("finance-1", "finance", 0.9), chunk("finance-2", "finance", 0.8)],
                [chunk("finance-1", "finance", 0.7)],
                [chunk("projects-1", "projects", 0.85)],
            ]
        )
        result = AgenticRAGOrchestrator(retriever).run(
            "What was Q1 revenue and budget allocation?", "U002", object()
        )
        self.assertEqual(3, len(result["subqueries"]))
        self.assertEqual(["finance", "projects"], [item["metadata"]["resource"] for item in result["chunks"][:2]])
        self.assertFalse(result["retrieval_retried"])

    def test_retries_when_initial_evidence_is_weak(self) -> None:
        retriever = FakeRetriever([[], [chunk("hr-1", "hr_policy", 0.9)]])
        result = AgenticRAGOrchestrator(retriever).run("Explain leave rules", "U001", object())
        self.assertTrue(result["retrieval_retried"])
        self.assertEqual(1, len(result["chunks"]))

    def test_stops_when_primary_source_is_restricted(self) -> None:
        retriever = FakeRetriever([[chunk("finance-1", "finance_report", 0.9)]])
        retriever.last_top_result_blocked = True
        retriever.last_blocked_resources = ["finance_report"]
        retriever.last_unauthorized_sources_blocked = 1
        result = AgenticRAGOrchestrator(retriever).run("Show me the finance report", "U008", object())
        self.assertTrue(result["top_result_blocked"])
        self.assertEqual([], result["chunks"])
        self.assertFalse(result["retrieval_retried"])


if __name__ == "__main__":
    unittest.main()
