"""End-to-end enterprise RAG pipeline with RBAC enforcement."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .agent import AgenticRAGOrchestrator
from .generator import AnswerGenerator
from .ingestion import DataIngestionPipeline
from .rbac import RBACManager
from .retrieval import HybridRetriever


class EnterpriseRAGPipeline:
    """Coordinate ingestion, retrieval, RBAC filtering, and answer generation."""

    def __init__(self, base_dir: str | Path | None = None) -> None:
        """Initialize RBAC, ingestion, retrieval, and generation components."""
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).resolve().parents[1]
        self.rbac_manager = RBACManager(self.base_dir)
        self.ingestion_pipeline = DataIngestionPipeline(self.base_dir)
        self.retriever = HybridRetriever()
        self.generator = AnswerGenerator()
        self.agent = AgenticRAGOrchestrator(self.retriever)
        self.documents: list[dict[str, Any]] = []

    def setup(self) -> None:
        """Ingest every source and build the hybrid retrieval index."""
        self.documents = self.ingestion_pipeline.ingest_all()
        self.retriever.build_index(self.documents)

    def query(self, user_id: str, question: str) -> dict[str, Any]:
        """Run an RBAC-aware query and return answer, citation, confidence, and filtering details."""
        user_role = self.rbac_manager.get_user_role(user_id)
        agent_result = self.agent.run(question, user_id, self.rbac_manager)
        retrieved_chunks = agent_result["chunks"]
        blocked_resources = agent_result["blocked_resources"]
        rbac_filtered = agent_result["unauthorized_sources_blocked"] > 0
        top_result_blocked = agent_result["top_result_blocked"]
        generation = self.generator.generate(question, retrieved_chunks, user_id, user_role)

        if top_result_blocked:
            blocked_list = ", ".join(sorted(set(blocked_resources)))
            generation["answer"] = f"Access denied: the most relevant sources are restricted for role '{user_role}'. Blocked resources: {blocked_list}."
            generation["citations"] = []
            generation["confidence"] = 0.0
            generation["hallucination_risk"] = "LOW"

        return {
            "query": question,
            "user_id": user_id,
            "user_role": user_role,
            "answer": generation["answer"],
            "citations": generation["citations"],
            "confidence": generation["confidence"],
            "sources_accessed": sorted({chunk["metadata"]["resource"] for chunk in retrieved_chunks}),
            "rbac_filtered": rbac_filtered,
            "unauthorized_sources_blocked": agent_result["unauthorized_sources_blocked"],
            "hallucination_risk": generation["hallucination_risk"],
            "agentic_workflow": {
                "subqueries": agent_result["subqueries"],
                "retrieval_retried": agent_result["retrieval_retried"],
                "evidence_quality": agent_result["evidence_quality"],
                "reasoning_trace": agent_result["reasoning_trace"],
            },
        }
