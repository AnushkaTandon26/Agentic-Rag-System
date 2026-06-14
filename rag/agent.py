"""Autonomous retrieval planning, evidence evaluation, and context engineering."""

from __future__ import annotations

import re
from typing import Any


class AgenticRAGOrchestrator:
    """Run a bounded agentic workflow around an RBAC-aware retriever."""

    def __init__(
        self,
        retriever: Any,
        max_subqueries: int = 3,
        max_context_chunks: int = 6,
        max_context_chars: int = 3000,
        retry_threshold: float = 0.35,
    ) -> None:
        self.retriever = retriever
        self.max_subqueries = max_subqueries
        self.max_context_chunks = max_context_chunks
        self.max_context_chars = max_context_chars
        self.retry_threshold = retry_threshold

    def run(self, question: str, user_id: str, rbac_manager: Any) -> dict[str, Any]:
        """Plan retrieval, evaluate evidence, retry when weak, and build grounded context."""
        trace: list[dict[str, str]] = []
        subqueries = self._plan_subqueries(question)
        trace.append(self._trace("plan", "completed", f"Created {len(subqueries)} retrieval task(s)."))

        all_chunks: list[dict[str, Any]] = []
        blocked_resources: set[str] = set()
        top_result_blocked = False

        for index, subquery in enumerate(subqueries):
            chunks = self.retriever.retrieve(subquery, user_id, rbac_manager, top_k=5)
            all_chunks.extend(chunks)
            blocked_resources.update(self.retriever.last_blocked_resources)
            if index == 0:
                top_result_blocked = self.retriever.last_top_result_blocked
            trace.append(
                self._trace("retrieve", "completed", f"Retrieved {len(chunks)} authorized chunk(s) for task {index + 1}.")
            )
            if top_result_blocked:
                trace.append(self._trace("guardrail", "blocked", "The primary answer source is restricted."))
                break

        evidence_quality = 0.0 if top_result_blocked else self._evaluate_evidence(all_chunks)
        trace.append(self._trace("evaluate", "completed", f"Evidence quality score is {evidence_quality:.3f}."))

        retried = False
        if not top_result_blocked and evidence_quality < self.retry_threshold:
            retried = True
            retry_chunks = self.retriever.retrieve(self._rewrite_query(question), user_id, rbac_manager, top_k=8)
            all_chunks.extend(retry_chunks)
            blocked_resources.update(self.retriever.last_blocked_resources)
            trace.append(
                self._trace("retry", "completed", f"Broadened search retrieved {len(retry_chunks)} authorized chunk(s).")
            )
            evidence_quality = self._evaluate_evidence(all_chunks)

        context_chunks = [] if top_result_blocked else self._engineer_context(all_chunks)
        trace.append(
            self._trace("context", "completed", f"Selected {len(context_chunks)} diverse chunk(s) for generation.")
        )
        return {
            "chunks": context_chunks,
            "subqueries": subqueries,
            "evidence_quality": evidence_quality,
            "retrieval_retried": retried,
            "reasoning_trace": trace,
            "top_result_blocked": top_result_blocked,
            "blocked_resources": sorted(blocked_resources),
            "unauthorized_sources_blocked": len(blocked_resources),
        }

    def _plan_subqueries(self, question: str) -> list[str]:
        tasks = [question.strip()]
        normalized_question = question.strip(" ,?.").lower()
        clauses = re.split(r"\s+(?:and|also|versus|vs\.?)\s+|[;]", question, flags=re.IGNORECASE)
        for clause in clauses:
            clause = clause.strip(" ,?.")
            if len(clause.split()) >= 2 and clause.lower() != normalized_question:
                tasks.append(clause)
        return self._unique(tasks)[: self.max_subqueries]

    def _engineer_context(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        deduplicated: dict[str, dict[str, Any]] = {}
        for chunk in chunks:
            doc_id = chunk["metadata"].get("doc_id", chunk["content"])
            if doc_id not in deduplicated or chunk.get("score", 0.0) > deduplicated[doc_id].get("score", 0.0):
                deduplicated[doc_id] = chunk

        ranked = sorted(deduplicated.values(), key=lambda item: item.get("score", 0.0), reverse=True)
        if ranked:
            relevance_floor = ranked[0].get("score", 0.0) * 0.45
            ranked = [chunk for chunk in ranked if chunk.get("score", 0.0) >= relevance_floor]
        selected: list[dict[str, Any]] = []
        deferred: list[dict[str, Any]] = []
        seen_resources: set[str] = set()
        used_chars = 0
        for chunk in ranked:
            resource = chunk["metadata"].get("resource", "")
            if resource in seen_resources:
                deferred.append(chunk)
                continue
            if self._fits_context(chunk, used_chars, selected):
                selected.append(chunk)
                seen_resources.add(resource)
                used_chars += len(chunk["content"])

        for chunk in deferred:
            if self._fits_context(chunk, used_chars, selected):
                selected.append(chunk)
                used_chars += len(chunk["content"])
        return selected

    def _fits_context(self, chunk: dict[str, Any], used_chars: int, selected: list[dict[str, Any]]) -> bool:
        return len(selected) < self.max_context_chunks and used_chars + len(chunk["content"]) <= self.max_context_chars

    @staticmethod
    def _evaluate_evidence(chunks: list[dict[str, Any]]) -> float:
        if not chunks:
            return 0.0
        unique: dict[str, float] = {}
        for chunk in chunks:
            doc_id = chunk["metadata"].get("doc_id", chunk["content"])
            unique[doc_id] = max(unique.get(doc_id, 0.0), float(chunk.get("score", 0.0)))
        top_scores = sorted(unique.values(), reverse=True)[:3]
        return round(min(1.0, sum(min(1.0, score) for score in top_scores) / len(top_scores)), 3)

    @staticmethod
    def _rewrite_query(question: str) -> str:
        stop_words = {"what", "which", "who", "when", "where", "why", "how", "is", "are", "the", "a", "an", "me"}
        terms = [
            token for token in re.findall(r"[a-zA-Z0-9_]+", question.lower()) if token not in stop_words
        ]
        return " ".join(terms + ["policy", "requirements", "details"])

    @staticmethod
    def _unique(values: list[str]) -> list[str]:
        seen = set()
        unique = []
        for value in values:
            normalized = value.lower()
            if normalized not in seen:
                seen.add(normalized)
                unique.append(value)
        return unique

    @staticmethod
    def _trace(step: str, status: str, detail: str) -> dict[str, str]:
        return {"step": step, "status": status, "detail": detail}
