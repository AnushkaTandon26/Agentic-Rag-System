"""Hybrid semantic and keyword retrieval with RBAC filtering."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any

try:
    import chromadb
except Exception:  # pragma: no cover
    chromadb = None

try:
    from rank_bm25 import BM25Okapi
except Exception:  # pragma: no cover
    BM25Okapi = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None


class HybridRetriever:
    """Build and query semantic, keyword, and hybrid indexes."""

    RESOURCE_ALIASES = {
        "hr_policy": {"hr", "human resources", "leave", "employee policy"},
        "finance_report": {"finance", "financial", "revenue", "budget", "q1"},
        "it_security_policy": {"it security", "vpn", "security protocols", "remote access"},
        "product_roadmap": {"product", "roadmap", "release", "features"},
        "compliance_manual": {"compliance", "legal", "regulatory", "manual"},
        "employees": {"employees", "staff", "salary", "manager"},
        "projects": {"projects", "project", "budget", "project lead"},
        "system_logs": {"system logs", "system audit logs", "logs", "security logs"},
        "audit_trail": {"audit trail", "audit logs", "audit", "permission history"},
    }

    def __init__(self) -> None:
        """Initialize a ChromaDB client when available and prepare BM25 placeholders."""
        self.documents: list[dict[str, Any]] = []
        self.embeddings: list[list[float]] = []
        self.bm25 = None
        self.tokenized_corpus: list[list[str]] = []
        self.model = self._load_model()
        self.client = chromadb.Client() if chromadb is not None else None
        self.collection = None
        self.last_unauthorized_sources_blocked = 0
        self.last_blocked_resources: list[str] = []
        self.last_top_result_blocked = False

    def build_index(self, documents: list[dict[str, Any]]) -> None:
        """Embed documents with all-MiniLM-L6-v2 when available and store them in ChromaDB collection enterprise_docs."""
        self.documents = documents
        self.embeddings = [self._embed(document["content"]) for document in documents]
        self.tokenized_corpus = [self._tokenize(document["content"]) for document in documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus) if BM25Okapi is not None and self.tokenized_corpus else None

        if self.client is not None:
            try:
                self.client.delete_collection("enterprise_docs")
            except Exception:
                pass
            self.collection = self.client.get_or_create_collection("enterprise_docs")
            if documents:
                self.collection.add(
                    ids=[document["metadata"]["doc_id"] for document in documents],
                    documents=[document["content"] for document in documents],
                    metadatas=[document["metadata"] for document in documents],
                    embeddings=self.embeddings,
                )

    def semantic_search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Query ChromaDB or local vectors for semantically similar chunks."""
        query_embedding = self._embed(query)
        if self.collection is not None:
            try:
                result = self.collection.query(query_embeddings=[query_embedding], n_results=min(top_k, len(self.documents)))
                matches = []
                for index, content in enumerate(result.get("documents", [[]])[0]):
                    distance = result.get("distances", [[1.0]])[0][index]
                    metadata = result.get("metadatas", [[]])[0][index]
                    matches.append({"content": content, "metadata": metadata, "score": 1.0 / (1.0 + distance), "retrieval": "semantic"})
                return matches
            except Exception:
                pass

        scored = []
        for document, embedding in zip(self.documents, self.embeddings):
            scored.append({**document, "score": self._cosine(query_embedding, embedding), "retrieval": "semantic"})
        return sorted(scored, key=lambda item: item["score"], reverse=True)[:top_k]

    def keyword_search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Use BM25Okapi or a token-overlap fallback for keyword matching."""
        query_tokens = self._tokenize(query)
        if self.bm25 is not None:
            scores = self.bm25.get_scores(query_tokens)
        else:
            query_set = set(query_tokens)
            scores = [
                len(query_set.intersection(tokens)) / max(1, len(query_set))
                for tokens in self.tokenized_corpus
            ]
        scored = []
        for document, score in zip(self.documents, scores):
            scored.append({**document, "score": float(score), "retrieval": "keyword"})
        return sorted(scored, key=lambda item: item["score"], reverse=True)[:top_k]

    def hybrid_search(self, query: str, top_k: int = 5, alpha: float = 0.5) -> list[dict[str, Any]]:
        """Combine normalized semantic and keyword scores using alpha weighting and return ranked results."""
        semantic_results = self.semantic_search(query, top_k=max(top_k, len(self.documents)))
        keyword_results = self.keyword_search(query, top_k=max(top_k, len(self.documents)))
        semantic_scores = self._score_map(semantic_results)
        keyword_scores = self._score_map(keyword_results)
        all_ids = set(semantic_scores) | set(keyword_scores)
        ranked = []
        documents_by_id = {document["metadata"]["doc_id"]: document for document in self.documents}
        for doc_id in all_ids:
            document = documents_by_id[doc_id]
            resource = document["metadata"].get("resource", "")
            route_bonus = self._resource_route_score(query, resource)
            score = alpha * semantic_scores.get(doc_id, 0.0) + (1 - alpha) * keyword_scores.get(doc_id, 0.0) + route_bonus
            ranked.append({**documents_by_id[doc_id], "score": score, "retrieval": "hybrid"})
        return sorted(ranked, key=lambda item: item["score"], reverse=True)[:top_k]

    def filter_by_rbac(self, results: list[dict[str, Any]], accessible_resources: list[str]) -> list[dict[str, Any]]:
        """Remove results whose metadata resource is not in accessible_resources."""
        accessible = set(accessible_resources)
        return [result for result in results if result["metadata"].get("resource") in accessible]

    def retrieve(self, query: str, user_id: str, rbac_manager: Any, top_k: int = 5) -> list[dict[str, Any]]:
        """Run hybrid search, filter results by RBAC, audit attempts, and return authorized chunks."""
        accessible_resources = rbac_manager.get_accessible_resources(user_id)
        raw_results = self.hybrid_search(query, top_k=top_k * 4)
        filtered = self.filter_by_rbac(raw_results, accessible_resources)
        self.last_top_result_blocked = bool(raw_results) and raw_results[0]["metadata"].get("resource") not in accessible_resources
        blocked_resources = []
        for result in raw_results:
            resource = result["metadata"].get("resource", "")
            granted = resource in accessible_resources
            rbac_manager.log_access_attempt(user_id, resource, granted)
            if not granted:
                blocked_resources.append(resource)
        self.last_blocked_resources = sorted(set(blocked_resources))
        self.last_unauthorized_sources_blocked = len(blocked_resources)
        return filtered[:top_k]

    def _load_model(self) -> Any:
        if SentenceTransformer is None:
            return None
        try:
            return SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            return None

    def _embed(self, text: str) -> list[float]:
        if self.model is not None:
            try:
                return [float(value) for value in self.model.encode(text).tolist()]
            except Exception:
                pass
        vector = [0.0] * 128
        for token in self._tokenize(text):
            digest = hashlib.md5(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:2], "big") % len(vector)
            sign = 1.0 if digest[2] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-zA-Z0-9_]+", text.lower())

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        if not left or not right:
            return 0.0
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(a * a for a in left)) or 1.0
        right_norm = math.sqrt(sum(b * b for b in right)) or 1.0
        return max(0.0, dot / (left_norm * right_norm))

    @staticmethod
    def _score_map(results: list[dict[str, Any]]) -> dict[str, float]:
        if not results:
            return {}
        max_score = max(result["score"] for result in results) or 1.0
        return {result["metadata"]["doc_id"]: result["score"] / max_score for result in results}

    @classmethod
    def _resource_route_score(cls, query: str, resource: str) -> float:
        normalized_query = " ".join(cls._tokenize(query))
        aliases = cls.RESOURCE_ALIASES.get(resource, set())
        matches = sum(1 for alias in aliases if alias in normalized_query)
        return min(1.0, matches * 0.45)
