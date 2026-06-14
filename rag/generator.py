"""Extractive answer generation for enterprise RAG."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None


class AnswerGenerator:
    """Generate grounded extractive answers from retrieved chunks."""

    def __init__(self) -> None:
        """Load the all-MiniLM-L6-v2 sentence-transformers model when available."""
        self.model = self._load_model()

    def generate_answer(self, query: str, retrieved_chunks: list[dict[str, Any]]) -> str:
        """Extract top relevant sentences from chunks using cosine similarity and combine them into an answer."""
        if not retrieved_chunks:
            return "No authorized relevant context was available for this question."
        query_embedding = self._embed(query)
        query_words = set(self._tokenize(query))
        scored_sentences = []
        seen_sentences = set()
        for chunk in retrieved_chunks:
            for sentence in self._split_sentences(chunk["content"]):
                normalized_sentence = " ".join(self._tokenize(sentence))
                if len(normalized_sentence.split()) < 5 or normalized_sentence in seen_sentences:
                    continue
                seen_sentences.add(normalized_sentence)
                sentence_words = set(self._tokenize(sentence))
                keyword_overlap = len(query_words.intersection(sentence_words)) / max(1, len(query_words))
                similarity = self._cosine(query_embedding, self._embed(sentence))
                scored_sentences.append((0.7 * similarity + 0.3 * keyword_overlap, sentence))
        top_sentences = [sentence for _, sentence in sorted(scored_sentences, reverse=True)[:4]]
        return " ".join(top_sentences) if top_sentences else "No concise answer could be extracted from the authorized context."

    def format_with_citations(self, answer: str, sources: list[dict[str, Any]]) -> str:
        """Append source and chunk citations to an answer string."""
        citation_text = " ".join(f"[Source: {source['source']}, Chunk: {source['chunk']}]" for source in sources)
        return f"{answer} {citation_text}".strip()

    def calculate_confidence(self, query: str, chunks: list[dict[str, Any]]) -> float:
        """Return a 0-1 confidence score from average query-to-chunk cosine similarity."""
        if not chunks:
            return 0.0
        query_embedding = self._embed(query)
        similarities = [self._cosine(query_embedding, self._embed(chunk["content"])) for chunk in chunks]
        return round(max(0.0, min(1.0, sum(similarities) / len(similarities))), 3)

    def check_hallucination_risk(self, answer: str, chunks: list[dict[str, Any]]) -> str:
        """Return LOW when answer words overlap well with chunks, otherwise HIGH."""
        if not answer or not chunks:
            return "HIGH"
        answer_words = set(self._tokenize(answer))
        chunk_words = set(self._tokenize(" ".join(chunk["content"] for chunk in chunks)))
        if not answer_words:
            return "HIGH"
        overlap = len(answer_words.intersection(chunk_words)) / len(answer_words)
        return "LOW" if overlap >= 0.65 else "HIGH"

    def generate(self, query: str, chunks: list[dict[str, Any]], user_id: str, user_role: str) -> dict[str, Any]:
        """Run extractive generation and return answer, citations, confidence, and hallucination risk."""
        answer = self.generate_answer(query, chunks)
        sources = self._unique_sources(chunks)
        return {
            "answer": self.format_with_citations(answer, sources) if sources else answer,
            "citations": sources,
            "confidence": self.calculate_confidence(query, chunks),
            "hallucination_risk": self.check_hallucination_risk(answer, chunks),
            "user_id": user_id,
            "user_role": user_role,
        }

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
    def _split_sentences(text: str) -> list[str]:
        return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+|\s+\|\s+", text) if sentence.strip()]

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
    def _unique_sources(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen = set()
        sources = []
        for chunk in chunks:
            metadata = chunk["metadata"]
            key = (metadata.get("source"), metadata.get("chunk"))
            if key not in seen:
                seen.add(key)
                sources.append({"source": metadata.get("source"), "chunk": metadata.get("chunk")})
        return sources
