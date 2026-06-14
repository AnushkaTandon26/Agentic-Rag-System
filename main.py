"""Demo runner for the Enterprise RAG Intelligence Challenge."""

from __future__ import annotations

from pathlib import Path

try:
    from . import synthetic_data_generator
    from .rag.pipeline import EnterpriseRAGPipeline
except ImportError:
    import synthetic_data_generator
    from rag.pipeline import EnterpriseRAGPipeline


def _format_citations(citations: list[dict[str, object]]) -> str:
    if not citations:
        return "None"
    return ", ".join(f"{citation['source']} chunk {citation['chunk']}" for citation in citations)


def main() -> None:
    synthetic_data_generator.generate_synthetic_data()

    base_dir = Path(__file__).resolve().parent
    pipeline = EnterpriseRAGPipeline(base_dir)
    pipeline.setup()

    queries = [
        ("U001", "What is the leave policy for employees?"),
        ("U002", "What was Q1 revenue and budget allocation?"),
        ("U003", "What are the VPN access security protocols?"),
        ("U008", "Show me the financial report details"),
        ("U005", "Give me a summary of all system audit logs"),
    ]

    for user_id, question in queries:
        result = pipeline.query(user_id, question)
        print("=" * 88)
        print(f"User: {result['user_id']}")
        print(f"Role: {result['user_role']}")
        print(f"Question: {result['query']}")
        print(f"Answer: {result['answer']}")
        print(f"Citations: {_format_citations(result['citations'])}")
        print(f"Confidence Score: {result['confidence']:.3f}")
        print(f"Evidence Quality: {result['agentic_workflow']['evidence_quality']:.3f}")
        print(f"Agent Retrieval Tasks: {len(result['agentic_workflow']['subqueries'])}")
        print(f"Agent Retried Retrieval: {result['agentic_workflow']['retrieval_retried']}")
        print(f"RBAC Blocked Sources Count: {result['unauthorized_sources_blocked']}")
    print("=" * 88)


if __name__ == "__main__":
    main()
