# Enterprise Agentic RAG System

This project is a role-aware Agentic Retrieval-Augmented Generation system built for the Enterprise RAG Intelligence Challenge. It searches internal documents, CSV records, and JSON logs while checking whether the requesting user is allowed to access each source.

The answer generation is fully extractive. It does not use the OpenAI API or any other hosted text-generation model.

## What the project demonstrates

- Synthetic enterprise data across text, CSV, and JSON formats
- Semantic search with `all-MiniLM-L6-v2`
- Vector storage and similarity search with ChromaDB
- Keyword search with BM25
- Hybrid ranking that combines semantic and keyword results
- Autonomous query planning, evidence evaluation, and retrieval retry workflows
- Context engineering that deduplicates, diversifies, ranks, and bounds evidence
- Role-based access control before answer generation
- Source citations, confidence scores, and hallucination-risk checks
- Audit logging for granted and denied access attempts

## Project structure

```text
enterprise_rag/
|-- data/
|   |-- database/       # Employee and project CSV files
|   |-- documents/      # HR, finance, IT, product, and compliance documents
|   |-- logs/           # System and audit logs
|   |-- metadata/       # RBAC access policies
|   `-- user_roles/     # User-to-role mappings
|-- rag/
|   |-- agent.py        # Autonomous retrieval and context workflow
|   |-- generator.py    # Extractive answer generation
|   |-- ingestion.py    # Multi-format loading and chunking
|   |-- pipeline.py     # End-to-end agentic pipeline
|   |-- rbac.py         # Access-control checks and audit logging
|   `-- retrieval.py    # Semantic, BM25, vector, and hybrid retrieval
|-- tests/
|   `-- test_agent.py
|-- web/
|   `-- index.html      # Local browser interface
|-- main.py
|-- web_app.py          # Dependency-free web server and query API
|-- requirements.txt
`-- synthetic_data_generator.py
```

## Agentic RAG workflow

For each question, the agent:

1. Decomposes compound questions into bounded retrieval tasks.
2. Performs RBAC-aware hybrid searches over the vector and keyword indexes.
3. Evaluates the quality of authorized evidence.
4. Autonomously rewrites and retries weak searches.
5. Engineers compact context by deduplicating chunks, prioritizing high scores, diversifying sources, and enforcing a context budget.
6. Generates an extractive answer with citations and hallucination-risk checks.

The pipeline returns an `agentic_workflow` object containing subqueries, the retry decision, evidence quality, and a concise action trace. The trace reports workflow decisions rather than private chain-of-thought. Answers remain grounded in authorized context to reduce hallucinations.

## Setup

Python 3.10 or newer is recommended.

```powershell
cd enterprise_rag
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The first run may download the `all-MiniLM-L6-v2` sentence-transformer model from Hugging Face. The warning about unauthenticated Hugging Face requests is harmless for this demo.

## Run the demo

```powershell
python main.py
```

The program regenerates the synthetic dataset, builds the hybrid vector index, and runs five example queries. Each result displays the user role, grounded answer, citations, confidence score, evidence quality, agent task count, retry decision, and number of sources blocked by RBAC.

## Run the frontend

```powershell
python web_app.py
```

Wait until the terminal prints that the frontend is ready, then open `http://127.0.0.1:8000` in your browser. The interface lets you select an enterprise user, ask custom questions, inspect citations and metrics, and see the agent workflow and RBAC decisions.

Stop the server with `Ctrl+C`.

## Run tests

```powershell
python -m unittest discover -s tests -v
```

## How access control works

`data/metadata/access_policies.json` maps each resource to its allowed roles. `data/user_roles/users.json` maps user IDs to roles.

Every agent retrieval task is filtered before evidence reaches the context or answer generator. Access decisions are appended to `data/logs/rbac_audit.json` while the program runs.

For example, user `U008` has the `employee` role and is not allowed to access `finance_report`. A finance-report query from this user returns an access-denied response instead of exposing restricted content.

## Notes

- No API key is required.
- Results can vary slightly depending on the installed sentence-transformer version.
- `venv`, `__pycache__`, downloaded models, and generated RBAC audit history should not be included in the submission archive.
