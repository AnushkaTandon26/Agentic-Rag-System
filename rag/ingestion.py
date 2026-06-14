"""Data ingestion and chunking for enterprise RAG sources."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except Exception:  # pragma: no cover - fallback keeps demo runnable without optional deps.
    pd = None


class DataIngestionPipeline:
    """Load documents, structured data, logs, and convert them into chunks."""

    DEPARTMENTS = {
        "hr_policy": "HR",
        "finance_report": "Finance",
        "it_security_policy": "IT",
        "product_roadmap": "Product",
        "compliance_manual": "Legal",
        "employees": "HR",
        "projects": "Project Office",
        "system_logs": "IT",
        "audit_trail": "Compliance",
    }
    CONFIDENTIALITY = {
        "hr_policy": "internal",
        "finance_report": "restricted",
        "it_security_policy": "restricted",
        "product_roadmap": "confidential",
        "compliance_manual": "restricted",
        "employees": "confidential",
        "projects": "confidential",
        "system_logs": "restricted",
        "audit_trail": "restricted",
    }

    def __init__(self, base_dir: str | Path | None = None) -> None:
        """Initialize the ingestion pipeline with the enterprise project directory."""
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).resolve().parents[1]
        self.data_dir = self.base_dir / "data"

    def load_documents(self) -> list[dict[str, str]]:
        """Read all .txt files from data/documents/ as source records."""
        records = []
        for path in sorted((self.data_dir / "documents").glob("*.txt")):
            records.append({"text": path.read_text(encoding="utf-8"), "source": path.name, "resource_name": path.stem})
        return records

    def load_csv_data(self) -> list[dict[str, str]]:
        """Read employees.csv and projects.csv with pandas when available and convert rows to text."""
        records = []
        for filename in ["employees.csv", "projects.csv"]:
            path = self.data_dir / "database" / filename
            if not path.exists():
                continue
            resource_name = path.stem
            if pd is not None:
                dataframe = pd.read_csv(path)
                row_text = dataframe.astype(str).apply(lambda row: ", ".join(f"{col}: {row[col]}" for col in dataframe.columns), axis=1)
                text = f"{resource_name} database records. " + " | ".join(row_text.tolist())
            else:
                with path.open(newline="", encoding="utf-8") as file:
                    reader = csv.DictReader(file)
                    text = f"{resource_name} database records. " + " | ".join(
                        ", ".join(f"{key}: {value}" for key, value in row.items()) for row in reader
                    )
            records.append({"text": text, "source": filename, "resource_name": resource_name})
        return records

    def load_json_logs(self) -> list[dict[str, str]]:
        """Read system_logs.json and audit_trail.json and convert entries to text."""
        records = []
        for filename in ["system_logs.json", "audit_trail.json"]:
            path = self.data_dir / "logs" / filename
            if not path.exists():
                continue
            resource_name = path.stem
            entries = json.loads(path.read_text(encoding="utf-8"))
            text_entries = []
            for entry in entries:
                text_entries.append(", ".join(f"{key}: {value}" for key, value in entry.items()))
            records.append({"text": f"{resource_name} log entries. " + " | ".join(text_entries), "source": filename, "resource_name": resource_name})
        return records

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
        """Split text into overlapping character chunks."""
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be non-negative and smaller than chunk_size")
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end].strip())
            if end == len(text):
                break
            start = end - overlap
        return [chunk for chunk in chunks if chunk]

    def attach_metadata(self, chunk: str, source: str, resource_name: str) -> dict[str, Any]:
        """Attach source, resource, department, and confidentiality metadata to a chunk."""
        return {
            "content": chunk,
            "metadata": {
                "source": source,
                "resource": resource_name,
                "department": self.DEPARTMENTS.get(resource_name, "General"),
                "confidentiality": self.CONFIDENTIALITY.get(resource_name, "internal"),
            },
        }

    def ingest_all(self) -> list[dict[str, Any]]:
        """Load every supported source, chunk the content, attach metadata, and return document dictionaries."""
        documents = []
        for record in self.load_documents() + self.load_csv_data() + self.load_json_logs():
            chunks = self.chunk_text(record["text"])
            for chunk_index, chunk in enumerate(chunks, start=1):
                document = self.attach_metadata(chunk, record["source"], record["resource_name"])
                document["metadata"]["chunk"] = chunk_index
                document["metadata"]["doc_id"] = f"{record['resource_name']}-{chunk_index}"
                documents.append(document)
        return documents
