"""Role-based access control utilities for enterprise RAG."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


class RBACManager:
    """Manage user roles, resource policies, and RBAC audit logging."""

    def __init__(self, base_dir: str | Path | None = None) -> None:
        """Initialize the manager with the enterprise project directory."""
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).resolve().parents[1]
        self.policies = self.load_policies()
        self.users = self.load_users()

    def load_policies(self) -> dict[str, list[str]]:
        """Read RBAC resource policies from data/metadata/access_policies.json."""
        policy_path = self.base_dir / "data" / "metadata" / "access_policies.json"
        if not policy_path.exists():
            return {}
        return json.loads(policy_path.read_text(encoding="utf-8"))

    def load_users(self) -> dict[str, dict[str, str]]:
        """Read user-role mappings from data/user_roles/users.json."""
        users_path = self.base_dir / "data" / "user_roles" / "users.json"
        if not users_path.exists():
            return {}
        users = json.loads(users_path.read_text(encoding="utf-8"))
        return {user["user_id"]: user for user in users}

    def get_user_role(self, user_id: str) -> str:
        """Return the role string for a user ID, or unknown when absent."""
        return self.users.get(user_id, {}).get("role", "unknown")

    def can_access(self, user_id: str, resource: str) -> bool:
        """Return True when the user's role is allowed to access a resource."""
        role = self.get_user_role(user_id)
        resource_key = self._normalize_resource(resource)
        allowed_roles = self.policies.get(resource_key, [])
        return role in allowed_roles or "all_employees" in allowed_roles

    def get_accessible_resources(self, user_id: str) -> list[str]:
        """Return all resource names accessible to the given user."""
        return [resource for resource in self.policies if self.can_access(user_id, resource)]

    def log_access_attempt(self, user_id: str, resource: str, granted: bool) -> None:
        """Append a resource access decision to data/logs/rbac_audit.json."""
        audit_path = self.base_dir / "data" / "logs" / "rbac_audit.json"
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        entries = []
        if audit_path.exists():
            try:
                entries = json.loads(audit_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                entries = []
        entries.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "user_id": user_id,
                "role": self.get_user_role(user_id),
                "resource": self._normalize_resource(resource),
                "granted": granted,
            }
        )
        try:
            audit_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
        except OSError:
            # Audit persistence must not make an otherwise authorized query unavailable.
            return

    @staticmethod
    def _normalize_resource(resource: str) -> str:
        path = Path(resource)
        return path.stem if path.suffix else resource
