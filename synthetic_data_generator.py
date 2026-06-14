"""Generate deterministic synthetic enterprise data for the RAG demo."""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timedelta
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def _write_if_changed(path: Path, content: str) -> None:
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    path.write_text(content, encoding="utf-8")


DOCUMENT_TOPICS = {
    "hr_policy.txt": [
        "The human resources policy explains employee leave, attendance, benefits, workplace conduct, and manager approval workflows.",
        "Full time employees receive annual leave, sick leave, parental leave, bereavement leave, and public holiday benefits according to local law.",
        "Leave requests should be submitted through the employee portal at least ten business days before planned absence whenever possible.",
        "Managers review requests against staffing levels, project commitments, fairness, and documented business continuity requirements.",
        "Employees must record accurate working hours, protect confidential personnel information, and report conflicts of interest promptly.",
        "Remote work is allowed when performance expectations, data security rules, and team collaboration standards continue to be met.",
        "Benefits include health insurance, wellness support, learning reimbursement, retirement contributions, and confidential employee assistance services.",
        "Policy exceptions require written HR approval and may be audited for compliance, equity, and operational impact.",
    ],
    "finance_report.txt": [
        "The quarterly finance report summarizes revenue, operating expenses, budget allocation, cash position, and forecast assumptions for leadership review.",
        "Q1 revenue closed at 18.4 million dollars, exceeding the approved plan by six percent because enterprise renewals improved.",
        "Sales expenses increased moderately after targeted campaigns, while infrastructure costs remained within the cloud optimization budget.",
        "Budget allocation prioritized product development, customer success, compliance readiness, and security monitoring for regulated clients.",
        "Finance recommends maintaining a conservative hiring plan until recurring revenue coverage and collection timing remain stable.",
        "Department managers must justify discretionary spending, document vendor approvals, and reconcile purchase orders before month end.",
        "The report identifies currency exposure, delayed receivables, procurement risk, and renewal concentration as watch items.",
        "Executive dashboards should use audited figures only, because draft spreadsheets may include estimates pending controller approval.",
    ],
    "it_security_policy.txt": [
        "The information technology security policy defines acceptable use, identity management, VPN access, endpoint protection, and incident response.",
        "Employees must use multi factor authentication for corporate applications, administrative consoles, source repositories, and remote access services.",
        "VPN access is restricted to managed devices with current patches, encrypted disks, approved antivirus software, and active device certificates.",
        "Privileged access requires ticket approval, manager validation, least privilege assignment, and review by the security operations team.",
        "Passwords must never be shared, stored in plain text, or reused across personal and enterprise systems.",
        "Security events including phishing attempts, lost devices, suspicious logins, and policy violations must be reported immediately.",
        "System owners are responsible for vulnerability remediation, backup testing, access reviews, and secure configuration baselines.",
        "Exceptions expire automatically unless the chief information security officer grants a documented extension after risk assessment.",
    ],
    "product_roadmap.txt": [
        "The product roadmap outlines planned platform capabilities, customer commitments, engineering dependencies, launch phases, and market assumptions.",
        "The next release focuses on workflow automation, analytics dashboards, mobile approvals, accessibility improvements, and enterprise administration controls.",
        "Product management will validate priorities using customer interviews, usage telemetry, support escalations, and competitive analysis.",
        "Engineering milestones include API versioning, search relevance improvements, billing integration, and observability upgrades across services.",
        "Roadmap dates are planning targets rather than contractual commitments unless approved by legal and customer success leadership.",
        "Confidential features should not be discussed externally before launch messaging, pricing, and security documentation are finalized.",
        "Beta programs require named sponsors, feedback criteria, rollback plans, and support coverage for participating accounts.",
        "Quarterly reviews may adjust scope when compliance deadlines, reliability work, or strategic partnerships become higher priority.",
    ],
    "compliance_manual.txt": [
        "The compliance manual describes governance practices for privacy, records retention, vendor review, audit readiness, and regulatory obligations.",
        "Employees handling regulated information must follow data classification labels, retention schedules, secure transfer rules, and approved storage locations.",
        "Compliance evidence includes policy acknowledgements, access reviews, training completion, incident records, and management signoffs.",
        "Legal and compliance teams coordinate responses to audits, customer questionnaires, regulator inquiries, and contractual control requests.",
        "Sensitive records should be retained only for approved periods and disposed of using documented secure deletion procedures.",
        "Third party vendors require risk assessment, security review, privacy review, contract approval, and periodic monitoring.",
        "Any suspected breach, unauthorized disclosure, or control failure must be escalated through the incident response process immediately.",
        "The manual supports continuous improvement through control testing, corrective action tracking, and executive compliance reporting.",
    ],
}


def _ensure_directories() -> None:
    for path in [
        DATA_DIR / "documents",
        DATA_DIR / "database",
        DATA_DIR / "logs",
        DATA_DIR / "metadata",
        DATA_DIR / "user_roles",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def _two_hundred_words(sentences: list[str]) -> str:
    words: list[str] = []
    index = 0
    while len(words) < 200:
        words.extend(sentences[index % len(sentences)].split())
        index += 1
    return " ".join(words[:200]) + "."


def _write_documents() -> None:
    for filename, sentences in DOCUMENT_TOPICS.items():
        _write_if_changed(DATA_DIR / "documents" / filename, _two_hundred_words(sentences))


def _write_csv_files() -> None:
    employees = [
        ("E001", "Alice Morgan", "HR", "hr", 86000, "Maya Patel", "confidential"),
        ("E002", "Bob Chen", "Finance", "finance", 94000, "Iris Novak", "restricted"),
        ("E003", "Carol Singh", "IT", "it", 99000, "Noah Brooks", "restricted"),
        ("E004", "Dave Romero", "Product", "product", 102000, "Lena Ortiz", "confidential"),
        ("E005", "Eve Johnson", "Administration", "admin", 125000, "Board Office", "top_secret"),
        ("E006", "Frank Wilson", "Legal", "legal", 118000, "Priya Shah", "restricted"),
        ("E007", "Grace Kim", "Engineering", "engineering", 111000, "Omar Diaz", "confidential"),
        ("E008", "Henry Brown", "Operations", "employee", 68000, "Maya Patel", "internal"),
        ("E009", "Isabella Rossi", "Compliance", "compliance", 107000, "Priya Shah", "restricted"),
        ("E010", "Jack Miller", "Security", "security", 104000, "Noah Brooks", "restricted"),
        ("E011", "Karen Davis", "HR", "hr", 79000, "Alice Morgan", "confidential"),
        ("E012", "Leo Garcia", "Finance", "finance", 88000, "Bob Chen", "restricted"),
        ("E013", "Mina Park", "IT", "it", 97000, "Carol Singh", "restricted"),
        ("E014", "Nora Evans", "Product", "project_lead", 109000, "Dave Romero", "confidential"),
        ("E015", "Owen Clark", "Engineering", "engineering", 115000, "Grace Kim", "confidential"),
        ("E016", "Paula Adams", "Legal", "legal", 112000, "Frank Wilson", "restricted"),
        ("E017", "Quinn Reed", "Sales", "employee", 76000, "Maya Patel", "internal"),
        ("E018", "Ravi Kumar", "Security", "security", 101000, "Jack Miller", "restricted"),
        ("E019", "Sara Lopez", "Compliance", "compliance", 98000, "Isabella Rossi", "restricted"),
        ("E020", "Tom Nguyen", "Operations", "employee", 72000, "Henry Brown", "internal"),
    ]
    employee_output = io.StringIO(newline="")
    employee_writer = csv.writer(employee_output, lineterminator="\n")
    employee_writer.writerow(["emp_id", "name", "department", "role", "salary", "manager", "access_level"])
    employee_writer.writerows(employees)
    _write_if_changed(DATA_DIR / "database" / "employees.csv", employee_output.getvalue())

    projects = [
        ("P001", "Apollo Automation", "Product", 750000, "active", "Nora Evans", "confidential"),
        ("P002", "Ledger Modernization", "Finance", 420000, "active", "Bob Chen", "restricted"),
        ("P003", "Zero Trust VPN", "IT", 610000, "active", "Carol Singh", "restricted"),
        ("P004", "People Portal Refresh", "HR", 280000, "planning", "Alice Morgan", "internal"),
        ("P005", "Audit Evidence Hub", "Compliance", 360000, "active", "Isabella Rossi", "restricted"),
        ("P006", "Contract Intelligence", "Legal", 330000, "planning", "Frank Wilson", "confidential"),
        ("P007", "Observability Upgrade", "Engineering", 540000, "active", "Grace Kim", "confidential"),
        ("P008", "Customer Mobile Launch", "Product", 690000, "delayed", "Dave Romero", "confidential"),
        ("P009", "Security Data Lake", "Security", 470000, "active", "Jack Miller", "restricted"),
        ("P010", "Operations Insights", "Operations", 260000, "complete", "Henry Brown", "internal"),
    ]
    project_output = io.StringIO(newline="")
    project_writer = csv.writer(project_output, lineterminator="\n")
    project_writer.writerow(["project_id", "name", "department", "budget", "status", "lead", "confidentiality_level"])
    project_writer.writerows(projects)
    _write_if_changed(DATA_DIR / "database" / "projects.csv", project_output.getvalue())


def _write_logs() -> None:
    base_time = datetime(2026, 1, 15, 9, 0, 0)
    actions = ["login", "search", "download", "update", "view", "export"]
    resources = ["hr_policy", "finance_report", "it_security_policy", "projects", "system_logs", "audit_trail"]
    system_logs = []
    for index in range(30):
        system_logs.append(
            {
                "timestamp": (base_time + timedelta(minutes=index * 17)).isoformat(),
                "user_id": f"U{(index % 8) + 1:03d}",
                "action": actions[index % len(actions)],
                "resource": resources[index % len(resources)],
                "status": "success" if index % 7 else "review_required",
                "ip_address": f"10.24.{index % 6}.{50 + index}",
            }
        )
    _write_if_changed(DATA_DIR / "logs" / "system_logs.json", json.dumps(system_logs, indent=2))

    audit_trail = []
    departments = ["HR", "Finance", "IT", "Product", "Admin", "Legal", "Engineering", "Operations"]
    queries = [
        "leave balance lookup",
        "quarterly revenue summary",
        "vpn exception review",
        "roadmap dependency search",
        "privileged log export",
    ]
    for index in range(20):
        audit_trail.append(
            {
                "timestamp": (base_time + timedelta(hours=index * 3)).isoformat(),
                "user": f"U{(index % 8) + 1:03d}",
                "department": departments[index % len(departments)],
                "query": queries[index % len(queries)],
                "data_accessed": resources[(index + 2) % len(resources)],
                "permission_level": ["internal", "confidential", "restricted", "admin"][index % 4],
            }
        )
    _write_if_changed(DATA_DIR / "logs" / "audit_trail.json", json.dumps(audit_trail, indent=2))


def _write_metadata() -> None:
    policies = {
        "finance_report": ["finance", "admin"],
        "hr_policy": ["hr", "admin", "all_employees"],
        "it_security_policy": ["it", "admin"],
        "product_roadmap": ["product", "admin", "engineering"],
        "compliance_manual": ["legal", "admin", "compliance"],
        "employees": ["hr", "admin"],
        "projects": ["admin", "project_lead"],
        "system_logs": ["it", "admin", "security"],
        "audit_trail": ["admin", "compliance", "legal"],
    }
    _write_if_changed(DATA_DIR / "metadata" / "access_policies.json", json.dumps(policies, indent=2))

    users = [
        {"user_id": "U001", "name": "Alice", "role": "hr"},
        {"user_id": "U002", "name": "Bob", "role": "finance"},
        {"user_id": "U003", "name": "Carol", "role": "it"},
        {"user_id": "U004", "name": "Dave", "role": "product"},
        {"user_id": "U005", "name": "Eve", "role": "admin"},
        {"user_id": "U006", "name": "Frank", "role": "legal"},
        {"user_id": "U007", "name": "Grace", "role": "engineering"},
        {"user_id": "U008", "name": "Henry", "role": "employee"},
    ]
    _write_if_changed(DATA_DIR / "user_roles" / "users.json", json.dumps(users, indent=2))


def generate_synthetic_data() -> None:
    """Create all synthetic documents, CSV databases, logs, metadata, and users."""
    _ensure_directories()
    _write_documents()
    _write_csv_files()
    _write_logs()
    _write_metadata()


if __name__ == "__main__":
    generate_synthetic_data()
    print(f"Synthetic enterprise data generated under {DATA_DIR}")
