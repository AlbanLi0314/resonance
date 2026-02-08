"""
Data Validation for Academic Matcher
Ensures crawler data matches expected schema before processing
"""
from typing import List, Dict, Tuple, Optional
import json


# ============================================================
# Expected Data Schema
# ============================================================
# This is the contract with the crawler team.
# If crawler output changes, update this schema.

REQUIRED_FIELDS = {
    "id": str,          # Unique identifier (e.g., "cornell_mse_001")
    "name": str,        # Full name (e.g., "John Smith")
    "email": str,       # Email address
    "raw_text": str,    # Combined text for embedding (research description, bio, etc.)
}

OPTIONAL_FIELDS = {
    "position": str,           # e.g., "PhD Student", "Professor"
    "department": str,         # e.g., "Materials Science and Engineering"
    "lab": str,                # Lab name
    "advisor": str,            # Advisor name (for students/postdocs)
    "personal_website": str,   # URL
    "google_scholar": str,     # URL
    "research_interests": str, # Comma-separated keywords
    "papers": list,            # List of paper dicts
    "sources": list,           # Where data was scraped from
}

# Alternative field names that crawler might use
# Maps alternative name -> canonical name
FIELD_ALIASES = {
    "bio": "raw_text",
    "description": "raw_text",
    "profile": "raw_text",
    "about": "raw_text",
    "full_name": "name",
    "email_address": "email",
    "mail": "email",
    "researcher_id": "id",
    "uid": "id",
}


class DataValidationError(Exception):
    """Raised when data doesn't match expected schema"""
    pass


def normalize_field_names(researcher: dict) -> dict:
    """
    Convert alternative field names to canonical names.

    Args:
        researcher: Raw researcher dict from crawler

    Returns:
        Researcher dict with normalized field names
    """
    normalized = {}

    for key, value in researcher.items():
        # Check if this is an alias
        canonical_key = FIELD_ALIASES.get(key, key)

        # If we already have the canonical field, don't overwrite
        if canonical_key in normalized and normalized[canonical_key]:
            continue

        normalized[canonical_key] = value

    return normalized


def validate_researcher(researcher: dict, index: int) -> Tuple[bool, List[str]]:
    """
    Validate a single researcher against schema.

    Args:
        researcher: Researcher dict
        index: Index in the list (for error reporting)

    Returns:
        (is_valid, list_of_errors)
    """
    errors = []

    # Check required fields
    for field, expected_type in REQUIRED_FIELDS.items():
        if field not in researcher:
            errors.append(f"Missing required field: {field}")
        elif researcher[field] is None:
            errors.append(f"Field '{field}' is None")
        elif not isinstance(researcher[field], expected_type):
            errors.append(f"Field '{field}' should be {expected_type.__name__}, got {type(researcher[field]).__name__}")
        elif expected_type == str and not researcher[field].strip():
            errors.append(f"Field '{field}' is empty")

    return len(errors) == 0, errors


def validate_dataset(researchers: List[dict]) -> Tuple[List[dict], Dict]:
    """
    Validate entire dataset and return valid entries.

    Args:
        researchers: List of researcher dicts from crawler

    Returns:
        (valid_researchers, validation_report)
    """
    valid = []
    report = {
        "total_input": len(researchers),
        "valid_count": 0,
        "invalid_count": 0,
        "errors_by_type": {},
        "invalid_entries": [],  # List of (index, name, errors)
    }

    seen_ids = set()
    seen_emails = set()
    duplicates = {"by_id": 0, "by_email": 0}

    for i, raw_researcher in enumerate(researchers):
        # Normalize field names
        researcher = normalize_field_names(raw_researcher)

        # Validate
        is_valid, errors = validate_researcher(researcher, i)

        if not is_valid:
            report["invalid_count"] += 1
            report["invalid_entries"].append({
                "index": i,
                "name": researcher.get("name", "Unknown"),
                "errors": errors
            })
            # Count error types
            for err in errors:
                err_type = err.split(":")[0] if ":" in err else err
                report["errors_by_type"][err_type] = report["errors_by_type"].get(err_type, 0) + 1
            continue

        # Check for duplicates
        rid = researcher.get("id", "")
        email = researcher.get("email", "").lower()

        if rid and rid in seen_ids:
            duplicates["by_id"] += 1
            continue
        if email and email in seen_emails:
            duplicates["by_email"] += 1
            continue

        seen_ids.add(rid)
        seen_emails.add(email)
        valid.append(researcher)
        report["valid_count"] += 1

    report["duplicates"] = duplicates

    return valid, report


def print_validation_report(report: Dict):
    """Print a human-readable validation report"""
    print("\n" + "=" * 50)
    print("DATA VALIDATION REPORT")
    print("=" * 50)
    print(f"Total input:     {report['total_input']}")
    print(f"Valid:           {report['valid_count']}")
    print(f"Invalid:         {report['invalid_count']}")

    dups = report.get('duplicates', {})
    if dups.get('by_id', 0) > 0 or dups.get('by_email', 0) > 0:
        print(f"Duplicates:      {dups.get('by_id', 0)} by ID, {dups.get('by_email', 0)} by email")

    if report['errors_by_type']:
        print("\nErrors by type:")
        for err_type, count in sorted(report['errors_by_type'].items(), key=lambda x: -x[1]):
            print(f"  - {err_type}: {count}")

    if report['invalid_entries'] and len(report['invalid_entries']) <= 10:
        print("\nInvalid entries:")
        for entry in report['invalid_entries']:
            print(f"  [{entry['index']}] {entry['name']}: {entry['errors']}")
    elif report['invalid_entries']:
        print(f"\nFirst 10 invalid entries (of {len(report['invalid_entries'])}):")
        for entry in report['invalid_entries'][:10]:
            print(f"  [{entry['index']}] {entry['name']}: {entry['errors']}")

    print("=" * 50)


def save_validation_report(report: Dict, path: str):
    """Save validation report to JSON file"""
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Validation report saved to: {path}")


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    # Test with sample data
    test_data = [
        {
            "id": "test_001",
            "name": "John Smith",
            "email": "john@example.com",
            "raw_text": "Research in DNA origami and nanomaterials."
        },
        {
            "id": "test_002",
            "name": "",  # Invalid: empty name
            "email": "jane@example.com",
            "raw_text": "Research in catalysis."
        },
        {
            "bio": "Alternative field name test.",  # Uses alias
            "full_name": "Bob Wilson",
            "mail": "bob@example.com",
            "uid": "test_003"
        },
        {
            "id": "test_001",  # Duplicate ID
            "name": "John Smith Clone",
            "email": "john2@example.com",
            "raw_text": "Duplicate researcher."
        },
    ]

    valid, report = validate_dataset(test_data)
    print_validation_report(report)
    print(f"\nValid researchers: {[r['name'] for r in valid]}")
