#!/usr/bin/env python3
"""
Transform production data from crawler to match the Academic Matcher schema.

Input files:
- phd_with_publications.json (PhD students)
- all_faculty.json (Faculty members)

Output:
- data/raw/researchers_production.json (combined, validated dataset)
"""
import json
import sys
import hashlib
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_validator import validate_dataset, print_validation_report, save_validation_report
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR


def generate_id(name: str, email: str, prefix: str = "cornell") -> str:
    """Generate a unique ID from name and email."""
    # Create a hash-based ID for uniqueness
    base = f"{name.lower()}_{email.lower()}"
    hash_suffix = hashlib.md5(base.encode()).hexdigest()[:8]
    # Clean name for ID
    clean_name = name.lower().replace(" ", "_").replace(".", "")[:20]
    return f"{prefix}_{clean_name}_{hash_suffix}"


def transform_faculty(faculty_data: list) -> list:
    """Transform faculty data to match schema."""
    transformed = []

    for f in faculty_data:
        # Skip if no email
        email = f.get("email", "").strip()
        if not email:
            continue

        name = f.get("name", "").strip()
        if not name:
            continue

        # Build raw_text from available fields
        raw_text = f.get("raw_text", "")
        if not raw_text:
            # Fallback: combine biography and research_interests
            parts = []
            if f.get("biography"):
                parts.append(f["biography"])
            if f.get("research_interests"):
                parts.append(f["research_interests"])
            raw_text = "\n".join(parts)

        if not raw_text.strip():
            continue

        researcher = {
            "id": generate_id(name, email, "faculty"),
            "name": name,
            "email": email,
            "raw_text": raw_text,
            "position": f.get("position", ""),
            "department": f.get("department", ""),
            "lab": f.get("lab_name", ""),
            "personal_website": f.get("profile_url", ""),
            "research_interests": f.get("research_interests", ""),
            "sources": ["all_faculty.json"],
        }

        # Add optional fields if present
        if f.get("lab_url"):
            researcher["lab_url"] = f["lab_url"]
        if f.get("biography"):
            researcher["biography"] = f["biography"]

        transformed.append(researcher)

    return transformed


def transform_phd(phd_data: list) -> list:
    """Transform PhD student data to match schema."""
    transformed = []

    for p in phd_data:
        # Skip if no email
        email = p.get("email", "").strip()
        if not email:
            continue

        name = p.get("name", "").strip()
        if not name:
            continue

        # Build raw_text from available fields
        parts = []

        # Add department and lab context
        if p.get("department"):
            parts.append(f"PhD student in {p['department']}.")
        if p.get("lab_name"):
            parts.append(f"Member of {p['lab_name']}.")
        if p.get("advisor_full_name"):
            parts.append(f"Advised by {p['advisor_full_name']}.")

        # Add publications - these are gold for understanding research focus
        publications = p.get("publications", [])
        if publications:
            pub_titles = [pub.get("title", "") for pub in publications if pub.get("title")]
            if pub_titles:
                parts.append("Publications: " + "; ".join(pub_titles[:10]))  # Limit to 10

        raw_text = " ".join(parts)

        if not raw_text.strip():
            continue

        researcher = {
            "id": generate_id(name, email, "phd"),
            "name": name,
            "email": email,
            "raw_text": raw_text,
            "position": "PhD Student",
            "department": p.get("department", ""),
            "lab": p.get("lab_name", ""),
            "advisor": p.get("advisor_full_name", p.get("advisor", "")),
            "sources": ["phd_with_publications.json"],
        }

        # Add publications as structured data
        if publications:
            researcher["papers"] = [
                {
                    "title": pub.get("title", ""),
                    "year": pub.get("year", ""),
                    "doi": pub.get("doi"),
                }
                for pub in publications
                if pub.get("title")
            ]

        # Add lab URL if present
        if p.get("lab_url"):
            researcher["personal_website"] = p["lab_url"]

        transformed.append(researcher)

    return transformed


def main():
    print("=" * 60)
    print("PRODUCTION DATA TRANSFORMATION")
    print("=" * 60)

    # Load source files
    base_dir = Path(__file__).parent.parent

    phd_path = base_dir / "phd_with_publications.json"
    faculty_path = base_dir / "all_faculty.json"

    print(f"\nLoading source files...")

    with open(phd_path) as f:
        phd_data = json.load(f)
    print(f"  PhD students: {len(phd_data)}")

    with open(faculty_path) as f:
        faculty_data = json.load(f)
    print(f"  Faculty: {len(faculty_data)}")

    # Transform data
    print(f"\nTransforming data...")

    faculty_transformed = transform_faculty(faculty_data)
    print(f"  Faculty transformed: {len(faculty_transformed)}")

    phd_transformed = transform_phd(phd_data)
    print(f"  PhD students transformed: {len(phd_transformed)}")

    # Combine
    combined = faculty_transformed + phd_transformed
    print(f"\n  Total combined: {len(combined)}")

    # Validate
    print(f"\nValidating combined dataset...")
    valid_researchers, report = validate_dataset(combined)
    print_validation_report(report)

    # Save output
    output_path = RAW_DATA_DIR / "researchers_production.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(valid_researchers, f, indent=2)
    print(f"\nSaved {len(valid_researchers)} researchers to: {output_path}")

    # Save validation report
    report_path = PROCESSED_DATA_DIR / "production_validation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    save_validation_report(report, str(report_path))

    # Print summary statistics
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    # Count by type
    faculty_count = len([r for r in valid_researchers if r["id"].startswith("faculty_")])
    phd_count = len([r for r in valid_researchers if r["id"].startswith("phd_")])

    print(f"Faculty: {faculty_count}")
    print(f"PhD Students: {phd_count}")
    print(f"Total: {len(valid_researchers)}")

    # Count by department
    depts = {}
    for r in valid_researchers:
        dept = r.get("department", "Unknown")
        depts[dept] = depts.get(dept, 0) + 1

    print(f"\nBy Department:")
    for dept, count in sorted(depts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {dept}: {count}")

    return valid_researchers


if __name__ == "__main__":
    main()
