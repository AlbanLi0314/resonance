#!/usr/bin/env python3
"""
Add BEE department researchers to the existing database.
Merges BEE faculty + Zeyu Li into the existing pkl, computes embeddings.
"""
import sys
import json
import pickle
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embedding import Specter2Encoder, format_researcher_text

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
EXISTING_PKL = PROJECT_ROOT / "data" / "processed" / "researchers_dual_adapter.pkl"
OUTPUT_PKL = PROJECT_ROOT / "data" / "processed" / "researchers_dual_adapter.pkl"
BEE_BATCH1 = PROJECT_ROOT / "data" / "raw" / "bee_faculty.json"
BEE_BATCH2 = PROJECT_ROOT / "data" / "raw" / "bee_faculty_batch2.json"


def make_id(name: str) -> str:
    slug = name.lower().replace(" ", "_").replace(".", "")
    short_hash = hashlib.md5(name.encode()).hexdigest()[:8]
    return f"faculty_{slug}_{short_hash}"


def build_raw_text(r: dict) -> str:
    """Build raw_text similar to existing entries."""
    parts = [f"{r['name']}, {r['position']}, {r['department']}."]
    if r.get("biography"):
        parts.append(r["biography"])
    if r.get("research_interests"):
        parts.append(f"Research Interests: {r['research_interests']}")
    return " ".join(parts)


def main():
    # 1. Load existing data
    print(f"Loading existing data from {EXISTING_PKL}...")
    with open(EXISTING_PKL, "rb") as f:
        existing = pickle.load(f)
    print(f"  Existing researchers: {len(existing)}")

    existing_names = {r["name"].lower() for r in existing}

    # 2. Load BEE faculty
    new_researchers = []

    for batch_file in [BEE_BATCH1, BEE_BATCH2]:
        print(f"Loading {batch_file.name}...")
        with open(batch_file) as f:
            batch = json.load(f)
        for r in batch:
            if r["name"].lower() not in existing_names:
                new_researchers.append(r)
            else:
                print(f"  Skipping duplicate: {r['name']}")

    # 3. Add Zeyu Li manually
    zeyu_li = {
        "name": "Zeyu (Alban) Li",
        "position": "Ph.D. Candidate",
        "department": "Biological and Environmental Engineering",
        "email": "zl788@cornell.edu",
        "lab": "DNA Materials Lab (Prof. Dan Luo)",
        "personal_website": "https://zeyuli.net",
        "research_interests": "DNA biotechnology, nucleic acid science, polymer engineering, DNA-based functional materials, scalable manufacturing, DNA-barcoded tracers, nucleic acid purification, self-healing hydrogel composites, 3D printing of biomaterials.",
        "biography": "Zeyu (Alban) Li is a Ph.D. candidate in Biological and Environmental Engineering at Cornell University, working in the DNA Materials Lab led by Prof. Dan Luo. His research focuses on DNA technology and polymer engineering with applications in functional materials and scalable manufacturing. He led an 11 km² field deployment of DNA-barcoded tracers for a DoD-funded project, achieving detection 7 km downstream. He designed a nucleic acid purification platform reducing costs by ~91% and developed a 3D-printable, self-healing DNA-Al³⁺ hydrogel composite. He has co-authored 3 publications and co-invented 2 patents. He holds a B.Sc. in Chemistry (Hons) with Computer Science Minor from Hong Kong Baptist University. Technical skills include DNA technologies, qPCR, bio-cleanroom protocols, rheometry, material characterization, 3D printing, photolithography, nanopatterning, Python, Java, and AI integration."
    }

    if zeyu_li["name"].lower() not in existing_names:
        new_researchers.append(zeyu_li)
    else:
        print(f"  Skipping duplicate: {zeyu_li['name']}")

    print(f"\nNew researchers to add: {len(new_researchers)}")
    for r in new_researchers:
        print(f"  - {r['name']} ({r['position']})")

    if not new_researchers:
        print("Nothing to add!")
        return

    # 4. Build researcher dicts
    new_entries = []
    for r in new_researchers:
        raw_text = build_raw_text(r)
        entry = {
            "id": make_id(r["name"]),
            "name": r["name"],
            "email": r.get("email", ""),
            "raw_text": raw_text,
            "position": r.get("position", ""),
            "department": r.get("department", "Biological and Environmental Engineering"),
            "lab": r.get("lab", ""),
            "personal_website": r.get("personal_website", ""),
            "research_interests": r.get("research_interests", ""),
            "sources": ["bee_faculty_scrape"],
            "biography": r.get("biography", ""),
        }
        new_entries.append(entry)

    # 5. Compute embeddings
    print("\nLoading SPECTER2 encoder...")
    encoder = Specter2Encoder()
    encoder.load()

    print(f"Computing embeddings for {len(new_entries)} researchers...")
    texts = [format_researcher_text(r, format_type="optimized") for r in new_entries]
    embeddings = encoder.encode_batch(texts, batch_size=8, show_progress=True)

    for i, entry in enumerate(new_entries):
        entry["embedding"] = embeddings[i]
        entry["_formatted_text"] = texts[i]
        entry["_original_index"] = len(existing) + i
        entry["original_raw_text"] = entry["raw_text"]
        entry["embedding_adapter"] = "allenai/specter2"

    # 6. Merge and save
    merged = existing + new_entries
    print(f"\nTotal researchers: {len(merged)}")

    # Backup existing
    backup_path = EXISTING_PKL.with_suffix(".pkl.bak")
    print(f"Backing up to {backup_path}...")
    with open(backup_path, "wb") as f:
        pickle.dump(existing, f)

    # Save merged
    print(f"Saving to {OUTPUT_PKL}...")
    with open(OUTPUT_PKL, "wb") as f:
        pickle.dump(merged, f)

    # Print department stats
    depts = {}
    for r in merged:
        d = r.get("department", "unknown")
        depts[d] = depts.get(d, 0) + 1
    print("\nDepartment distribution:")
    for d, c in sorted(depts.items(), key=lambda x: -x[1]):
        print(f"  {c:3d}  {d}")

    print("\nDone! Restart v3.py to use updated data.")


if __name__ == "__main__":
    main()
