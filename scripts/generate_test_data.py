#!/usr/bin/env python3
"""
Generate Expanded Test Dataset
==============================
Creates a more challenging test dataset with:
- More researchers (50+)
- Confounding cases (similar fields, different specialties)
- Edge cases (minimal info, overlapping keywords)

This enables more meaningful embedding optimization experiments.
"""

import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

# Research areas with sub-specialties (for confounding cases)
RESEARCH_AREAS = {
    "DNA_Nanotechnology": {
        "keywords": ["DNA origami", "self-assembly", "nanomaterials", "nucleic acid"],
        "sub_areas": [
            ("DNA origami design", "designing 3D DNA nanostructures using computational tools"),
            ("DNA self-assembly", "thermal annealing and assembly protocols for DNA structures"),
            ("DNA-based drug delivery", "using DNA nanostructures for targeted therapeutics"),
            ("DNA computing", "molecular computing and logic gates using DNA"),
            ("DNA-protein conjugates", "hybrid materials combining DNA with proteins"),
        ]
    },
    "Computational_Materials": {
        "keywords": ["DFT", "machine learning", "computational", "simulation", "MD"],
        "sub_areas": [
            ("DFT for batteries", "density functional theory for battery cathode materials"),
            ("ML materials discovery", "neural networks for predicting material properties"),
            ("Molecular dynamics", "atomistic simulations of materials behavior"),
            ("High-throughput screening", "automated computational screening of materials"),
            ("Multiscale modeling", "bridging atomistic to continuum simulations"),
        ]
    },
    "Polymer_Science": {
        "keywords": ["polymer", "synthesis", "RAFT", "ATRP", "polymerization"],
        "sub_areas": [
            ("Controlled polymerization", "RAFT and ATRP synthesis methods"),
            ("Block copolymers", "self-assembling polymer architectures"),
            ("Sustainable polymers", "bio-based and recyclable polymers"),
            ("Polymer nanocomposites", "polymer-nanoparticle hybrid materials"),
            ("Conductive polymers", "conjugated polymers for electronics"),
        ]
    },
    "Catalysis": {
        "keywords": ["catalysis", "catalyst", "reaction", "conversion", "selectivity"],
        "sub_areas": [
            ("CO2 electrocatalysis", "electrochemical CO2 reduction to fuels"),
            ("Heterogeneous catalysis", "solid catalysts for industrial reactions"),
            ("Photocatalysis", "light-driven catalytic reactions"),
            ("Single-atom catalysis", "atomically dispersed metal catalysts"),
            ("Enzyme mimics", "artificial enzymes and biomimetic catalysis"),
        ]
    },
    "2D_Materials": {
        "keywords": ["graphene", "2D", "MoS2", "CVD", "heterostructure", "monolayer"],
        "sub_areas": [
            ("Graphene CVD", "chemical vapor deposition of graphene"),
            ("TMD synthesis", "transition metal dichalcogenide growth"),
            ("2D heterostructures", "van der Waals heterostructure assembly"),
            ("2D electronics", "transistors and devices from 2D materials"),
            ("2D photonics", "optical properties and devices"),
        ]
    },
    "Biomaterials": {
        "keywords": ["hydrogel", "bioprinting", "tissue", "scaffold", "stem cell"],
        "sub_areas": [
            ("3D bioprinting", "printing tissue constructs with cells"),
            ("Injectable hydrogels", "minimally invasive biomaterial delivery"),
            ("Bone tissue engineering", "scaffolds for bone regeneration"),
            ("Cardiac patches", "biomaterials for heart repair"),
            ("Neural interfaces", "materials for brain-machine interfaces"),
        ]
    },
    "Separations": {
        "keywords": ["membrane", "separation", "filtration", "desalination", "MOF"],
        "sub_areas": [
            ("MOF membranes", "metal-organic framework based separations"),
            ("Desalination", "reverse osmosis and forward osmosis"),
            ("Gas separation", "membranes for CO2 capture"),
            ("Ion-selective membranes", "membranes for specific ion transport"),
            ("Pervaporation", "membrane separation of liquid mixtures"),
        ]
    },
    "Metabolic_Engineering": {
        "keywords": ["metabolic", "fermentation", "biofuel", "CRISPR", "pathway"],
        "sub_areas": [
            ("Biofuel production", "engineering microbes for fuel synthesis"),
            ("Natural products", "biosynthesis of pharmaceuticals"),
            ("CO2 fixation", "engineering carbon fixation pathways"),
            ("Cell-free systems", "in vitro metabolic engineering"),
            ("Synthetic biology", "designing genetic circuits"),
        ]
    },
    "High_Temp_Materials": {
        "keywords": ["superalloy", "turbine", "creep", "oxidation", "coating"],
        "sub_areas": [
            ("Ni-based superalloys", "nickel superalloys for jet engines"),
            ("Thermal barrier coatings", "ceramic coatings for turbine blades"),
            ("High-entropy alloys", "multi-principal element alloys"),
            ("Refractory metals", "tungsten and molybdenum alloys"),
            ("Ceramic matrix composites", "SiC fiber reinforced ceramics"),
        ]
    },
    "Energy_Storage": {
        "keywords": ["battery", "lithium", "electrode", "electrolyte", "charging"],
        "sub_areas": [
            ("Li-ion cathodes", "cathode materials for lithium batteries"),
            ("Solid electrolytes", "solid-state battery electrolytes"),
            ("Na-ion batteries", "sodium-ion battery materials"),
            ("Supercapacitors", "high-power energy storage"),
            ("Battery recycling", "recovering materials from spent batteries"),
        ]
    },
}

DEPARTMENTS = [
    "Materials Science and Engineering",
    "Chemical Engineering",
    "Chemistry",
    "Mechanical Engineering",
    "Biomedical Engineering",
    "Physics",
    "Electrical Engineering",
]

POSITIONS = ["PhD Student", "PhD Student", "PhD Student", "Postdoc", "Assistant Professor", "Associate Professor", "Professor"]

FIRST_NAMES = ["John", "Emily", "Michael", "Sarah", "David", "Lisa", "Robert", "Anna", "Kevin", "Jennifer",
               "James", "Maria", "William", "Elizabeth", "Thomas", "Jessica", "Christopher", "Amanda", "Daniel", "Ashley",
               "Matthew", "Stephanie", "Andrew", "Nicole", "Joshua", "Michelle", "Ryan", "Laura", "Brandon", "Kimberly",
               "Wei", "Yuki", "Raj", "Priya", "Ahmed", "Fatima", "Chen", "Ming", "Sanjay", "Aisha"]

LAST_NAMES = ["Smith", "Chen", "Wang", "Johnson", "Lee", "Zhang", "Kim", "Martinez", "Brown", "Wu",
              "Garcia", "Miller", "Davis", "Wilson", "Moore", "Taylor", "Anderson", "Thomas", "Jackson", "White",
              "Harris", "Clark", "Lewis", "Robinson", "Walker", "Young", "Allen", "King", "Wright", "Scott",
              "Patel", "Kumar", "Singh", "Nakamura", "Tanaka", "Park", "Huang", "Liu", "Yang", "Zhao"]


def generate_researcher(idx: int, area: str, sub_area: tuple, dept: str):
    """Generate a single researcher profile"""
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    name = f"{first} {last}"
    position = random.choice(POSITIONS)

    sub_area_name, sub_area_desc = sub_area
    area_keywords = RESEARCH_AREAS[area]["keywords"]

    # Generate research interests
    interests = ", ".join(random.sample(area_keywords, min(3, len(area_keywords))))
    interests += f", {sub_area_name.lower()}"

    # Generate raw_text
    if position in ["Professor", "Associate Professor", "Assistant Professor"]:
        role_text = f"Prof. {name} leads a research group"
        advisor_text = ""
    else:
        advisor = f"Prof. {random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
        role_text = f"{name} is a {position.lower()}"
        advisor_text = f" under {advisor}"

    raw_text = f"{role_text} in {dept} at Cornell University{advisor_text}. Their research focuses on {sub_area_desc}. Key areas include {interests}."

    # Add some specific details based on area
    if "DNA" in area:
        raw_text += " They use techniques like AFM and TEM for characterization."
    elif "Computational" in area:
        raw_text += " Their work combines theory with high-performance computing."
    elif "Polymer" in area:
        raw_text += " They synthesize materials with controlled molecular weight and architecture."
    elif "Catalysis" in area:
        raw_text += " They use electrochemical and spectroscopic methods to study reaction mechanisms."
    elif "2D" in area:
        raw_text += " They grow materials using CVD and characterize them with Raman and electron microscopy."
    elif "Bio" in area:
        raw_text += " They work with cell culture and in vivo models to test their materials."
    elif "Separation" in area:
        raw_text += " They fabricate membranes and test permeability and selectivity."
    elif "Metabolic" in area:
        raw_text += " They use CRISPR and metabolomics to engineer pathways."
    elif "High_Temp" in area:
        raw_text += " They characterize materials at elevated temperatures using synchrotron techniques."
    elif "Energy" in area:
        raw_text += " They assemble and test cells to measure electrochemical performance."

    # Generate a paper
    paper_title = f"{sub_area_name}: A {'Computational' if 'Computational' in area else 'Novel'} Approach"
    paper_abstract = f"We present advances in {sub_area_desc.lower()}. Our results demonstrate improved performance."

    return {
        "id": f"cornell_{area.lower()}_{idx:03d}",
        "name": name,
        "position": position,
        "department": dept,
        "lab": f"{sub_area_name} Lab",
        "email": f"{first.lower()[0]}{last.lower()}@cornell.edu",
        "research_interests": interests,
        "raw_text": raw_text,
        "papers": [{"title": paper_title, "year": 2024, "abstract": paper_abstract}],
        "_area": area,  # Hidden field for ground truth
        "_sub_area": sub_area_name,
    }


def generate_ground_truth_queries(researchers: list) -> dict:
    """Generate queries with expected answers"""
    queries = []

    # Group researchers by area
    by_area = {}
    for r in researchers:
        area = r["_area"]
        if area not in by_area:
            by_area[area] = []
        by_area[area].append(r)

    # Generate specific queries for each area
    query_templates = {
        "DNA_Nanotechnology": [
            ("DNA origami folding and assembly", ["DNA origami", "self-assembly"]),
            ("nucleic acid nanostructures for drug delivery", ["DNA", "drug delivery"]),
        ],
        "Computational_Materials": [
            ("machine learning for battery materials discovery", ["machine learning", "battery"]),
            ("DFT calculations for energy materials", ["DFT", "computational"]),
        ],
        "Polymer_Science": [
            ("RAFT polymerization synthesis", ["RAFT", "polymer"]),
            ("block copolymer self-assembly", ["block copolymer", "polymer"]),
        ],
        "Catalysis": [
            ("CO2 electrochemical reduction", ["CO2", "electrocatalysis"]),
            ("heterogeneous catalyst design", ["catalyst", "heterogeneous"]),
        ],
        "2D_Materials": [
            ("graphene CVD growth optimization", ["graphene", "CVD"]),
            ("MoS2 monolayer synthesis", ["MoS2", "2D"]),
        ],
        "Biomaterials": [
            ("3D bioprinting of tissue scaffolds", ["bioprinting", "tissue"]),
            ("hydrogel materials for regenerative medicine", ["hydrogel", "stem cell"]),
        ],
        "Separations": [
            ("MOF membrane for water desalination", ["MOF", "desalination"]),
            ("gas separation membranes", ["membrane", "separation"]),
        ],
        "Metabolic_Engineering": [
            ("CRISPR metabolic engineering for biofuels", ["CRISPR", "biofuel"]),
            ("fermentation process optimization", ["fermentation", "metabolic"]),
        ],
        "High_Temp_Materials": [
            ("nickel superalloy for turbine blades", ["superalloy", "turbine"]),
            ("thermal barrier coatings", ["coating", "oxidation"]),
        ],
        "Energy_Storage": [
            ("solid state battery electrolytes", ["solid electrolyte", "battery"]),
            ("lithium-ion cathode materials", ["lithium", "cathode"]),
        ],
    }

    for area, area_queries in query_templates.items():
        if area not in by_area:
            continue
        area_researchers = by_area[area]

        for query_text, keywords in area_queries:
            # Find best matching researcher
            best_match = None
            best_score = 0
            for r in area_researchers:
                score = sum(1 for kw in keywords if kw.lower() in r["raw_text"].lower())
                if score > best_score:
                    best_score = score
                    best_match = r

            if best_match:
                queries.append({
                    "id": f"query_{len(queries)+1:03d}",
                    "query": query_text,
                    "expected_id": best_match["id"],
                    "expected_name": best_match["name"],
                    "expected_relevant_keywords": keywords,
                    "area": area,
                })

    return queries


def main():
    print("Generating expanded test dataset...")

    researchers = []
    idx = 1

    # Generate researchers for each area
    for area, config in RESEARCH_AREAS.items():
        sub_areas = config["sub_areas"]

        # Generate 1-2 researchers per sub-area
        for sub_area in sub_areas:
            dept = random.choice(DEPARTMENTS[:3])  # Mostly MSE, ChemE, Chem
            researcher = generate_researcher(idx, area, sub_area, dept)
            researchers.append(researcher)
            idx += 1

            # Add a second researcher for some sub-areas (creates harder matching)
            if random.random() > 0.5:
                dept2 = random.choice(DEPARTMENTS)
                researcher2 = generate_researcher(idx, area, sub_area, dept2)
                researchers.append(researcher2)
                idx += 1

    print(f"Generated {len(researchers)} researchers")

    # Generate ground truth queries
    queries = generate_ground_truth_queries(researchers)
    print(f"Generated {len(queries)} test queries")

    # Save expanded researcher data
    output = {
        "metadata": {
            "created_at": "2026-01-26",
            "source": "Cornell University (SYNTHETIC DATA FOR TESTING)",
            "total_count": len(researchers),
            "note": "Expanded dataset with confounding cases for embedding optimization"
        },
        "researchers": researchers
    }

    output_file = DATA_DIR / "raw" / "researchers_expanded.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Saved to: {output_file}")

    # Save expanded test queries
    queries_output = {
        "description": "Expanded test queries for embedding optimization",
        "queries": queries
    }

    queries_file = DATA_DIR / "test_queries_expanded.json"
    with open(queries_file, 'w') as f:
        json.dump(queries_output, f, indent=2)
    print(f"Saved to: {queries_file}")

    # Print summary
    print("\n" + "="*50)
    print("DATASET SUMMARY")
    print("="*50)

    area_counts = {}
    for r in researchers:
        area = r["_area"]
        area_counts[area] = area_counts.get(area, 0) + 1

    for area, count in sorted(area_counts.items()):
        print(f"  {area}: {count} researchers")

    print(f"\nTotal: {len(researchers)} researchers, {len(queries)} queries")


if __name__ == "__main__":
    main()
