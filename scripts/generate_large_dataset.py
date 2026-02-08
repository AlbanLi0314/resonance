#!/usr/bin/env python3
"""
Generate Large Synthetic Dataset
================================
Creates a larger, more realistic synthetic dataset for robust testing.
"""

import json
import random
from pathlib import Path
from typing import Dict, List

# Research areas with detailed sub-specialties
RESEARCH_DOMAINS = {
    "DNA_Nanotechnology": {
        "keywords": ["DNA origami", "self-assembly", "nanomaterials", "nucleic acid", "nanostructures"],
        "specialties": [
            ("DNA origami design", "designing 3D DNA nanostructures using computational tools and experimental validation"),
            ("DNA self-assembly", "thermal annealing and assembly protocols for complex DNA architectures"),
            ("DNA-based drug delivery", "using DNA nanostructures for targeted therapeutics and controlled release"),
            ("DNA computing", "molecular computing, logic gates, and information processing using DNA"),
            ("DNA-protein conjugates", "hybrid materials combining DNA scaffolds with functional proteins"),
            ("DNA hydrogels", "DNA-crosslinked hydrogels for biomedical applications"),
            ("DNA sensors", "DNA-based biosensors for detecting molecules and pathogens"),
        ]
    },
    "Computational_Materials": {
        "keywords": ["DFT", "machine learning", "computational", "simulation", "molecular dynamics", "ab initio"],
        "specialties": [
            ("DFT for batteries", "density functional theory calculations for battery electrode materials"),
            ("ML materials discovery", "neural networks and deep learning for predicting material properties"),
            ("Molecular dynamics", "atomistic simulations of materials behavior and phase transitions"),
            ("High-throughput screening", "automated computational screening of thousands of candidate materials"),
            ("Multiscale modeling", "bridging atomistic to continuum simulations for complex phenomena"),
            ("Materials informatics", "data mining and machine learning on materials databases"),
            ("Quantum chemistry", "accurate electronic structure calculations for molecules and solids"),
        ]
    },
    "Polymer_Science": {
        "keywords": ["polymer", "synthesis", "RAFT", "ATRP", "polymerization", "macromolecular"],
        "specialties": [
            ("Controlled polymerization", "RAFT, ATRP, and NMP synthesis methods for precise polymers"),
            ("Block copolymers", "self-assembling polymer architectures for nanolithography"),
            ("Sustainable polymers", "bio-based, biodegradable, and recyclable polymers"),
            ("Polymer nanocomposites", "polymer matrices reinforced with nanoparticles"),
            ("Conductive polymers", "conjugated polymers for organic electronics"),
            ("Polymer brushes", "surface-grafted polymers for responsive coatings"),
            ("Supramolecular polymers", "non-covalent assemblies and dynamic materials"),
        ]
    },
    "Catalysis": {
        "keywords": ["catalysis", "catalyst", "reaction", "conversion", "selectivity", "mechanism"],
        "specialties": [
            ("CO2 electrocatalysis", "electrochemical CO2 reduction to fuels and chemicals"),
            ("Heterogeneous catalysis", "solid catalysts for industrial chemical reactions"),
            ("Photocatalysis", "light-driven catalytic reactions for solar fuels"),
            ("Single-atom catalysis", "atomically dispersed metal catalysts for high selectivity"),
            ("Enzyme mimics", "artificial enzymes and biomimetic catalysis"),
            ("Electrocatalysis", "catalysts for fuel cells and electrolyzers"),
            ("Organocatalysis", "metal-free organic catalysts for asymmetric synthesis"),
        ]
    },
    "2D_Materials": {
        "keywords": ["graphene", "2D", "MoS2", "CVD", "heterostructure", "monolayer", "van der Waals"],
        "specialties": [
            ("Graphene synthesis", "CVD growth and transfer of high-quality graphene"),
            ("TMD synthesis", "transition metal dichalcogenide growth and characterization"),
            ("2D heterostructures", "van der Waals heterostructure assembly and properties"),
            ("2D electronics", "transistors and integrated circuits from 2D materials"),
            ("2D photonics", "optical properties and optoelectronic devices"),
            ("2D magnetism", "magnetic 2D materials and spintronic applications"),
            ("MXenes", "2D transition metal carbides and nitrides"),
        ]
    },
    "Biomaterials": {
        "keywords": ["hydrogel", "bioprinting", "tissue", "scaffold", "stem cell", "biocompatible"],
        "specialties": [
            ("3D bioprinting", "printing tissue constructs with living cells"),
            ("Injectable hydrogels", "minimally invasive biomaterial delivery systems"),
            ("Bone tissue engineering", "scaffolds and growth factors for bone regeneration"),
            ("Cardiac patches", "biomaterials for heart tissue repair"),
            ("Neural interfaces", "materials for brain-machine interfaces and neural probes"),
            ("Drug delivery systems", "controlled release particles and implants"),
            ("Wound healing", "biomaterials for skin regeneration and wound care"),
        ]
    },
    "Membrane_Separations": {
        "keywords": ["membrane", "separation", "filtration", "desalination", "MOF", "permeability"],
        "specialties": [
            ("MOF membranes", "metal-organic framework based separation membranes"),
            ("Desalination", "reverse osmosis and forward osmosis for water treatment"),
            ("Gas separation", "membranes for CO2 capture and hydrogen purification"),
            ("Ion-selective membranes", "membranes for specific ion transport and batteries"),
            ("Pervaporation", "membrane separation of liquid mixtures"),
            ("Nanofiltration", "selective removal of organic molecules and divalent ions"),
            ("Membrane bioreactors", "combining membranes with biological treatment"),
        ]
    },
    "Metabolic_Engineering": {
        "keywords": ["metabolic", "fermentation", "biofuel", "CRISPR", "pathway", "biosynthesis"],
        "specialties": [
            ("Biofuel production", "engineering microbes for sustainable fuel synthesis"),
            ("Natural products", "biosynthesis of pharmaceuticals and fine chemicals"),
            ("CO2 fixation", "engineering carbon fixation pathways in microbes"),
            ("Cell-free systems", "in vitro metabolic engineering without living cells"),
            ("Synthetic biology", "designing genetic circuits and cellular programs"),
            ("Protein engineering", "directed evolution and rational design of enzymes"),
            ("Microbiome engineering", "modifying microbial communities for applications"),
        ]
    },
    "High_Temperature_Materials": {
        "keywords": ["superalloy", "turbine", "creep", "oxidation", "coating", "refractory"],
        "specialties": [
            ("Ni-based superalloys", "nickel superalloys for jet engine hot sections"),
            ("Thermal barrier coatings", "ceramic coatings for turbine blade protection"),
            ("High-entropy alloys", "multi-principal element alloys with unique properties"),
            ("Refractory metals", "tungsten, molybdenum, and tantalum alloys"),
            ("Ceramic matrix composites", "SiC fiber reinforced ceramics for aerospace"),
            ("Additive manufacturing", "3D printing of high-temperature alloys"),
            ("Oxidation resistant alloys", "alumina and chromia forming alloys"),
        ]
    },
    "Energy_Storage": {
        "keywords": ["battery", "lithium", "electrode", "electrolyte", "capacitor", "charging"],
        "specialties": [
            ("Li-ion cathodes", "layered, spinel, and olivine cathode materials"),
            ("Solid electrolytes", "ceramic and polymer solid-state electrolytes"),
            ("Na-ion batteries", "sodium-ion battery materials as Li alternatives"),
            ("Supercapacitors", "high-power electrochemical capacitors"),
            ("Li-S batteries", "lithium-sulfur batteries for high energy density"),
            ("Battery recycling", "recovering valuable materials from spent batteries"),
            ("Silicon anodes", "high-capacity silicon-based anode materials"),
        ]
    },
    "Quantum_Materials": {
        "keywords": ["quantum", "superconductor", "topological", "spin", "qubit", "coherence"],
        "specialties": [
            ("Superconductors", "high-temperature and unconventional superconductors"),
            ("Topological insulators", "materials with protected surface states"),
            ("Quantum computing materials", "materials for qubits and quantum devices"),
            ("Spintronics", "spin-based electronics and magnetic memory"),
            ("Strongly correlated systems", "materials with strong electron interactions"),
            ("Quantum dots", "semiconductor nanocrystals for quantum applications"),
        ]
    },
    "Soft_Matter": {
        "keywords": ["colloid", "liquid crystal", "emulsion", "surfactant", "self-assembly", "rheology"],
        "specialties": [
            ("Colloidal assembly", "self-organization of colloidal particles"),
            ("Liquid crystals", "ordered fluids for displays and sensors"),
            ("Emulsions", "stabilization and applications of droplet systems"),
            ("Active matter", "self-propelled particles and living systems"),
            ("Gels and networks", "physical and chemical gelation mechanisms"),
            ("Interfacial phenomena", "surface tension and wetting behavior"),
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
    "Computer Science",
    "Applied Physics",
    "Civil Engineering",
]

POSITIONS = [
    "PhD Student", "PhD Student", "PhD Student", "PhD Student",  # 40%
    "Postdoc", "Postdoc",  # 20%
    "Assistant Professor", "Assistant Professor",  # 20%
    "Associate Professor",  # 10%
    "Professor",  # 10%
]

FIRST_NAMES = [
    "John", "Emily", "Michael", "Sarah", "David", "Lisa", "Robert", "Anna", "Kevin", "Jennifer",
    "James", "Maria", "William", "Elizabeth", "Thomas", "Jessica", "Christopher", "Amanda", "Daniel", "Ashley",
    "Matthew", "Stephanie", "Andrew", "Nicole", "Joshua", "Michelle", "Ryan", "Laura", "Brandon", "Kimberly",
    "Wei", "Yuki", "Raj", "Priya", "Ahmed", "Fatima", "Chen", "Ming", "Sanjay", "Aisha",
    "Carlos", "Sofia", "Hans", "Ingrid", "Pierre", "Marie", "Takeshi", "Yoko", "Olga", "Ivan",
    "Jun", "Mei", "Hiroshi", "Sakura", "Mohammed", "Layla", "Andrei", "Natasha", "Luis", "Carmen",
]

LAST_NAMES = [
    "Smith", "Chen", "Wang", "Johnson", "Lee", "Zhang", "Kim", "Martinez", "Brown", "Wu",
    "Garcia", "Miller", "Davis", "Wilson", "Moore", "Taylor", "Anderson", "Thomas", "Jackson", "White",
    "Harris", "Clark", "Lewis", "Robinson", "Walker", "Young", "Allen", "King", "Wright", "Scott",
    "Patel", "Kumar", "Singh", "Nakamura", "Tanaka", "Park", "Huang", "Liu", "Yang", "Zhao",
    "Mueller", "Schmidt", "Schneider", "Fischer", "Weber", "Hoffmann", "Becker", "Schulz",
    "Petrov", "Ivanov", "Popov", "Sokolov", "Fernandez", "Rodriguez", "Gonzalez", "Lopez",
]


def generate_researcher(idx: int, domain: str, specialty: tuple, dept: str) -> Dict:
    """Generate a single synthetic researcher"""
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    name = f"{first} {last}"
    position = random.choice(POSITIONS)

    specialty_name, specialty_desc = specialty
    domain_keywords = RESEARCH_DOMAINS[domain]["keywords"]

    # Generate research interests (3-5 keywords)
    num_keywords = random.randint(3, 5)
    interests = random.sample(domain_keywords, min(num_keywords, len(domain_keywords)))
    interests.append(specialty_name.lower().replace("-", " "))
    interests_str = ", ".join(interests)

    # Generate raw_text with some variation
    if position in ["Professor", "Associate Professor", "Assistant Professor"]:
        role_text = f"Prof. {name} leads a research group in {dept}"
        advisor_text = ""
    else:
        advisor = f"Prof. {random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
        role_text = f"{name} is a {position.lower()} in {dept}"
        advisor_text = f" working with {advisor}"

    raw_text = f"{role_text} at Cornell University{advisor_text}. "
    raw_text += f"Their research focuses on {specialty_desc}. "
    raw_text += f"Key interests include {interests_str}. "

    # Add some random details
    techniques = [
        "They use advanced characterization techniques.",
        "Their work combines experiments with computational modeling.",
        "They collaborate with industry partners.",
        "Their group has published extensively in top journals.",
        "They have received multiple awards for their research.",
    ]
    raw_text += random.choice(techniques)

    # Generate papers
    num_papers = random.randint(1, 3)
    papers = []
    for p in range(num_papers):
        paper_title = f"{specialty_name}: {'Novel' if p == 0 else 'Advanced'} Approaches and Applications"
        paper_abstract = f"We present {'new' if p == 0 else 'further'} advances in {specialty_desc.lower()}. "
        paper_abstract += f"Our results demonstrate improved performance in {random.choice(interests)}."
        papers.append({
            "title": paper_title,
            "year": 2024 - p,
            "abstract": paper_abstract
        })

    return {
        "id": f"cornell_{domain.lower()}_{idx:04d}",
        "name": name,
        "position": position,
        "department": dept,
        "lab": f"{specialty_name} Lab",
        "email": f"{first.lower()}.{last.lower()}@cornell.edu",
        "research_interests": interests_str,
        "raw_text": raw_text,
        "papers": papers,
        "_domain": domain,
        "_specialty": specialty_name,
    }


def generate_test_queries(researchers: List[Dict]) -> List[Dict]:
    """Generate test queries with ground truth"""
    queries = []

    # Group researchers by domain
    by_domain = {}
    for r in researchers:
        domain = r.get("_domain", "Unknown")
        if domain not in by_domain:
            by_domain[domain] = []
        by_domain[domain].append(r)

    # Generate 2-3 queries per domain
    for domain, config in RESEARCH_DOMAINS.items():
        if domain not in by_domain:
            continue

        domain_researchers = by_domain[domain]
        keywords = config["keywords"]

        # Query 1: General domain query
        query1 = f"{' '.join(random.sample(keywords, 2))} research"
        best_match1 = random.choice(domain_researchers)
        queries.append({
            "id": f"query_{len(queries)+1:03d}",
            "query": query1,
            "expected_id": best_match1["id"],
            "expected_name": best_match1["name"],
            "expected_keywords": keywords[:3],
            "domain": domain,
        })

        # Query 2: Specific specialty query
        if domain_researchers:
            target = random.choice(domain_researchers)
            specialty = target.get("_specialty", "")
            query2 = f"{specialty} applications"
            queries.append({
                "id": f"query_{len(queries)+1:03d}",
                "query": query2,
                "expected_id": target["id"],
                "expected_name": target["name"],
                "expected_keywords": [specialty.lower()],
                "domain": domain,
            })

    return queries


def generate_large_dataset(num_researchers: int = 300, output_file: Path = None) -> Dict:
    """
    Generate a large synthetic dataset.

    Args:
        num_researchers: Target number of researchers
        output_file: Path to save the dataset

    Returns:
        Summary statistics
    """
    print(f"Generating {num_researchers} synthetic researchers...")

    researchers = []
    idx = 1

    # Calculate researchers per domain
    domains = list(RESEARCH_DOMAINS.keys())
    per_domain = num_researchers // len(domains)

    for domain in domains:
        specialties = RESEARCH_DOMAINS[domain]["specialties"]

        for _ in range(per_domain):
            specialty = random.choice(specialties)
            dept = random.choice(DEPARTMENTS[:5])  # Prefer core depts
            researcher = generate_researcher(idx, domain, specialty, dept)
            researchers.append(researcher)
            idx += 1

    # Shuffle to mix domains
    random.shuffle(researchers)

    # Generate test queries
    queries = generate_test_queries(researchers)

    print(f"Generated {len(researchers)} researchers")
    print(f"Generated {len(queries)} test queries")

    # Save dataset
    if output_file:
        output_file = Path(output_file)

        dataset = {
            "metadata": {
                "created_at": "2026-01-26",
                "source": "Synthetic Data for Testing",
                "total_count": len(researchers),
                "domains": list(RESEARCH_DOMAINS.keys()),
            },
            "researchers": researchers
        }

        with open(output_file, 'w') as f:
            json.dump(dataset, f, indent=2)
        print(f"Saved researchers to: {output_file}")

        # Save queries
        queries_file = output_file.parent / "test_queries_large.json"
        queries_data = {
            "description": "Test queries for large synthetic dataset",
            "queries": queries
        }
        with open(queries_file, 'w') as f:
            json.dump(queries_data, f, indent=2)
        print(f"Saved queries to: {queries_file}")

    # Domain distribution
    domain_counts = {}
    for r in researchers:
        d = r.get("_domain", "Unknown")
        domain_counts[d] = domain_counts.get(d, 0) + 1

    return {
        "num_researchers": len(researchers),
        "num_queries": len(queries),
        "domains": domain_counts,
        "output_file": str(output_file) if output_file else None,
    }


if __name__ == "__main__":
    from pathlib import Path
    output = Path(__file__).parent.parent / "data" / "overnight_results" / "researchers_large.json"
    result = generate_large_dataset(300, output)
    print(f"\nResult: {result}")
