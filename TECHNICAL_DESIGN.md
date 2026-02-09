# Academic Dating App - Complete Technical Design (Detailed)

> **Version**: v1.0
> **Date**: 2026-01-24
> **Team**: Zeyu, Chenyun, Yifan

---

## 1. Project Overview

### 1.1 The Core Problem

When researchers read an interesting paper and want to discuss it, they face three dilemmas:
- Lab mates' research directions don't match — no one to talk to
- Directly contacting the paper's authors has a low response rate (especially for lesser-known researchers)
- There's definitely someone on campus who knows this field, but you don't know who

### 1.2 Our Solution

**"Tinder for Academia"**: User uploads a paper → AI conversation refines the need → Semantic matching → Recommend the right people on campus to discuss with

### 1.3 Core Value Proposition

**We're not building AI — we're using AI to solve a real pain point in academia.**

How we differ from existing products:
| Existing Product | Problem | Our Solution |
|------------------|---------|--------------|
| Google Scholar | Can only search papers, not "people who know this" | Directly recommend people + contact info |
| ResearchGate | Requires the other person to be registered and active | Passive users can also be discovered |
| Emailing the author directly | Low response rate, especially for lesser-known researchers | Recommend people on campus — easier to meet in person |

### 1.4 Distinctive Interaction Design

**"Client-Consultant" mindset**: The user can casually describe their interests (like a client), while the AI proactively digs into the real need (like a consultant).

Users don't need to articulate "what kind of person I'm looking for" — they just need to say "this paper is interesting," and the AI will ask follow-up questions to refine the request.

---

## 2. System Architecture

### 2.1 Overall Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Offline Data Preparation                         │
│                    (Done ahead of time, before users arrive)            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────┐ │
│   │ Data Scraping│ → │ Text Merging │ → │ Embedding   │ → │  FAISS  │ │
│   │ DrissionPage │    │ Concatenate │    │ Generation  │    │  Index  │ │
│   │             │    │ all info    │    │ (SPECTER2)  │    │ Storage │ │
│   └─────────────┘    └─────────────┘    └─────────────┘    └─────────┘ │
│                                                                         │
│   Data sources: University website → Personal pages → Google Scholar    │
│                 → LinkedIn                                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                         Online Service Layer                            │
│                    (Real-time processing when users arrive)             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────┐ │
│   │ User Uploads │ → │ LLM Need    │ → │ Query       │ → │  FAISS  │ │
│   │ Paper        │    │ Discovery   │    │ Embedding   │    │ Search  │ │
│   │             │    │ (Multi-turn)│    │ (SPECTER2)  │    │ (ms)    │ │
│   └─────────────┘    └─────────────┘    └─────────────┘    └─────────┘ │
│                                                                         │
│                                    ↓                                    │
│                                                                         │
│                          ┌─────────────────┐                            │
│                          │ Results Display  │                            │
│                          │ Recommended      │                            │
│                          │ people + reasons │                            │
│                          │ + contact info   │                            │
│                          └─────────────────┘                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 User Tier Design

| Tier | Description | Data Source | Capabilities |
|------|-------------|-------------|--------------|
| **Tier 1 Registered Users** | People who actively sign up | AI questionnaire + uploaded papers | Two-way matching; can see who is interested in them |
| **Tier 2 Passive Users** | Faculty/PhD students at the university | Scraped from the web | Can only be discovered by others; contact info provided |

**Demo phase**: Only Tier 2 (passive users), because:
- No user registration needed — faster demo
- Directly demonstrates core value: "Upload paper → Find the right person"

---

## 3. Module Design Details

### 3.1 Module 1: Data Scraping

#### 3.1.1 Data Source Priority

| Priority | Source | Information Available | Scraping Difficulty |
|----------|--------|----------------------|---------------------|
| 1 | University website | Name, title, department, email, research description | Easy |
| 2 | Personal/lab website | Detailed research intro, projects, news | Medium |
| 3 | Google Scholar | Paper list, citation count, collaborators | Medium |
| 4 | LinkedIn | Skills, Experience, Education | Requires login |

#### 3.1.2 Scraping Tool

**Using DrissionPage**

Why we chose it:
- More stable than Selenium
- More powerful than BeautifulSoup (handles JS-rendered pages)
- Python ecosystem, easy to learn

#### 3.1.3 Scraping Workflow

```
Step 1: Get the list of people
        Cornell website → Faculty page → Extract all professors' names and profile links
        Cornell website → PhD students page → Extract all students' names and advisors

Step 2: Scrape detailed information for each person
        For each person:
            - Scrape university profile page (research interests, contact info)
            - Scrape personal website (if available)
            - Scrape Google Scholar (paper abstracts)
            - Scrape LinkedIn (if publicly available)

Step 3: Merge information
        Concatenate all information for each person into a single long text
        Store in JSON format
```

#### 3.1.4 Data Storage Format

```json
{
  "id": "cornell_mse_001",
  "name": "San Zhang",
  "position": "PhD Student",
  "department": "Materials Science and Engineering",
  "advisor": "Prof. X",
  "email": "sz123@cornell.edu",
  "personal_website": "https://zhangsan.com",
  "lab": "DNA Materials Lab",
  "research_interests": "DNA origami, self-assembly, nanomaterials",
  "raw_text": "[Concatenated full text] San Zhang is a PhD student in the Materials Science department at Cornell, advised by Prof. X. His research focuses on DNA origami, self-assembled nanomaterials... Published papers: 1. xxx 2. xxx ...",
  "papers": [
    {"title": "xxx", "abstract": "xxx", "year": 2024},
    {"title": "xxx", "abstract": "xxx", "year": 2023}
  ],
  "sources": ["cornell.edu", "scholar.google.com"]
}
```

#### 3.1.5 Information Threshold Filtering

**Principle**: Too little information = no matching value = not indexed

**Minimum requirements**:
- Must have: name, email, department
- Must have at least one of: research description (>50 words) OR at least 1 paper abstract

People who don't meet the minimum requirements → skipped, no embedding generated

---

### 3.2 Module 2: Embedding Generation

#### 3.2.1 Why SPECTER2

| Model | Type | Features | Best For |
|-------|------|----------|----------|
| sentence-transformers | General-purpose | Fast, good for general text | General semantic search |
| **SPECTER2** | Academic-specific | Trained on 6M paper citation relationships | **Academic paper/researcher matching** |

SPECTER2 advantages:
- Understands academic jargon ("CVD", "AFM", "XRD" and other abbreviations)
- Learned citation relationships between papers (papers on similar research will have closer embeddings)
- Developed and maintained by the Allen AI team (the team behind Semantic Scholar)

#### 3.2.2 Code Example

```python
from transformers import AutoTokenizer
from adapters import AutoAdapterModel

# Load SPECTER2 model
tokenizer = AutoTokenizer.from_pretrained('allenai/specter2_base')
model = AutoAdapterModel.from_pretrained('allenai/specter2_base')
model.load_adapter("allenai/specter2", source="hf", load_as="specter2", set_active=True)

def generate_embedding(text):
    """Convert a text string into an embedding vector"""
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512
    )
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :]  # Take the [CLS] token vector
    return embedding.detach().numpy()

# Generate embedding for each researcher
for researcher in all_researchers:
    text = researcher['raw_text']  # Concatenated full text
    embedding = generate_embedding(text)
    researcher['embedding'] = embedding
```

#### 3.2.3 Validating Embedding Effectiveness

**Simple test method** (takes 5 minutes):

```python
from scipy.spatial.distance import cosine

# Manually create 3 test cases
person_a = "Zhang San, researches DNA origami, self-assembled nanostructures, biomaterials"
person_b = "Li Si, researches DNA nanotechnology, nucleic acid self-assembly, molecular machines"
person_c = "Wang Wu, researches high-temperature alloys, turbine blade materials, aerospace engines"

emb_a = generate_embedding(person_a)
emb_b = generate_embedding(person_b)
emb_c = generate_embedding(person_c)

# Calculate distances (smaller = more similar)
dist_ab = cosine(emb_a, emb_b)  # Zhang San vs Li Si (both work on DNA)
dist_ac = cosine(emb_a, emb_c)  # Zhang San vs Wang Wu (completely different fields)

print(f"Zhang San - Li Si distance: {dist_ab}")   # Should be small
print(f"Zhang San - Wang Wu distance: {dist_ac}")  # Should be large

# Verify: if dist_ab < dist_ac, then embeddings are working
assert dist_ab < dist_ac, "Embedding validation failed!"
print("Embedding validation passed!")
```

---

### 3.3 Module 3: FAISS Vector Index

#### 3.3.1 What is FAISS

**In plain terms**: A tool that can find the most similar vectors among millions of embeddings in milliseconds.

**Why we need it**:
- With 1000 researchers, computing distances for all 1000 every time → slow
- FAISS uses mathematical tricks to skip most irrelevant entries → fast

#### 3.3.2 Code Example

```python
import faiss
import numpy as np

# Suppose we have 200 researchers, each embedding is 768-dimensional
all_embeddings = np.array([r['embedding'] for r in all_researchers])  # shape: (200, 768)

# Create FAISS index
dimension = 768
index = faiss.IndexFlatL2(dimension)  # Using L2 (Euclidean) distance

# Add all embeddings to the index
index.add(all_embeddings.astype('float32'))

print(f"Index contains {index.ntotal} vectors")

# Search: given a query embedding, find the top 10 most similar
def search_similar(query_embedding, top_k=10):
    query = query_embedding.astype('float32').reshape(1, -1)
    distances, indices = index.search(query, top_k)
    return indices[0], distances[0]  # Return indices and distances

# Usage example
query_emb = generate_embedding("DNA origami assembly methods, improving folding yield")
top_indices, top_distances = search_similar(query_emb, top_k=10)

# Display results
for i, (idx, dist) in enumerate(zip(top_indices, top_distances)):
    researcher = all_researchers[idx]
    print(f"#{i+1}: {researcher['name']} (distance: {dist:.4f})")
```

#### 3.3.3 Index Type Selection

| Data Size | Recommended Index | Features |
|-----------|-------------------|----------|
| <1,000 people | `IndexFlatL2` | Exact search, simple and direct |
| 1,000–100K people | `IndexIVFFlat` | Approximate search, faster |
| >100K people | `IndexHNSW` | Graph-based, fastest |

**Demo phase**: `IndexFlatL2` is sufficient (200 people — exact search is already millisecond-level)

---

### 3.4 Module 4: LLM-Powered Need Discovery (Core Innovation)

#### 3.4.1 Design Philosophy

**Problem**: Users are often vague
- "I think this paper is interesting" → What's interesting? The method? The conclusions?
- "I want to find someone to talk to" → Talk about what? Theory? Experiments? Applications?

**Solution**: Have the LLM act like a smart consultant, proactively asking follow-up questions

#### 3.4.2 Conversation Flow Design

```
┌─────────────────────────────────────────────────────────────────┐
│ Step 1: Paper Analysis                                          │
├─────────────────────────────────────────────────────────────────┤
│ User uploads a paper                                            │
│       ↓                                                         │
│ LLM analyzes the paper and identifies discussion dimensions:    │
│   - Research background / motivation                            │
│   - Experimental methods / techniques                           │
│   - Theoretical models                                          │
│   - Characterization methods                                    │
│   - Application scenarios                                       │
│   - Future directions                                           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 2: Proactive Inquiry                                       │
├─────────────────────────────────────────────────────────────────┤
│ LLM: "I've read this paper. It covers three main topics:        │
│       1. DNA origami design methods                             │
│       2. Thermal annealing assembly process                     │
│       3. AFM characterization results                           │
│       Which aspect would you like to dive deeper into?"         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 3: User Response (can be vague)                            │
├─────────────────────────────────────────────────────────────────┤
│ User: "Well... I think their folding approach is pretty cool,   │
│        I keep having trouble getting mine to fold properly..."  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 4: LLM Extracts + Confirms                                │
├─────────────────────────────────────────────────────────────────┤
│ LLM: "It sounds like you mainly want to discuss DNA origami     │
│       assembly techniques, specifically how to improve folding  │
│       yield? Is that right?"                                    │
│                                                                 │
│ User: "Yes, exactly!"                                           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 5: Generate Structured Query                               │
├─────────────────────────────────────────────────────────────────┤
│ LLM internally generates:                                       │
│ "DNA origami assembly, folding optimization, yield improvement" │
│                                                                 │
│ This text is sent to generate a query embedding for matching    │
└─────────────────────────────────────────────────────────────────┘
```

#### 3.4.3 System Prompt Design

```
You are an academic matching assistant. The user will upload a paper, and your task is to:

1. Analyze the paper and identify the main discussion dimensions
   (methods, theory, characterization, applications, etc.)
2. Ask the user in concise language which aspect they want to dive deeper into
3. Based on the user's response (which may be vague), extract the core topic
   they actually want to discuss
4. If the user's response is unclear, ask follow-up questions (2 rounds max)
5. Ultimately output a clear, semantically matchable need description

Notes:
- Users may speak casually; you need to understand their true intent
  like a smart consultant
- Don't ask too many questions — wrap it up within 2-3 conversation turns
- The final need description should be in English technical keywords
  for optimal matching
```

#### 3.4.4 Edge Case Handling (via Prompt)

| Edge Case | Handling Approach |
|-----------|-------------------|
| User uploads a scanned PDF | LLM detects no extractable text; prompts user to upload a text-based version |
| User uploads an image | LLM prompts "Please upload a PDF or text-based paper" |
| User uploads a Chinese paper | LLM handles it normally (SPECTER2 supports Chinese but works better with English); suggests providing an English abstract |
| User is too vague | LLM asks follow-up questions (2 rounds max) |

---

### 3.5 Module 5: Results Display

#### 3.5.1 Display Content

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Recommendation #1: San Zhang (Match Score: 92%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Identity:  PhD Student @ Materials Science and Engineering
Lab:       DNA Materials Lab (Advisor: Prof. X)
Contact:   sz123@cornell.edu

Why recommended:
   San Zhang has published 3 DNA origami-related papers, with research
   focusing on:
   - Thermal annealing assembly optimization for DNA origami
   - Yield improvement for large-scale origami structures
   - AFM characterization methods

Personal website: https://zhangsan.com
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

#### 3.5.2 Match Score Calculation

```python
def calculate_match_score(distance, max_distance=2.0):
    """Convert distance to a match percentage"""
    # Smaller distance = higher match score
    # Assume max distance is 2.0; anything beyond is 0% match
    score = max(0, (max_distance - distance) / max_distance * 100)
    return round(score, 1)
```

#### 3.5.3 Explainability (Why This Person Was Recommended)

**Method 1: Keyword-based matching**
```python
def explain_match(user_need, researcher):
    """Generate recommendation rationale"""
    # Extract keywords from user need
    user_keywords = extract_keywords(user_need)  # ["DNA origami", "assembly", "yield"]

    # Find where these keywords appear in the researcher's information
    matches = []
    for keyword in user_keywords:
        if keyword in researcher['raw_text']:
            matches.append(keyword)

    return f"This researcher's work involves: {', '.join(matches)}"
```

**Method 2: LLM-generated explanation** (smarter, but requires an extra API call)
```python
def explain_match_with_llm(user_need, researcher):
    prompt = f"""
    The user wants to discuss: {user_need}

    This researcher's information: {researcher['raw_text'][:500]}

    Please explain in 1-2 sentences why this researcher is recommended.
    """
    return call_llm(prompt)
```

---

## 4. Technology Stack Summary

| Module | Technology | Description |
|--------|-----------|-------------|
| Data Scraping | DrissionPage | Python scraper that handles JS-rendered pages |
| Data Storage | JSON files | Simplest approach for the demo phase |
| Embedding Generation | SPECTER2 (HuggingFace) | Academic paper-specific model, best performance |
| Vector Search | FAISS | Open-sourced by Meta, millisecond-level retrieval |
| LLM Conversation | Gemini 2.0 Flash | API provided by the hackathon |
| Frontend | Streamlit | Rapid web app prototyping |
| Backend | FastAPI | Python ecosystem, simple and efficient |

---

## 5. Data Scope (Demo)

| Department | Faculty | PhD | Total |
|------------|---------|-----|-------|
| Materials Science & Engineering | ~30 | ~90 | ~120 |
| Chemical Engineering | ~25 | ~80 | ~105 |
| **Total** | **~55** | **~170** | **~225** |

---

## 6. References

### 6.1 Reusable Code Resources

| Resource | Link | Usage |
|----------|------|-------|
| OpenReview Paper-Reviewer Matching | https://github.com/openreview/openreview-expertise | Architecture reference |
| SPECTER2 Model | https://huggingface.co/allenai/specter2 | Embedding generation |
| sentence-transformers Examples | https://www.sbert.net/examples/applications/semantic-search/ | Semantic search code |
| FAISS Tutorial | https://huggingface.co/learn/llm-course/en/chapter5/6 | Vector indexing |

### 6.2 Paper References

- SPECTER: Document-level Representation Learning using Citation-informed Transformers (ACL 2020)
- SciBERT: A Pretrained Language Model for Scientific Text (EMNLP 2019)

---

## 7. Risks and Contingency Plans

| Risk | Likelihood | Contingency Plan |
|------|------------|------------------|
| SPECTER2 loads too slowly | Medium | Use a general-purpose sentence-transformers model |
| University website has anti-scraping measures | Low | Manually collect data |
| Gemini API is unstable | Low | Switch to GPT-4 or Claude |
| FAISS installation fails | Low | Use pure numpy for cosine distance computation |

---

## 8. Success Criteria

### 8.1 Minimum Viable Demo

- [ ] Can upload a paper
- [ ] AI can analyze the paper and ask questions
- [ ] Returns at least 5 matched researchers
- [ ] Each result includes contact information

### 8.2 Bonus Features

- [ ] Match score percentage
- [ ] Explainable recommendation rationale
- [ ] Polished UI
- [ ] Smooth user experience

---

## Appendix A: File Structure

```
academic_dating_app/
├── data/
│   ├── raw/                    # Raw scraped data
│   │   ├── cornell_mse.json
│   │   └── cornell_cheme.json
│   ├── processed/              # Processed data (with embeddings)
│   │   └── all_researchers.json
│   └── faiss_index/            # FAISS index files
│       └── researchers.index
├── src/
│   ├── scraper/                # Data scraping module
│   │   ├── cornell_scraper.py
│   │   └── utils.py
│   ├── embedding/              # Embedding generation module
│   │   ├── specter2_encoder.py
│   │   └── faiss_indexer.py
│   ├── matching/               # Matching logic module
│   │   ├── search.py
│   │   └── explainer.py
│   └── llm/                    # LLM conversation module
│       ├── paper_analyzer.py
│       └── need_extractor.py
├── app/
│   ├── main.py                 # Streamlit main app
│   └── components/             # UI components
├── tests/
│   └── test_embedding.py       # Embedding validation tests
├── requirements.txt
└── README.md
```

---

## Appendix B: API Design

```python
# POST /api/upload_paper
# Upload a paper and return the analysis results
Request:
{
    "paper_file": <PDF binary>
}
Response:
{
    "paper_id": "xxx",
    "title": "xxx",
    "abstract": "xxx",
    "discussion_topics": ["methods", "theory", "applications"]
}

# POST /api/chat
# Conversation to refine user needs
Request:
{
    "paper_id": "xxx",
    "user_message": "I want to talk about that folding approach"
}
Response:
{
    "assistant_message": "Are you interested in discussing DNA origami assembly techniques?",
    "need_clarification": true
}

# POST /api/search
# Search for matched researchers
Request:
{
    "paper_id": "xxx",
    "refined_need": "DNA origami assembly optimization"
}
Response:
{
    "results": [
        {
            "name": "San Zhang",
            "position": "PhD Student",
            "email": "sz123@cornell.edu",
            "match_score": 92.3,
            "reason": "Researches DNA origami assembly optimization"
        },
        ...
    ]
}
```
