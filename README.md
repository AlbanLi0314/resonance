# ✦ Resonance

**Find the perfect researcher to discuss your paper with.**

Resonance is an AI-powered academic matchmaking tool that connects you with Cornell University researchers based on the papers you're reading. Upload a PDF, have a conversation with AI about your interests, and get matched with the most relevant researchers on campus.

## How It Works

1. **Upload** — Drop a research paper (PDF)
2. **Discuss** — Chat with Gemini AI about what aspects interest you
3. **Match** — Get ranked researcher recommendations with explanations

## Gemini 3 Integration

Resonance uses the Gemini 3 API in four critical ways:

- **Paper Analysis** — Gemini extracts title, abstract, and key topics from uploaded PDFs
- **Conversational Interest Discovery** — Multi-turn dialogue to understand exactly what the user wants to discuss
- **Query Expansion** — Enriches search queries with related academic terms (+4.5% precision improvement)
- **Match Explanation** — Generates human-readable explanations for why each researcher is a good fit

## Tech Stack

- **Embeddings**: SPECTER2 (scientific document embeddings)
- **Search**: Hybrid retrieval — FAISS dense search + BM25 sparse search with Reciprocal Rank Fusion
- **LLM**: Gemini 3 Flash for analysis, chat, query expansion, and explanations
- **Frontend**: Streamlit with custom CSS (glassmorphism dark theme)
- **Dataset**: 230 Cornell researchers across 3 departments

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set your Gemini API key
export GEMINI_API_KEY="your-key-here"

# Build the search index (first time only)
python scripts/step1_build_index.py

# Run the app
streamlit run v3.py
```

## Performance

| Metric | Score |
|--------|-------|
| P@1 (first result accuracy) | 77.8% |
| nDCG@5 | 0.884 |
| MRR | 0.847 |

Evaluated on 45 test queries with human-labeled ground truth.

## Project Structure

```
resonance/
├── v3.py                 # Main Streamlit app
├── src/
│   ├── embedding.py      # SPECTER2 encoder
│   ├── hybrid_search.py  # FAISS + BM25 fusion
│   ├── query_expansion.py    # Gemini query expansion
│   ├── match_explainer.py    # Gemini match explanations
│   ├── config.py         # Configuration
│   └── ...
├── data/
│   ├── processed/        # Researcher embeddings (pkl)
│   └── index/            # FAISS index files
├── assets/               # Logo and images
└── ground_truth/         # Evaluation framework
```

## License

MIT License - see [LICENSE](LICENSE)

---

*Built for the Google DeepMind Gemini 3 Hackathon*
