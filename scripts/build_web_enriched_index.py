#!/usr/bin/env python3
"""
Build FAISS Index from Web-Enriched Data
=========================================
Uses SPECTER2 with proximity adapter for document embeddings.
"""
import json
import pickle
import faiss
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def build_raw_text_for_embedding(researcher: dict) -> str:
    """
    Build optimized text for embedding from researcher data.

    Prioritizes:
    1. Web-enriched data (research_summary, keywords)
    2. Original research_interests
    3. Department context
    """
    parts = []

    name = researcher.get("name", "")
    department = researcher.get("department", "")

    parts.append(f"{name}")

    if department:
        parts.append(f", {department}")

    # Prefer web-enriched summary if available
    web_summary = researcher.get("web_research_summary", "")
    if web_summary and len(web_summary) > 50:
        parts.append(f": {web_summary}")
    else:
        # Fall back to original research interests
        interests = researcher.get("research_interests", "")
        if interests:
            # Clean up
            interests = interests.replace("Research Interests\n", "")
            parts.append(f": {interests[:500]}")

    # Add web keywords
    web_keywords = researcher.get("web_keywords", [])
    if web_keywords:
        parts.append(f" Keywords: {', '.join(web_keywords[:10])}")

    # Add web research areas
    web_areas = researcher.get("web_research_areas", [])
    if web_areas:
        parts.append(f" Areas: {', '.join(web_areas[:8])}")

    return "".join(parts)


def main():
    print("=" * 60)
    print("BUILD WEB-ENRICHED FAISS INDEX")
    print("=" * 60)

    # Load enriched data
    input_path = PROJECT_ROOT / "data/enriched/researchers_web_enriched.json"
    print(f"\nLoading: {input_path}")

    with open(input_path) as f:
        data = json.load(f)

    researchers = data["researchers"]
    print(f"Loaded {len(researchers)} researchers")

    # Build texts for embedding
    print("\nBuilding embedding texts...")
    texts = []
    for r in researchers:
        text = build_raw_text_for_embedding(r)
        texts.append(text)
        r["raw_text_for_embedding"] = text

    # Show samples
    print("\n--- Sample Texts ---")
    for i in [0, 50, 100]:
        if i < len(texts):
            print(f"\n[{i}] {researchers[i]['name']}:")
            print(f"    {texts[i][:200]}...")

    # Encode with SPECTER2
    print("\n" + "=" * 60)
    print("Encoding with SPECTER2 (proximity adapter)...")
    print("=" * 60)

    from transformers import AutoTokenizer
    from adapters import AutoAdapterModel

    model_name = "allenai/specter2_base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoAdapterModel.from_pretrained(model_name)

    # Load proximity adapter for documents
    model.load_adapter("allenai/specter2", source="hf", load_as="proximity", set_active=True)
    model.eval()

    print("Model loaded.")

    # Batch encode
    batch_size = 32
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        import torch
        with torch.no_grad():
            outputs = model(**inputs)
            # CLS token embedding
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(embeddings)

        print(f"  Encoded {min(i+batch_size, len(texts))}/{len(texts)}")

    embeddings = np.vstack(all_embeddings).astype("float32")
    print(f"\nEmbedding shape: {embeddings.shape}")

    # Add embeddings to researchers
    for i, r in enumerate(researchers):
        r["embedding"] = embeddings[i]

    # Build FAISS index
    print("\nBuilding FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    print(f"Index size: {index.ntotal}")

    # Save outputs
    output_dir = PROJECT_ROOT / "data/index"
    output_dir.mkdir(parents=True, exist_ok=True)

    index_path = output_dir / "faiss_web_enriched.index"
    faiss.write_index(index, str(index_path))
    print(f"Saved index: {index_path}")

    pkl_path = PROJECT_ROOT / "data/processed/researchers_web_enriched.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(researchers, f)
    print(f"Saved PKL: {pkl_path}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
