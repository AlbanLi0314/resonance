"""
Dual Adapter SPECTER2 Encoder
=============================
Uses different adapters for queries vs documents as recommended by AllenAI:
- adhoc_query: For user search queries
- proximity: For researcher documents

Expected improvement: 5-15% better retrieval performance.
"""
import os
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path

# Load .env
_env_file = Path(__file__).parent.parent / ".env"
if _env_file.exists():
    with open(_env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


class DualAdapterEncoder:
    """
    SPECTER2 encoder with asymmetric adapters for queries and documents.

    This follows the recommended usage from AllenAI:
    - User queries → specter2_adhoc_query adapter
    - Researcher documents → specter2_proximity adapter

    Usage:
        encoder = DualAdapterEncoder()
        encoder.load()

        # For user queries
        query_emb = encoder.encode_query("quantum computing materials")

        # For researcher documents
        doc_embs = encoder.encode_documents(["researcher text 1", "researcher text 2"])
    """

    def __init__(self,
                 model_name: str = "allenai/specter2_base",
                 query_adapter: str = "allenai/specter2_adhoc_query",
                 doc_adapter: str = "allenai/specter2_proximity",
                 max_length: int = 512):
        """
        Initialize dual adapter encoder.

        Args:
            model_name: Base SPECTER2 model
            query_adapter: Adapter for encoding queries
            doc_adapter: Adapter for encoding documents
            max_length: Maximum token length
        """
        self.model_name = model_name
        self.query_adapter = query_adapter
        self.doc_adapter = doc_adapter
        self.max_length = max_length

        self.model = None
        self.tokenizer = None
        self.device = None
        self._loaded = False

    def load(self):
        """Load model and both adapters"""
        import torch
        from transformers import AutoTokenizer
        from adapters import AutoAdapterModel

        print(f"Loading SPECTER2 Dual Adapter Encoder...")
        print(f"  Base model: {self.model_name}")
        print(f"  Query adapter: {self.query_adapter}")
        print(f"  Document adapter: {self.doc_adapter}")

        # Device selection - force CPU for compatibility
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("  Device: CUDA GPU")
        else:
            self.device = torch.device("cpu")
            print("  Device: CPU (MPS disabled for compatibility)")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Load base model
        self.model = AutoAdapterModel.from_pretrained(self.model_name)

        # Load BOTH adapters
        print("  Loading query adapter (adhoc_query)...")
        self.model.load_adapter(self.query_adapter, source="hf",
                               load_as="adhoc_query", set_active=False)

        print("  Loading document adapter (proximity)...")
        self.model.load_adapter(self.doc_adapter, source="hf",
                               load_as="proximity", set_active=False)

        self.model.to(self.device)
        self.model.eval()

        self._loaded = True
        print("✓ Dual Adapter Encoder loaded successfully!")

        return self

    def _encode(self, texts: List[str], adapter_name: str) -> np.ndarray:
        """Internal encoding with specified adapter"""
        import torch

        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Activate the specified adapter
        self.model.set_active_adapters(adapter_name)

        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)

        # Encode
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Take [CLS] token embedding
            embeddings = outputs.last_hidden_state[:, 0, :]

        return embeddings.cpu().numpy()

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a search query using the adhoc_query adapter.

        Args:
            query: User's search query

        Returns:
            numpy array of shape (768,)
        """
        embeddings = self._encode([query], "adhoc_query")
        return embeddings.flatten()

    def encode_queries(self, queries: List[str]) -> np.ndarray:
        """
        Encode multiple queries using the adhoc_query adapter.

        Args:
            queries: List of search queries

        Returns:
            numpy array of shape (n_queries, 768)
        """
        return self._encode(queries, "adhoc_query")

    def encode_document(self, text: str) -> np.ndarray:
        """
        Encode a single document using the proximity adapter.

        Args:
            text: Document text (researcher profile)

        Returns:
            numpy array of shape (768,)
        """
        embeddings = self._encode([text], "proximity")
        return embeddings.flatten()

    def encode_documents(self, texts: List[str], batch_size: int = 8,
                        show_progress: bool = True) -> np.ndarray:
        """
        Encode multiple documents using the proximity adapter.

        Args:
            texts: List of document texts
            batch_size: Batch size for encoding
            show_progress: Show progress bar

        Returns:
            numpy array of shape (n_documents, 768)
        """
        from tqdm import tqdm

        all_embeddings = []

        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding documents", unit="batch")

        for i in iterator:
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self._encode(batch_texts, "proximity")
            all_embeddings.append(batch_embeddings)

        return np.vstack(all_embeddings)

    # Compatibility methods for existing code
    def encode(self, text: str) -> np.ndarray:
        """Encode single text (defaults to query mode for backward compatibility)"""
        return self.encode_query(text)

    def encode_batch(self, texts: List[str], batch_size: int = 8,
                    show_progress: bool = True) -> np.ndarray:
        """Encode batch (defaults to document mode for backward compatibility)"""
        return self.encode_documents(texts, batch_size, show_progress)


def rebuild_index_with_dual_encoder(
    input_pkl: str = "data/processed/researchers_optimized.pkl",
    output_pkl: str = "data/processed/researchers_dual_adapter.pkl",
    output_index: str = "data/index/faiss_dual_adapter.index"
) -> Dict:
    """
    Rebuild the FAISS index using the proximity adapter for documents.

    This re-encodes all researcher documents with the correct adapter.

    Args:
        input_pkl: Path to existing researchers pkl (with old embeddings)
        output_pkl: Path to save new researchers pkl
        output_index: Path to save new FAISS index

    Returns:
        Dict with statistics
    """
    import pickle
    from pathlib import Path
    from src.embedding import format_researcher_text

    project_root = Path(__file__).parent.parent
    input_path = project_root / input_pkl
    output_pkl_path = project_root / output_pkl
    output_index_path = project_root / output_index

    print("=" * 60)
    print("REBUILDING INDEX WITH DUAL ADAPTER")
    print("=" * 60)

    # Load existing data
    print(f"\n1. Loading researchers from {input_pkl}...")
    with open(input_path, "rb") as f:
        researchers = pickle.load(f)
    print(f"   Loaded {len(researchers)} researchers")

    # Initialize dual encoder
    print("\n2. Loading dual adapter encoder...")
    encoder = DualAdapterEncoder()
    encoder.load()

    # Prepare texts for encoding
    print("\n3. Preparing researcher texts...")
    texts = []
    for r in researchers:
        # Use the optimized text format
        text = format_researcher_text(r, format_type="optimized")
        texts.append(text)
    print(f"   Prepared {len(texts)} texts")

    # Re-encode with proximity adapter
    print("\n4. Encoding with proximity adapter...")
    embeddings = encoder.encode_documents(texts, batch_size=8, show_progress=True)
    print(f"   Generated embeddings shape: {embeddings.shape}")

    # Update researcher embeddings
    print("\n5. Updating researcher embeddings...")
    for i, r in enumerate(researchers):
        r['embedding'] = embeddings[i]
        r['embedding_adapter'] = 'proximity'  # Mark which adapter was used

    # Save updated pkl
    print(f"\n6. Saving to {output_pkl}...")
    output_pkl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_pkl_path, "wb") as f:
        pickle.dump(researchers, f)

    # Build and save FAISS index
    print(f"\n7. Building FAISS index...")
    import faiss
    embeddings_float32 = embeddings.astype('float32')
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings_float32)

    output_index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_index_path))
    print(f"   Saved index with {index.ntotal} vectors")

    print("\n" + "=" * 60)
    print("REBUILD COMPLETE")
    print("=" * 60)
    print(f"  Researchers PKL: {output_pkl_path}")
    print(f"  FAISS Index: {output_index_path}")
    print(f"  Embedding adapter: proximity")
    print("=" * 60)

    return {
        "researchers_count": len(researchers),
        "embedding_shape": embeddings.shape,
        "output_pkl": str(output_pkl_path),
        "output_index": str(output_index_path)
    }


# Quick test
if __name__ == "__main__":
    print("Testing Dual Adapter Encoder...")
    print("=" * 50)

    encoder = DualAdapterEncoder()
    encoder.load()

    # Test query encoding
    print("\n1. Testing query encoding (adhoc_query adapter)...")
    query = "quantum computing materials for superconducting qubits"
    query_emb = encoder.encode_query(query)
    print(f"   Query: '{query[:50]}...'")
    print(f"   Embedding shape: {query_emb.shape}")

    # Test document encoding
    print("\n2. Testing document encoding (proximity adapter)...")
    docs = [
        "Dr. Smith researches quantum materials and superconductivity",
        "Prof. Jones works on battery materials and electrochemistry"
    ]
    doc_embs = encoder.encode_documents(docs, show_progress=False)
    print(f"   Documents: {len(docs)}")
    print(f"   Embeddings shape: {doc_embs.shape}")

    # Test similarity (query should be closer to first doc)
    print("\n3. Testing similarity...")
    from scipy.spatial.distance import cosine
    sim1 = 1 - cosine(query_emb, doc_embs[0])
    sim2 = 1 - cosine(query_emb, doc_embs[1])
    print(f"   Query vs Doc1 (quantum): {sim1:.4f}")
    print(f"   Query vs Doc2 (battery): {sim2:.4f}")
    print(f"   Correct ranking: {'✓ Yes' if sim1 > sim2 else '✗ No'}")

    print("\n" + "=" * 50)
    print("Test complete!")
