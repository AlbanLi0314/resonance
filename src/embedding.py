"""
SPECTER2 Embedding Encoder
Handles loading model and encoding text to vectors

Optimized Configuration (based on experiments):
- Text Format: T5_minimal = "{name}, {department}: {research_interests}"
- Model: allenai/specter2_base with allenai/specter2 adapter
- Distance: L2 (FAISS IndexFlatL2)
- Improvement: +40% Precision@1 over baseline raw_text format
"""
import os
import numpy as np
from typing import List, Union, Dict
from tqdm import tqdm

# Note: MPS (Apple Silicon GPU) has mutex issues with SPECTER2/adapters library.
# Instead of patching MPS globally (which breaks other models like cross-encoders),
# we force SPECTER2 to use CPU by default.


# ============================================================
# Text Formatting Functions (Optimized)
# ============================================================

def format_researcher_text(researcher: Dict, format_type: str = "optimized") -> str:
    """
    Format researcher data into text for embedding.

    Args:
        researcher: Dict with researcher fields
        format_type: One of "optimized", "raw_text", "minimal", "structured"

    Returns:
        Formatted text string for embedding

    The "optimized" format (T5_minimal) achieved 70% P@1 vs 50% for raw_text.
    """
    if format_type == "raw_text":
        # Baseline: use raw_text field directly
        return researcher.get("raw_text", "")

    elif format_type == "optimized" or format_type == "minimal":
        # T5_minimal: "{name}, {department}: {research_interests}"
        # This format achieved best results in experiments
        name = researcher.get("name", "Unknown")
        dept = researcher.get("department", "Unknown")
        interests = researcher.get("research_interests", "")
        return f"{name}, {dept}: {interests}"

    elif format_type == "structured":
        # T3_structured: Include paper titles
        interests = researcher.get("research_interests", "")
        papers = researcher.get("papers", [])
        paper_titles = ", ".join([p.get("title", "") for p in papers[:3]])
        return f"Research focus: {interests}. Recent work: {paper_titles}"

    elif format_type == "papers_only":
        # T4_papers_only: Only paper titles and abstracts
        papers = researcher.get("papers", [])
        texts = []
        for p in papers:
            texts.append(f"{p.get('title', '')}. {p.get('abstract', '')}")
        return " ".join(texts)

    else:
        # Default to optimized
        return format_researcher_text(researcher, "optimized")


def format_researchers_batch(researchers: List[Dict], format_type: str = "optimized") -> List[str]:
    """
    Format a batch of researchers for embedding.

    Args:
        researchers: List of researcher dicts
        format_type: Text format to use (default: "optimized")

    Returns:
        List of formatted text strings
    """
    return [format_researcher_text(r, format_type) for r in researchers]


class Specter2Encoder:
    """
    Wrapper for SPECTER2 model to encode academic text into embeddings

    Usage:
        encoder = Specter2Encoder()
        encoder.load()
        embedding = encoder.encode("DNA origami self-assembly")
        embeddings = encoder.encode_batch(["text1", "text2", "text3"])

    For batch processing (to avoid MPS mutex issues on Apple Silicon):
        encoder = Specter2Encoder(force_cpu=True)
    """

    def __init__(self, model_name: str = "allenai/specter2_base",
                 adapter_name: str = "allenai/specter2",
                 max_length: int = 512,
                 force_cpu: bool = False):
        self.model_name = model_name
        self.adapter_name = adapter_name
        self.max_length = max_length
        self.force_cpu = force_cpu
        self.model = None
        self.tokenizer = None
        self.device = None

    def load(self):
        """Load SPECTER2 model and adapter"""
        import torch
        from transformers import AutoTokenizer
        from adapters import AutoAdapterModel

        print(f"Loading SPECTER2 model: {self.model_name}")

        # Device selection:
        # - CUDA: Use if available
        # - MPS (Apple Silicon): DISABLED - causes mutex issues with adapters library
        # - CPU: Default fallback
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using GPU: CUDA")
        else:
            # Force CPU for SPECTER2 to avoid MPS mutex issues
            # MPS causes threading errors with the adapters library
            self.device = torch.device("cpu")
            print("Using CPU (MPS disabled for SPECTER2 compatibility)")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Load model with adapter
        self.model = AutoAdapterModel.from_pretrained(self.model_name)
        self.model.load_adapter(self.adapter_name, source="hf", 
                               load_as="specter2", set_active=True)
        self.model.to(self.device)
        self.model.eval()
        
        print("SPECTER2 model loaded successfully!")
        return self
    
    def encode(self, text: str) -> np.ndarray:
        """
        Encode a single text into embedding vector

        Args:
            text: Input text string

        Returns:
            numpy array of shape (768,)

        Raises:
            ValueError: If text is empty or None
            RuntimeError: If model not loaded or encoding fails
        """
        import torch

        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Validate input
        if not text or not text.strip():
            raise ValueError("Cannot encode empty text")

        # Clean text - remove excessive whitespace
        text = ' '.join(text.split())

        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Take [CLS] token embedding
            embedding = outputs.last_hidden_state[:, 0, :]

        return embedding.cpu().numpy().flatten()
    
    def encode_batch(self, texts: List[str], batch_size: int = 8,
                     show_progress: bool = True) -> np.ndarray:
        """
        Encode multiple texts into embeddings

        Args:
            texts: List of input text strings
            batch_size: Number of texts per batch
            show_progress: Whether to show progress bar

        Returns:
            numpy array of shape (n_texts, 768)
        """
        import torch

        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        all_embeddings = []

        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding", unit="batch")

        for i in iterator:
            batch_texts = texts[i:i + batch_size]

            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :]

            all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    print("Testing Specter2Encoder...")
    
    encoder = Specter2Encoder()
    encoder.load()
    
    # Test single encode
    text = "DNA origami self-assembly for nanomaterial synthesis"
    emb = encoder.encode(text)
    print(f"Single encode shape: {emb.shape}")  # Should be (768,)
    
    # Test batch encode
    texts = [
        "DNA origami folding optimization",
        "Machine learning for battery materials",
        "Heterogeneous catalysis for CO2 conversion"
    ]
    embs = encoder.encode_batch(texts)
    print(f"Batch encode shape: {embs.shape}")  # Should be (3, 768)
    
    # Test similarity
    from scipy.spatial.distance import cosine
    d01 = cosine(embs[0], embs[1])
    d02 = cosine(embs[0], embs[2])
    print(f"Distance DNA-Battery: {d01:.4f}")
    print(f"Distance DNA-Catalysis: {d02:.4f}")
    
    print("Test completed!")
