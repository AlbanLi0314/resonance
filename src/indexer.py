"""
FAISS Index Manager
Handles building, saving, loading, and searching the vector index
"""
import numpy as np
from pathlib import Path
from typing import Tuple, List


class FaissIndexer:
    """
    Wrapper for FAISS index operations
    
    Usage:
        indexer = FaissIndexer(dimension=768)
        indexer.build(embeddings)
        indexer.save("path/to/index.faiss")
        
        indexer.load("path/to/index.faiss")
        distances, indices = indexer.search(query_embedding, top_k=10)
    """
    
    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self.index = None
        
    def build(self, embeddings: np.ndarray):
        """
        Build FAISS index from embeddings
        
        Args:
            embeddings: numpy array of shape (n_vectors, dimension)
        """
        import faiss
        
        # Ensure float32
        embeddings = embeddings.astype('float32')
        
        n_vectors, dim = embeddings.shape
        assert dim == self.dimension, f"Dimension mismatch: {dim} vs {self.dimension}"
        
        print(f"Building FAISS index with {n_vectors} vectors of dimension {dim}")
        
        # Use simple L2 distance index (exact search)
        # For <1000 vectors, this is fast enough
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings)
        
        print(f"Index built successfully! Total vectors: {self.index.ntotal}")
        return self
    
    def save(self, path: str):
        """Save index to file"""
        import faiss
        
        if self.index is None:
            raise RuntimeError("No index to save. Call build() first.")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(self.index, str(path))
        print(f"Index saved to: {path}")
        
    def load(self, path: str):
        """Load index from file"""
        import faiss
        
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Index file not found: {path}")
        
        self.index = faiss.read_index(str(path))
        print(f"Index loaded from: {path}")
        print(f"Total vectors in index: {self.index.ntotal}")
        return self
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar vectors
        
        Args:
            query_embedding: numpy array of shape (dimension,) or (1, dimension)
            top_k: number of results to return
            
        Returns:
            distances: numpy array of shape (top_k,)
            indices: numpy array of shape (top_k,)
        """
        if self.index is None:
            raise RuntimeError("No index loaded. Call build() or load() first.")
        
        # Reshape if needed
        query = query_embedding.astype('float32')
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        # Search
        distances, indices = self.index.search(query, top_k)
        
        return distances[0], indices[0]
    
    @property
    def total_vectors(self) -> int:
        """Return total number of vectors in index"""
        if self.index is None:
            return 0
        return self.index.ntotal


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    print("Testing FaissIndexer...")
    
    # Create fake embeddings
    n_vectors = 100
    dimension = 768
    embeddings = np.random.randn(n_vectors, dimension).astype('float32')
    
    # Build index
    indexer = FaissIndexer(dimension=dimension)
    indexer.build(embeddings)
    
    # Test search
    query = np.random.randn(dimension).astype('float32')
    distances, indices = indexer.search(query, top_k=5)
    
    print(f"Query results:")
    for i, (dist, idx) in enumerate(zip(distances, indices)):
        print(f"  #{i+1}: index={idx}, distance={dist:.4f}")
    
    # Test save/load
    test_path = "/tmp/test_faiss.index"
    indexer.save(test_path)
    
    new_indexer = FaissIndexer(dimension=dimension)
    new_indexer.load(test_path)
    
    # Verify same results
    distances2, indices2 = new_indexer.search(query, top_k=5)
    assert np.allclose(distances, distances2), "Results mismatch!"
    assert np.array_equal(indices, indices2), "Indices mismatch!"
    
    print("Test completed!")
