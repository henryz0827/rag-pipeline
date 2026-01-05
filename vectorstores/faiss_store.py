"""FAISS vector store implementation."""
import os
import pickle
from typing import List, Optional, Dict, Any
import numpy as np

from .base import BaseVectorStore, Document, SearchResult


class FAISSVectorStore(BaseVectorStore):
    """FAISS-based vector store for local/in-memory storage."""

    def __init__(
        self,
        dimension: int = 384,
        index_type: str = "flat",
        metric: str = "cosine",
        normalize: bool = True
    ):
        """Initialize FAISS vector store.

        Args:
            dimension: Dimension of embedding vectors.
            index_type: Type of FAISS index ('flat', 'ivf_flat', 'hnsw').
            metric: Distance metric ('cosine', 'l2', 'ip').
            normalize: Whether to normalize vectors before indexing.
        """
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "FAISS is required. Install with: pip install faiss-cpu"
            )

        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric
        self.normalize = normalize
        self._faiss = faiss

        # Initialize index
        self._index = self._create_index()
        self._documents: Dict[int, Document] = {}
        self._id_counter = 0

    def _create_index(self):
        """Create FAISS index based on configuration."""
        faiss = self._faiss

        if self.metric == "cosine" or self.metric == "ip":
            # Inner product (cosine similarity when normalized)
            if self.index_type == "flat":
                index = faiss.IndexFlatIP(self.dimension)
            elif self.index_type == "ivf_flat":
                quantizer = faiss.IndexFlatIP(self.dimension)
                index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
            elif self.index_type == "hnsw":
                index = faiss.IndexHNSWFlat(self.dimension, 32)
            else:
                index = faiss.IndexFlatIP(self.dimension)
        else:
            # L2 distance
            if self.index_type == "flat":
                index = faiss.IndexFlatL2(self.dimension)
            elif self.index_type == "ivf_flat":
                quantizer = faiss.IndexFlatL2(self.dimension)
                index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
            elif self.index_type == "hnsw":
                index = faiss.IndexHNSWFlat(self.dimension, 32)
            else:
                index = faiss.IndexFlatL2(self.dimension)

        return index

    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors for cosine similarity."""
        if self.normalize:
            try:
                from sklearn.preprocessing import normalize
                return normalize(vectors)
            except ImportError:
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                norms[norms == 0] = 1  # Avoid division by zero
                return vectors / norms
        return vectors

    def add_documents(
        self,
        documents: List[Document],
        embeddings: Optional[List[List[float]]] = None
    ) -> List[str]:
        """Add documents to the vector store.

        Args:
            documents: List of documents to add.
            embeddings: Pre-computed embeddings (required for FAISS).

        Returns:
            List of document IDs.
        """
        if embeddings is None:
            raise ValueError(
                "Embeddings are required for FAISSVectorStore. "
                "Compute embeddings before adding documents."
            )

        if len(documents) != len(embeddings):
            raise ValueError(
                f"Number of documents ({len(documents)}) must match "
                f"number of embeddings ({len(embeddings)})"
            )

        # Convert to numpy array
        vectors = np.array(embeddings).astype(np.float32)
        vectors = self._normalize_vectors(vectors)

        # Train index if needed (for IVF indexes)
        if hasattr(self._index, 'is_trained') and not self._index.is_trained:
            self._index.train(vectors)

        # Add vectors
        self._index.add(vectors)

        # Store documents
        ids = []
        for doc in documents:
            doc_id = str(self._id_counter)
            if doc.id:
                doc_id = doc.id
            self._documents[self._id_counter] = doc
            ids.append(doc_id)
            self._id_counter += 1

        return ids

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        threshold: Optional[float] = None
    ) -> List[SearchResult]:
        """Search for similar documents.

        Args:
            query_embedding: Query embedding vector.
            top_k: Number of results to return.
            threshold: Optional similarity threshold (0-1 for cosine).

        Returns:
            List of search results.
        """
        if self._index.ntotal == 0:
            return []

        # Prepare query vector
        query_vector = np.array([query_embedding]).astype(np.float32)
        query_vector = self._normalize_vectors(query_vector)

        # Search
        k = min(top_k, self._index.ntotal)
        distances, indices = self._index.search(query_vector, k)

        results = []
        for score, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue

            # For inner product with normalized vectors, score is cosine similarity
            if threshold is not None and score < threshold:
                continue

            if idx in self._documents:
                results.append(SearchResult(
                    document=self._documents[idx],
                    score=float(score)
                ))

        return results

    def delete(self, ids: List[str]) -> bool:
        """Delete documents by ID.

        Note: FAISS does not support direct deletion.
        This marks documents as deleted but doesn't remove from index.

        Args:
            ids: List of document IDs to delete.

        Returns:
            True if successful.
        """
        for id_str in ids:
            try:
                idx = int(id_str)
                if idx in self._documents:
                    del self._documents[idx]
            except ValueError:
                pass
        return True

    def save(self, path: str):
        """Save the vector store to disk.

        Args:
            path: Directory path to save to.
        """
        os.makedirs(path, exist_ok=True)

        # Save FAISS index
        self._faiss.write_index(self._index, os.path.join(path, "index.faiss"))

        # Save documents
        with open(os.path.join(path, "documents.pkl"), 'wb') as f:
            pickle.dump({
                'documents': self._documents,
                'id_counter': self._id_counter,
                'config': {
                    'dimension': self.dimension,
                    'index_type': self.index_type,
                    'metric': self.metric,
                    'normalize': self.normalize
                }
            }, f)

    @classmethod
    def load(cls, path: str) -> "FAISSVectorStore":
        """Load a vector store from disk.

        Args:
            path: Directory path to load from.

        Returns:
            Loaded FAISSVectorStore instance.
        """
        import faiss

        # Load documents and config
        with open(os.path.join(path, "documents.pkl"), 'rb') as f:
            data = pickle.load(f)

        config = data['config']
        store = cls(
            dimension=config['dimension'],
            index_type=config['index_type'],
            metric=config['metric'],
            normalize=config['normalize']
        )

        # Load FAISS index
        store._index = faiss.read_index(os.path.join(path, "index.faiss"))
        store._documents = data['documents']
        store._id_counter = data['id_counter']

        return store

    @property
    def count(self) -> int:
        """Return number of vectors in the index."""
        return self._index.ntotal
