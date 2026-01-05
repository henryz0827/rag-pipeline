"""Base vector store interface."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class Document:
    """Document with content and metadata."""
    content: str
    metadata: Optional[Dict[str, Any]] = None
    id: Optional[str] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SearchResult:
    """Search result with document and score."""
    document: Document
    score: float


class BaseVectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    def add_documents(
        self,
        documents: List[Document],
        embeddings: Optional[List[List[float]]] = None
    ) -> List[str]:
        """Add documents to the vector store.

        Args:
            documents: List of documents to add.
            embeddings: Optional pre-computed embeddings.

        Returns:
            List of document IDs.
        """
        pass

    @abstractmethod
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
            threshold: Optional similarity threshold.

        Returns:
            List of search results.
        """
        pass

    @abstractmethod
    def delete(self, ids: List[str]) -> bool:
        """Delete documents by ID.

        Args:
            ids: List of document IDs to delete.

        Returns:
            True if successful.
        """
        pass

    def search_by_text(
        self,
        query: str,
        embedding_model,
        top_k: int = 5,
        threshold: Optional[float] = None
    ) -> List[SearchResult]:
        """Search using text query.

        Args:
            query: Text query.
            embedding_model: Embedding model to use.
            top_k: Number of results to return.
            threshold: Optional similarity threshold.

        Returns:
            List of search results.
        """
        query_embedding = embedding_model.embed_query(query)
        return self.search(query_embedding, top_k, threshold)
