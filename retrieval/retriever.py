"""Retrieval module for RAG pipeline."""
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import re

from ..embeddings.base import BaseEmbedding
from ..vectorstores.base import BaseVectorStore, SearchResult


@dataclass
class RetrievalResult:
    """Result from retrieval operation."""
    content: str
    score: float
    metadata: Dict[str, Any]
    source: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata,
            "source": self.source
        }


class Retriever:
    """Retriever for finding relevant documents."""

    def __init__(
        self,
        embedding_model: BaseEmbedding,
        vector_store: BaseVectorStore,
        top_k: int = 5,
        threshold: Optional[float] = None,
        rerank: bool = False
    ):
        """Initialize retriever.

        Args:
            embedding_model: Embedding model for queries.
            vector_store: Vector store to search.
            top_k: Default number of results to return.
            threshold: Default similarity threshold.
            rerank: Whether to rerank results (future feature).
        """
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.top_k = top_k
        self.threshold = threshold
        self.rerank = rerank

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        filter_func: Optional[callable] = None
    ) -> List[RetrievalResult]:
        """Retrieve relevant documents for a query.

        Args:
            query: Search query.
            top_k: Number of results to return (overrides default).
            threshold: Similarity threshold (overrides default).
            filter_func: Optional function to filter results.

        Returns:
            List of retrieval results.
        """
        # Use defaults if not specified
        k = top_k if top_k is not None else self.top_k
        thresh = threshold if threshold is not None else self.threshold

        # Preprocess query
        processed_query = self._preprocess_query(query)

        # Get query embedding
        query_embedding = self.embedding_model.embed_query(processed_query)

        # Search vector store
        search_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=k,
            threshold=thresh
        )

        # Convert to RetrievalResult
        results = []
        for sr in search_results:
            result = RetrievalResult(
                content=sr.document.content,
                score=sr.score,
                metadata=sr.document.metadata or {},
                source=sr.document.metadata.get("source") if sr.document.metadata else None
            )
            results.append(result)

        # Apply custom filter if provided
        if filter_func:
            results = [r for r in results if filter_func(r)]

        return results

    def _preprocess_query(self, query: str) -> str:
        """Preprocess query before embedding.

        Args:
            query: Raw query string.

        Returns:
            Preprocessed query.
        """
        # Remove punctuation that might affect search
        query = re.sub(r'[,，！!？\?]', '', query)
        # Normalize whitespace
        query = ' '.join(query.split())
        return query.strip()

    def retrieve_with_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        context_window: int = 0
    ) -> List[RetrievalResult]:
        """Retrieve with additional context from surrounding chunks.

        Args:
            query: Search query.
            top_k: Number of results.
            threshold: Similarity threshold.
            context_window: Number of surrounding chunks to include.

        Returns:
            List of results with context.
        """
        results = self.retrieve(query, top_k, threshold)

        if context_window > 0:
            # This would require chunk index information in metadata
            # For now, return results as-is
            pass

        return results

    def hybrid_retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        keyword_weight: float = 0.3
    ) -> List[RetrievalResult]:
        """Hybrid retrieval combining vector search with keyword matching.

        Args:
            query: Search query.
            top_k: Number of results.
            keyword_weight: Weight for keyword matching (0-1).

        Returns:
            List of results.
        """
        # Get vector search results
        k = top_k if top_k is not None else self.top_k
        vector_results = self.retrieve(query, top_k=k * 2)

        # Simple keyword boosting
        query_terms = set(query.lower().split())

        for result in vector_results:
            content_terms = set(result.content.lower().split())
            keyword_match = len(query_terms & content_terms) / max(len(query_terms), 1)

            # Combine scores
            result.score = (1 - keyword_weight) * result.score + keyword_weight * keyword_match

        # Re-sort by combined score
        vector_results.sort(key=lambda x: x.score, reverse=True)

        return vector_results[:k]

    def batch_retrieve(
        self,
        queries: List[str],
        top_k: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> List[List[RetrievalResult]]:
        """Retrieve for multiple queries.

        Args:
            queries: List of search queries.
            top_k: Number of results per query.
            threshold: Similarity threshold.

        Returns:
            List of result lists.
        """
        return [self.retrieve(q, top_k, threshold) for q in queries]
