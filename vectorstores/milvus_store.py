"""Milvus vector store implementation."""
from typing import List, Optional, Dict, Any
import json

from .base import BaseVectorStore, Document, SearchResult


class MilvusVectorStore(BaseVectorStore):
    """Milvus-based distributed vector store."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 19530,
        collection_name: str = "documents",
        dimension: int = 384,
        user: Optional[str] = None,
        password: Optional[str] = None,
        db_name: str = "default",
        metric_type: str = "COSINE",
        index_type: str = "IVF_FLAT"
    ):
        """Initialize Milvus vector store.

        Args:
            host: Milvus server host.
            port: Milvus server port.
            collection_name: Name of the collection.
            dimension: Dimension of embedding vectors.
            user: Username for authentication.
            password: Password for authentication.
            db_name: Database name.
            metric_type: Metric type ('COSINE', 'L2', 'IP').
            index_type: Index type ('IVF_FLAT', 'HNSW', 'FLAT').
        """
        try:
            from pymilvus import MilvusClient, DataType
        except ImportError:
            raise ImportError(
                "pymilvus is required. Install with: pip install pymilvus"
            )

        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.dimension = dimension
        self.db_name = db_name
        self.metric_type = metric_type
        self.index_type = index_type
        self._DataType = DataType

        # Build connection URI
        uri = f"http://{host}:{port}"

        # Initialize client
        if user and password:
            self._client = MilvusClient(
                uri=uri,
                token=f"{user}:{password}",
                db_name=db_name
            )
        else:
            self._client = MilvusClient(uri=uri, db_name=db_name)

        # Create collection if not exists
        self._ensure_collection()

    def _ensure_collection(self):
        """Ensure collection exists with proper schema."""
        from pymilvus import DataType

        if self._client.has_collection(self.collection_name):
            return

        # Create schema
        schema = self._client.create_schema(
            auto_id=True,
            enable_dynamic_field=True
        )

        schema.add_field(
            field_name="id",
            datatype=DataType.INT64,
            is_primary=True
        )
        schema.add_field(
            field_name="embedding",
            datatype=DataType.FLOAT_VECTOR,
            dim=self.dimension
        )
        schema.add_field(
            field_name="content",
            datatype=DataType.VARCHAR,
            max_length=65535
        )
        schema.add_field(
            field_name="metadata",
            datatype=DataType.JSON
        )

        # Index params
        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type=self.index_type,
            metric_type=self.metric_type,
            params={"nlist": 1024} if self.index_type == "IVF_FLAT" else {}
        )

        # Create collection
        self._client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params
        )

    def add_documents(
        self,
        documents: List[Document],
        embeddings: Optional[List[List[float]]] = None
    ) -> List[str]:
        """Add documents to the vector store.

        Args:
            documents: List of documents to add.
            embeddings: Pre-computed embeddings (required).

        Returns:
            List of document IDs.
        """
        if embeddings is None:
            raise ValueError(
                "Embeddings are required for MilvusVectorStore. "
                "Compute embeddings before adding documents."
            )

        if len(documents) != len(embeddings):
            raise ValueError(
                f"Number of documents ({len(documents)}) must match "
                f"number of embeddings ({len(embeddings)})"
            )

        # Prepare data
        data = []
        for doc, emb in zip(documents, embeddings):
            data.append({
                "embedding": emb,
                "content": doc.content,
                "metadata": doc.metadata or {}
            })

        # Insert
        result = self._client.insert(
            collection_name=self.collection_name,
            data=data
        )

        # Return IDs
        ids = result.get("ids", [])
        return [str(id) for id in ids]

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
        results = self._client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            limit=top_k,
            output_fields=["content", "metadata"]
        )

        search_results = []
        for hits in results:
            for hit in hits:
                score = hit.get("distance", 0)

                # Apply threshold
                if threshold is not None:
                    if self.metric_type == "COSINE" and score < threshold:
                        continue
                    elif self.metric_type == "L2" and score > threshold:
                        continue

                entity = hit.get("entity", {})
                doc = Document(
                    content=entity.get("content", ""),
                    metadata=entity.get("metadata", {}),
                    id=str(hit.get("id", ""))
                )

                search_results.append(SearchResult(
                    document=doc,
                    score=score
                ))

        return search_results

    def delete(self, ids: List[str]) -> bool:
        """Delete documents by ID.

        Args:
            ids: List of document IDs to delete.

        Returns:
            True if successful.
        """
        int_ids = [int(id) for id in ids]
        self._client.delete(
            collection_name=self.collection_name,
            ids=int_ids
        )
        return True

    def drop_collection(self):
        """Drop the entire collection."""
        if self._client.has_collection(self.collection_name):
            self._client.drop_collection(self.collection_name)

    @property
    def count(self) -> int:
        """Return number of entities in the collection."""
        stats = self._client.get_collection_stats(self.collection_name)
        return stats.get("row_count", 0)
