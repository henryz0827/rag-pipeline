"""Local HuggingFace embeddings."""
import os
from typing import List, Optional

from .base import BaseEmbedding


class LocalHuggingFaceEmbedding(BaseEmbedding):
    """HuggingFace embeddings running locally."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        normalize_embeddings: bool = True,
        cache_folder: Optional[str] = None
    ):
        """Initialize local HuggingFace embedding model.

        Args:
            model_name: HuggingFace model name or local path.
            device: Device to run model on ('cpu', 'cuda', 'mps').
            normalize_embeddings: Whether to normalize embeddings.
            cache_folder: Folder to cache downloaded models.
        """
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        except ImportError:
            from langchain.embeddings import HuggingFaceEmbeddings

        self.model_name = model_name
        self.device = device
        self._dimension = None

        model_kwargs = {'device': device}
        encode_kwargs = {'normalize_embeddings': normalize_embeddings}

        if cache_folder:
            os.environ['TRANSFORMERS_CACHE'] = cache_folder

        self._model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors.
        """
        embeddings = self._model.embed_documents(texts)
        if self._dimension is None and embeddings:
            self._dimension = len(embeddings[0])
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text.

        Args:
            text: Query text to embed.

        Returns:
            Embedding vector.
        """
        embedding = self._model.embed_query(text)
        if self._dimension is None:
            self._dimension = len(embedding)
        return embedding

    @property
    def dimension(self) -> int:
        """Return the dimension of the embedding vectors."""
        if self._dimension is None:
            # Get dimension by embedding a test string
            test_embedding = self.embed_query("test")
            self._dimension = len(test_embedding)
        return self._dimension
