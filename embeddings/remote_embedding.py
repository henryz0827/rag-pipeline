"""Remote embedding service client."""
from typing import List, Optional
import requests

from .base import BaseEmbedding


class RemoteEmbedding(BaseEmbedding):
    """Remote embedding service client."""

    def __init__(
        self,
        api_url: str,
        model_name: str = "text2vec-base-multilingual",
        timeout: int = 30,
        dimension: int = 384
    ):
        """Initialize remote embedding client.

        Args:
            api_url: URL of the embedding service endpoint.
            model_name: Name of the model to use on the server.
            timeout: Request timeout in seconds.
            dimension: Dimension of the embedding vectors.
        """
        self.api_url = api_url
        self.model_name = model_name
        self.timeout = timeout
        self._dimension = dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents via remote service.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors.
        """
        response = requests.post(
            self.api_url,
            json={"texts": texts, "model": self.model_name},
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()["embeddings"]

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text.

        Args:
            text: Query text to embed.

        Returns:
            Embedding vector.
        """
        return self.embed_documents([text])[0]

    @property
    def dimension(self) -> int:
        """Return the dimension of the embedding vectors."""
        return self._dimension


class EmbeddingServiceError(Exception):
    """Exception raised when embedding service fails."""
    pass
