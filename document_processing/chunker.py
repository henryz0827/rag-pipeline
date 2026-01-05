"""Text chunking utilities."""
from enum import Enum
from typing import List, Optional, Dict, Any
import re

from ..vectorstores.base import Document


class ChunkingStrategy(Enum):
    """Chunking strategy options."""
    FIXED_SIZE = "fixed_size"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    RECURSIVE = "recursive"


class TextChunker:
    """Text chunker with multiple strategies."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
        separators: Optional[List[str]] = None
    ):
        """Initialize text chunker.

        Args:
            chunk_size: Target chunk size in characters.
            chunk_overlap: Overlap between chunks.
            strategy: Chunking strategy to use.
            separators: Custom separators for recursive strategy.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks.

        Args:
            text: Text to chunk.

        Returns:
            List of text chunks.
        """
        if self.strategy == ChunkingStrategy.FIXED_SIZE:
            return self._fixed_size_chunk(text)
        elif self.strategy == ChunkingStrategy.SENTENCE:
            return self._sentence_chunk(text)
        elif self.strategy == ChunkingStrategy.PARAGRAPH:
            return self._paragraph_chunk(text)
        elif self.strategy == ChunkingStrategy.RECURSIVE:
            return self._recursive_chunk(text)
        else:
            return self._fixed_size_chunk(text)

    def chunk_documents(
        self,
        documents: List[Document],
        preserve_metadata: bool = True
    ) -> List[Document]:
        """Chunk multiple documents.

        Args:
            documents: List of documents to chunk.
            preserve_metadata: Whether to copy metadata to chunks.

        Returns:
            List of chunked documents.
        """
        chunked_docs = []

        for doc in documents:
            chunks = self.chunk_text(doc.content)

            for i, chunk in enumerate(chunks):
                chunk_metadata = {}
                if preserve_metadata and doc.metadata:
                    chunk_metadata = doc.metadata.copy()

                chunk_metadata["chunk_index"] = i
                chunk_metadata["total_chunks"] = len(chunks)

                chunked_docs.append(Document(
                    content=chunk,
                    metadata=chunk_metadata
                ))

        return chunked_docs

    def _fixed_size_chunk(self, text: str) -> List[str]:
        """Split by fixed character count with overlap."""
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            # Try to break at word boundary
            if end < len(text):
                # Look for last space within chunk
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - self.chunk_overlap
            if start < 0:
                start = 0

        return chunks

    def _sentence_chunk(self, text: str) -> List[str]:
        """Split by sentences, combining to reach chunk size."""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_length = len(sentence)

            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                # Keep overlap
                overlap_sentences = []
                overlap_length = 0
                for s in reversed(current_chunk):
                    if overlap_length + len(s) <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_length += len(s)
                    else:
                        break
                current_chunk = overlap_sentences
                current_length = overlap_length

            current_chunk.append(sentence)
            current_length += sentence_length

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def _paragraph_chunk(self, text: str) -> List[str]:
        """Split by paragraphs, combining to reach chunk size."""
        paragraphs = text.split('\n\n')

        chunks = []
        current_chunk = []
        current_length = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_length = len(para)

            if current_length + para_length > self.chunk_size and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_length = 0

            current_chunk.append(para)
            current_length += para_length

        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))

        return chunks

    def _recursive_chunk(self, text: str) -> List[str]:
        """Recursively split using multiple separators."""
        return self._recursive_split(text, self.separators)

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """Recursive splitting implementation."""
        if not separators:
            return self._fixed_size_chunk(text)

        separator = separators[0]
        remaining_separators = separators[1:]

        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)

        chunks = []
        current_chunk = []
        current_length = 0

        for split in splits:
            split_length = len(split)

            # If single split is too large, recursively split it
            if split_length > self.chunk_size:
                if current_chunk:
                    chunks.append(separator.join(current_chunk))
                    current_chunk = []
                    current_length = 0

                # Recursively split the large piece
                sub_chunks = self._recursive_split(split, remaining_separators)
                chunks.extend(sub_chunks)
            else:
                # Check if adding this would exceed chunk size
                new_length = current_length + split_length
                if current_chunk:
                    new_length += len(separator)

                if new_length > self.chunk_size and current_chunk:
                    chunks.append(separator.join(current_chunk))

                    # Handle overlap
                    overlap_splits = []
                    overlap_length = 0
                    for s in reversed(current_chunk):
                        if overlap_length + len(s) <= self.chunk_overlap:
                            overlap_splits.insert(0, s)
                            overlap_length += len(s)
                        else:
                            break

                    current_chunk = overlap_splits
                    current_length = overlap_length

                current_chunk.append(split)
                current_length += split_length

        if current_chunk:
            chunks.append(separator.join(current_chunk))

        return [c.strip() for c in chunks if c.strip()]


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Count tokens in text (approximate).

    Args:
        text: Text to count tokens for.
        model: Model name for tokenizer selection.

    Returns:
        Approximate token count.
    """
    try:
        import tiktoken
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except ImportError:
        # Rough approximation: ~4 chars per token
        return len(text) // 4


class TokenChunker(TextChunker):
    """Chunker that uses token counts instead of character counts."""

    def __init__(
        self,
        chunk_size: int = 256,
        chunk_overlap: int = 32,
        model: str = "gpt-3.5-turbo"
    ):
        """Initialize token-based chunker.

        Args:
            chunk_size: Target chunk size in tokens.
            chunk_overlap: Overlap between chunks in tokens.
            model: Model name for tokenizer.
        """
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strategy=ChunkingStrategy.RECURSIVE
        )
        self.model = model

        try:
            import tiktoken
            self._encoding = tiktoken.encoding_for_model(model)
            self._use_tiktoken = True
        except ImportError:
            self._use_tiktoken = False

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self._use_tiktoken:
            return len(self._encoding.encode(text))
        return len(text) // 4

    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks by token count."""
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk = []
        current_tokens = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_tokens = self._count_tokens(sentence)

            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))

                # Handle overlap
                overlap_sentences = []
                overlap_tokens = 0
                for s in reversed(current_chunk):
                    s_tokens = self._count_tokens(s)
                    if overlap_tokens + s_tokens <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_tokens += s_tokens
                    else:
                        break

                current_chunk = overlap_sentences
                current_tokens = overlap_tokens

            current_chunk.append(sentence)
            current_tokens += sentence_tokens

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks
