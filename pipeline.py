"""Main RAG Pipeline class."""
from typing import List, Optional, Dict, Any, Union, Generator
from pathlib import Path

from .embeddings import LocalHuggingFaceEmbedding, RemoteEmbedding, BaseEmbedding
from .vectorstores import FAISSVectorStore, MilvusVectorStore, BaseVectorStore, Document
from .document_processing import DocumentLoader, TextChunker, ChunkingStrategy
from .retrieval import Retriever, RetrievalResult
from .prompts import RAGPromptTemplate


class RAGPipeline:
    """Complete RAG pipeline for building knowledge-based QA systems."""

    def __init__(
        self,
        embedding_model: Optional[Union[str, BaseEmbedding]] = None,
        vector_store_type: str = "faiss",
        vector_store_config: Optional[Dict[str, Any]] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        top_k: int = 5,
        threshold: Optional[float] = None,
        language: str = "en"
    ):
        """Initialize RAG pipeline.

        Args:
            embedding_model: Embedding model name or instance.
            vector_store_type: Type of vector store ('faiss' or 'milvus').
            vector_store_config: Configuration for vector store.
            chunk_size: Size of text chunks.
            chunk_overlap: Overlap between chunks.
            top_k: Default number of results to retrieve.
            threshold: Default similarity threshold.
            language: Language for prompts ('en' or 'cn').
        """
        # Initialize embedding model
        if embedding_model is None:
            self.embedding = LocalHuggingFaceEmbedding()
        elif isinstance(embedding_model, str):
            self.embedding = LocalHuggingFaceEmbedding(model_name=embedding_model)
        else:
            self.embedding = embedding_model

        # Initialize vector store
        vector_store_config = vector_store_config or {}
        dimension = self.embedding.dimension

        if vector_store_type == "faiss":
            self.vector_store = FAISSVectorStore(
                dimension=dimension,
                **vector_store_config
            )
        elif vector_store_type == "milvus":
            self.vector_store = MilvusVectorStore(
                dimension=dimension,
                **vector_store_config
            )
        else:
            raise ValueError(f"Unknown vector store type: {vector_store_type}")

        # Initialize components
        self.chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strategy=ChunkingStrategy.RECURSIVE
        )

        self.retriever = Retriever(
            embedding_model=self.embedding,
            vector_store=self.vector_store,
            top_k=top_k,
            threshold=threshold
        )

        self.prompt_template = RAGPromptTemplate(language=language)
        self.document_loader = DocumentLoader()

        self.top_k = top_k
        self.threshold = threshold

    def add_documents(
        self,
        documents: Union[List[str], List[Document], List[Dict[str, Any]]],
        metadata: Optional[Dict[str, Any]] = None,
        chunk: bool = True
    ) -> List[str]:
        """Add documents to the pipeline.

        Args:
            documents: List of text strings, Document objects, or dicts.
            metadata: Additional metadata for all documents.
            chunk: Whether to chunk documents before adding.

        Returns:
            List of document IDs.
        """
        # Convert to Document objects
        docs = []
        for doc in documents:
            if isinstance(doc, str):
                doc_metadata = metadata.copy() if metadata else {}
                docs.append(Document(content=doc, metadata=doc_metadata))
            elif isinstance(doc, dict):
                content = doc.get('content', doc.get('text', ''))
                doc_metadata = {k: v for k, v in doc.items() if k not in {'content', 'text'}}
                if metadata:
                    doc_metadata.update(metadata)
                docs.append(Document(content=content, metadata=doc_metadata))
            elif isinstance(doc, Document):
                if metadata:
                    doc.metadata = {**doc.metadata, **metadata} if doc.metadata else metadata
                docs.append(doc)

        # Chunk if requested
        if chunk:
            docs = self.chunker.chunk_documents(docs)

        # Generate embeddings
        texts = [doc.content for doc in docs]
        embeddings = self.embedding.embed_documents(texts)

        # Add to vector store
        ids = self.vector_store.add_documents(docs, embeddings)

        return ids

    def add_files(
        self,
        paths: Union[str, Path, List[Union[str, Path]]],
        metadata: Optional[Dict[str, Any]] = None,
        chunk: bool = True
    ) -> List[str]:
        """Add documents from files.

        Args:
            paths: File or directory path(s).
            metadata: Additional metadata.
            chunk: Whether to chunk documents.

        Returns:
            List of document IDs.
        """
        if isinstance(paths, (str, Path)):
            paths = [paths]

        all_docs = []
        for path in paths:
            docs = self.document_loader.load(path, metadata)
            all_docs.extend(docs)

        return self.add_documents(all_docs, chunk=chunk)

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> List[RetrievalResult]:
        """Retrieve relevant documents for a query.

        Args:
            query: Search query.
            top_k: Number of results.
            threshold: Similarity threshold.

        Returns:
            List of retrieval results.
        """
        return self.retriever.retrieve(
            query=query,
            top_k=top_k or self.top_k,
            threshold=threshold or self.threshold
        )

    def build_prompt(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        custom_template: Optional[str] = None
    ) -> str:
        """Build a prompt with retrieved context.

        Args:
            query: User question.
            top_k: Number of documents to retrieve.
            threshold: Similarity threshold.
            custom_template: Optional custom prompt template.

        Returns:
            Formatted prompt string.
        """
        # Retrieve relevant documents
        results = self.retrieve(query, top_k, threshold)

        # Use custom template if provided
        if custom_template:
            template = RAGPromptTemplate(template=custom_template)
        else:
            template = self.prompt_template

        # Format prompt
        return template.format_with_results(query, results)

    def generate(
        self,
        query: str,
        llm_client: Any = None,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        stream: bool = False,
        **llm_kwargs
    ) -> Union[str, Generator[str, None, None]]:
        """Generate a response using retrieved context and an LLM.

        Args:
            query: User question.
            llm_client: LLM client with generate/chat method.
            top_k: Number of documents to retrieve.
            threshold: Similarity threshold.
            stream: Whether to stream the response.
            **llm_kwargs: Additional arguments for the LLM.

        Returns:
            Generated response or generator for streaming.
        """
        if llm_client is None:
            raise ValueError(
                "LLM client is required for generation. "
                "Use build_prompt() to get the prompt without LLM."
            )

        # Build prompt
        prompt = self.build_prompt(query, top_k, threshold)

        # Generate response
        if stream:
            return self._stream_generate(llm_client, prompt, **llm_kwargs)
        else:
            return self._generate(llm_client, prompt, **llm_kwargs)

    def _generate(self, llm_client: Any, prompt: str, **kwargs) -> str:
        """Generate response from LLM."""
        # Try different common LLM client interfaces
        if hasattr(llm_client, 'generate'):
            return llm_client.generate(prompt, **kwargs)
        elif hasattr(llm_client, 'chat'):
            return llm_client.chat(prompt, **kwargs)
        elif hasattr(llm_client, 'complete'):
            return llm_client.complete(prompt, **kwargs)
        elif hasattr(llm_client, '__call__'):
            return llm_client(prompt, **kwargs)
        else:
            raise ValueError(
                "LLM client must have generate(), chat(), complete(), or __call__() method"
            )

    def _stream_generate(
        self,
        llm_client: Any,
        prompt: str,
        **kwargs
    ) -> Generator[str, None, None]:
        """Stream response from LLM."""
        if hasattr(llm_client, 'stream'):
            for chunk in llm_client.stream(prompt, **kwargs):
                yield chunk
        elif hasattr(llm_client, 'chat_stream'):
            for chunk in llm_client.chat_stream(prompt, **kwargs):
                yield chunk
        else:
            # Fallback to non-streaming
            yield self._generate(llm_client, prompt, **kwargs)

    def query(
        self,
        query: str,
        return_sources: bool = True,
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """Simple query interface returning context and prompt.

        Args:
            query: User question.
            return_sources: Whether to include source information.
            top_k: Number of results.

        Returns:
            Dictionary with prompt, context, and optionally sources.
        """
        results = self.retrieve(query, top_k)
        prompt = self.prompt_template.format_with_results(query, results)

        response = {
            "query": query,
            "prompt": prompt,
            "context": [r.content for r in results]
        }

        if return_sources:
            response["sources"] = [
                {
                    "content": r.content,
                    "score": r.score,
                    "metadata": r.metadata,
                    "source": r.source
                }
                for r in results
            ]

        return response

    def save(self, path: Union[str, Path]):
        """Save the pipeline to disk.

        Args:
            path: Directory to save to.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save vector store if FAISS
        if isinstance(self.vector_store, FAISSVectorStore):
            self.vector_store.save(str(path / "vector_store"))

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        embedding_model: Optional[Union[str, BaseEmbedding]] = None
    ) -> "RAGPipeline":
        """Load a pipeline from disk.

        Args:
            path: Directory to load from.
            embedding_model: Embedding model to use.

        Returns:
            Loaded RAGPipeline instance.
        """
        path = Path(path)

        # Create pipeline
        pipeline = cls(embedding_model=embedding_model, vector_store_type="faiss")

        # Load vector store
        pipeline.vector_store = FAISSVectorStore.load(str(path / "vector_store"))

        # Update retriever
        pipeline.retriever = Retriever(
            embedding_model=pipeline.embedding,
            vector_store=pipeline.vector_store,
            top_k=pipeline.top_k,
            threshold=pipeline.threshold
        )

        return pipeline

    @property
    def document_count(self) -> int:
        """Return number of documents in the pipeline."""
        return self.vector_store.count
