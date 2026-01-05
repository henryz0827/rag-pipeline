"""
RAG Pipeline - A standalone, modular RAG (Retrieval-Augmented Generation) pipeline.

This package provides components for building knowledge-based QA systems:
- Embeddings: Local and remote embedding models
- Vector Stores: FAISS (local) and Milvus (distributed)
- Document Processing: PDF, DOCX, text loading and chunking
- Retrieval: Configurable similarity search
- Prompts: Customizable prompt templates
"""

from .pipeline import RAGPipeline
from .embeddings import LocalHuggingFaceEmbedding, RemoteEmbedding
from .vectorstores import FAISSVectorStore, MilvusVectorStore, Document
from .retrieval import Retriever, RetrievalResult
from .document_processing import DocumentLoader, TextChunker
from .prompts import RAGPromptTemplate, QAPromptTemplate

__version__ = "1.0.0"

__all__ = [
    "RAGPipeline",
    "LocalHuggingFaceEmbedding",
    "RemoteEmbedding",
    "FAISSVectorStore",
    "MilvusVectorStore",
    "Document",
    "Retriever",
    "RetrievalResult",
    "DocumentLoader",
    "TextChunker",
    "RAGPromptTemplate",
    "QAPromptTemplate"
]
