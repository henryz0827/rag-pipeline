# Local RAG Pipeline

A standalone, modular RAG (Retrieval-Augmented Generation) pipeline for building knowledge-based QA systems.

## Features

- **Multiple Embedding Options**: Local HuggingFace embeddings or remote embedding service
- **Multiple Vector Stores**: FAISS (local/in-memory) and Milvus (distributed)
- **Document Processing**: PDF, DOCX, and text file processing with chunking
- **Flexible Retrieval**: Configurable similarity search with threshold filtering
- **Prompt Templates**: Customizable prompt templates for different use cases
- **Streaming Support**: Stream responses from LLM

## Project Structure

```
rag_pipeline/
├── config/
│   └── config.yaml          # Configuration file
├── embeddings/
│   ├── __init__.py
│   ├── base.py              # Base embedding interface
│   ├── local_embedding.py   # Local HuggingFace embeddings
│   └── remote_embedding.py  # Remote embedding service client
├── vectorstores/
│   ├── __init__.py
│   ├── base.py              # Base vector store interface
│   ├── faiss_store.py       # FAISS vector store
│   └── milvus_store.py      # Milvus vector store
├── document_processing/
│   ├── __init__.py
│   ├── loader.py            # Document loaders (PDF, DOCX, TXT)
│   └── chunker.py           # Text chunking utilities
├── retrieval/
│   ├── __init__.py
│   └── retriever.py         # Retrieval logic
├── prompts/
│   ├── __init__.py
│   └── templates.py         # Prompt templates
├── pipeline.py              # Main RAG pipeline class
├── embedding_server.py      # Optional: Run embedding as a service
├── requirements.txt         # Python dependencies
└── examples/
    └── example_usage.py     # Usage examples
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from pipeline import RAGPipeline

# Initialize pipeline with FAISS
pipeline = RAGPipeline(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    vector_store_type="faiss"
)

# Add documents
pipeline.add_documents([
    "Document 1 content...",
    "Document 2 content..."
])

# Retrieve relevant chunks
results = pipeline.retrieve("Your question here", top_k=3)

# Option A: build the RAG prompt and call any LLM yourself
prompt = pipeline.build_prompt("Your question here", top_k=3)

# Option B: let the pipeline call the LLM. There is no built-in LLM --
# pass a client object exposing generate(), chat(), complete(), or __call__()
# (and stream() / chat_stream() if you use stream=True):
class MyLLMClient:
    def generate(self, prompt: str, **kwargs) -> str:
        ...  # call your LLM API here and return the response text

response = pipeline.generate(
    "Your question here",
    llm_client=MyLLMClient(),
    top_k=3
)
```

See [examples/example_openai.py](examples/example_openai.py) for a complete
client wrapping the OpenAI API, including streaming.

## Configuration

Edit `config/config.yaml` to customize:
- Embedding model path
- Vector store settings
- Chunking parameters
- Retrieval thresholds

## License

MIT License
