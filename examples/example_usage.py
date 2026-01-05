"""
Example usage of the RAG Pipeline.

This script demonstrates various ways to use the RAG pipeline.
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import RAGPipeline
from embeddings import LocalHuggingFaceEmbedding, RemoteEmbedding
from vectorstores import FAISSVectorStore, Document
from document_processing import DocumentLoader, TextChunker
from prompts import RAGPromptTemplate


def example_basic_usage():
    """Basic usage example."""
    print("=" * 50)
    print("Example 1: Basic Usage")
    print("=" * 50)

    # Initialize pipeline with default settings
    pipeline = RAGPipeline(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        vector_store_type="faiss",
        chunk_size=300,
        chunk_overlap=30,
        top_k=3
    )

    # Add some documents
    documents = [
        "Python is a high-level programming language known for its simplicity and readability.",
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "RAG (Retrieval-Augmented Generation) combines retrieval and generation for better AI responses.",
        "FAISS is a library for efficient similarity search developed by Facebook AI Research.",
        "Vector embeddings are numerical representations of text that capture semantic meaning."
    ]

    pipeline.add_documents(documents, chunk=False)
    print(f"Added {pipeline.document_count} documents")

    # Query the pipeline
    query = "What is RAG?"
    results = pipeline.retrieve(query)

    print(f"\nQuery: {query}")
    print("\nResults:")
    for i, result in enumerate(results):
        print(f"  {i+1}. (score: {result.score:.4f}) {result.content[:100]}...")

    # Build prompt for LLM
    prompt = pipeline.build_prompt(query)
    print(f"\nGenerated prompt:\n{prompt[:500]}...")


def example_with_files():
    """Example loading documents from files."""
    print("\n" + "=" * 50)
    print("Example 2: Loading from Files")
    print("=" * 50)

    pipeline = RAGPipeline(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        vector_store_type="faiss"
    )

    # Note: Replace with actual file paths
    # pipeline.add_files("path/to/documents/")
    # pipeline.add_files(["file1.pdf", "file2.txt"])

    print("File loading example - replace paths with actual files")


def example_with_milvus():
    """Example using Milvus vector store."""
    print("\n" + "=" * 50)
    print("Example 3: Using Milvus")
    print("=" * 50)

    # Note: Requires running Milvus server
    # pipeline = RAGPipeline(
    #     embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    #     vector_store_type="milvus",
    #     vector_store_config={
    #         "host": "localhost",
    #         "port": 19530,
    #         "collection_name": "my_documents"
    #     }
    # )

    print("Milvus example - requires running Milvus server")


def example_custom_components():
    """Example with custom components."""
    print("\n" + "=" * 50)
    print("Example 4: Custom Components")
    print("=" * 50)

    # Custom embedding model
    embedding = LocalHuggingFaceEmbedding(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device="cpu",
        normalize_embeddings=True
    )

    # Custom vector store
    vector_store = FAISSVectorStore(
        dimension=embedding.dimension,
        index_type="flat",
        metric="cosine"
    )

    # Custom chunker
    chunker = TextChunker(
        chunk_size=200,
        chunk_overlap=20
    )

    # Use in pipeline
    pipeline = RAGPipeline(
        embedding_model=embedding,
        vector_store_type="faiss",
        chunk_size=200,
        chunk_overlap=20
    )

    print("Created pipeline with custom components")


def example_chinese_support():
    """Example with Chinese language support."""
    print("\n" + "=" * 50)
    print("Example 5: Chinese Language Support")
    print("=" * 50)

    pipeline = RAGPipeline(
        embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        vector_store_type="faiss",
        language="cn"  # Use Chinese prompts
    )

    # Chinese documents
    documents = [
        "Python是一种高级编程语言,以其简洁和易读性而闻名。",
        "机器学习是人工智能的一个子集,使系统能够从数据中学习。",
        "RAG(检索增强生成)结合了检索和生成,以提供更好的AI响应。"
    ]

    pipeline.add_documents(documents, chunk=False)
    print(f"Added {pipeline.document_count} Chinese documents")

    # Query in Chinese
    query = "什么是RAG?"
    prompt = pipeline.build_prompt(query)
    print(f"\nQuery: {query}")
    print(f"\nGenerated prompt:\n{prompt}")


def example_save_and_load():
    """Example saving and loading pipeline."""
    print("\n" + "=" * 50)
    print("Example 6: Save and Load")
    print("=" * 50)

    # Create and populate pipeline
    pipeline = RAGPipeline(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        vector_store_type="faiss"
    )

    documents = [
        "Document 1: Information about topic A.",
        "Document 2: Information about topic B.",
        "Document 3: Information about topic C."
    ]
    pipeline.add_documents(documents, chunk=False)

    # Save pipeline
    save_path = "./saved_pipeline"
    pipeline.save(save_path)
    print(f"Pipeline saved to {save_path}")

    # Load pipeline
    loaded_pipeline = RAGPipeline.load(
        save_path,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2"
    )
    print(f"Pipeline loaded with {loaded_pipeline.document_count} documents")


def example_with_llm():
    """Example with LLM integration (mock)."""
    print("\n" + "=" * 50)
    print("Example 7: LLM Integration")
    print("=" * 50)

    # Mock LLM client
    class MockLLM:
        def generate(self, prompt, **kwargs):
            return f"[Mock LLM Response based on prompt length: {len(prompt)} chars]"

    pipeline = RAGPipeline(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        vector_store_type="faiss"
    )

    documents = [
        "The capital of France is Paris.",
        "The Eiffel Tower is located in Paris.",
        "Paris is known as the City of Light."
    ]
    pipeline.add_documents(documents, chunk=False)

    # Generate with mock LLM
    llm = MockLLM()
    response = pipeline.generate(
        query="What is the capital of France?",
        llm_client=llm
    )
    print(f"LLM Response: {response}")


def main():
    """Run all examples."""
    example_basic_usage()
    example_with_files()
    example_with_milvus()
    example_custom_components()
    example_chinese_support()
    example_save_and_load()
    example_with_llm()

    print("\n" + "=" * 50)
    print("All examples completed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
