"""
Example using RAG Pipeline with OpenAI.

This example shows how to integrate with OpenAI's API for generation.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import RAGPipeline


class OpenAIClient:
    """Simple OpenAI client wrapper."""

    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI package required. Install with: pip install openai")

        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response from OpenAI."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response.choices[0].message.content

    def stream(self, prompt: str, **kwargs):
        """Stream response from OpenAI."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            **kwargs
        )
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


def main():
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        return

    # Initialize pipeline
    pipeline = RAGPipeline(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        vector_store_type="faiss",
        top_k=3
    )

    # Add documents
    documents = [
        "The RAG (Retrieval-Augmented Generation) pattern combines information retrieval with text generation.",
        "Vector databases like FAISS and Milvus are used to store and search embeddings efficiently.",
        "Embeddings are dense vector representations of text that capture semantic meaning.",
        "LLMs (Large Language Models) can generate more accurate responses when provided with relevant context.",
        "Chunking strategies help break large documents into manageable pieces for retrieval."
    ]

    pipeline.add_documents(documents, chunk=False)
    print(f"Added {pipeline.document_count} documents to the pipeline")

    # Initialize OpenAI client
    llm = OpenAIClient(model="gpt-3.5-turbo")

    # Example queries
    queries = [
        "What is RAG and how does it work?",
        "How are embeddings used in RAG systems?",
        "Why is chunking important?"
    ]

    for query in queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print("=" * 50)

        # Non-streaming response
        response = pipeline.generate(query, llm_client=llm)
        print(f"\nResponse:\n{response}")


def streaming_example():
    """Example with streaming response."""
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        return

    pipeline = RAGPipeline(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        vector_store_type="faiss"
    )

    documents = [
        "Python was created by Guido van Rossum and released in 1991.",
        "Python emphasizes code readability with its notable use of significant indentation.",
        "Python is dynamically typed and garbage-collected."
    ]
    pipeline.add_documents(documents, chunk=False)

    llm = OpenAIClient()

    query = "Tell me about Python programming language"
    print(f"Query: {query}\n")
    print("Streaming response:")

    for chunk in pipeline.generate(query, llm_client=llm, stream=True):
        print(chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
