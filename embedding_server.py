"""
Embedding Server - Run embeddings as a REST service.

Usage:
    python embedding_server.py --model sentence-transformers/all-MiniLM-L6-v2 --port 5001
"""
import argparse
import os
from typing import List, Optional

from flask import Flask, request, jsonify


app = Flask(__name__)

# Global model storage
models = {}


def get_embedding_model(model_path: str, device: str = "cpu"):
    """Load and cache embedding model."""
    if model_path in models:
        return models[model_path]

    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError:
        from langchain.embeddings import HuggingFaceEmbeddings

    model = HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )

    models[model_path] = model
    return model


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "models": list(models.keys())}), 200


@app.route('/embed', methods=['POST'])
def embed_text():
    """Embed texts endpoint.

    Request body:
    {
        "texts": ["text1", "text2", ...],
        "model": "model_name"  # optional, uses default if not specified
    }

    Response:
    {
        "embeddings": [[...], [...], ...],
        "model": "model_name",
        "dimension": 384
    }
    """
    try:
        data = request.get_json()
        texts = data.get('texts')
        model_name = data.get('model', app.config.get('DEFAULT_MODEL'))

        if not texts or not isinstance(texts, list):
            return jsonify({"error": "Please provide a list of texts"}), 400

        if not model_name:
            return jsonify({"error": "Please provide a model name"}), 400

        # Get or load model
        device = app.config.get('DEVICE', 'cpu')
        model = get_embedding_model(model_name, device)

        # Generate embeddings
        embeddings = model.embed_documents(texts)

        return jsonify({
            "embeddings": embeddings,
            "model": model_name,
            "dimension": len(embeddings[0]) if embeddings else 0
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/embed_query', methods=['POST'])
def embed_query():
    """Embed single query endpoint.

    Request body:
    {
        "text": "query text",
        "model": "model_name"  # optional
    }

    Response:
    {
        "embedding": [...],
        "model": "model_name"
    }
    """
    try:
        data = request.get_json()
        text = data.get('text')
        model_name = data.get('model', app.config.get('DEFAULT_MODEL'))

        if not text:
            return jsonify({"error": "Please provide text"}), 400

        if not model_name:
            return jsonify({"error": "Please provide a model name"}), 400

        # Get or load model
        device = app.config.get('DEVICE', 'cpu')
        model = get_embedding_model(model_name, device)

        # Generate embedding
        embedding = model.embed_query(text)

        return jsonify({
            "embedding": embedding,
            "model": model_name
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def main():
    parser = argparse.ArgumentParser(description="Embedding Server")
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Default embedding model"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5001,
        help="Server port"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to run model on"
    )
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Preload the default model on startup"
    )

    args = parser.parse_args()

    app.config['DEFAULT_MODEL'] = args.model
    app.config['DEVICE'] = args.device

    if args.preload:
        print(f"Preloading model: {args.model}")
        get_embedding_model(args.model, args.device)
        print("Model loaded successfully")

    print(f"Starting embedding server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port)


if __name__ == '__main__':
    main()
