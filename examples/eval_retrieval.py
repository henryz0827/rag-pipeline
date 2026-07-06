"""
Retrieval evaluation for the RAG pipeline.

Builds a small self-contained QA corpus (55 passages in deliberately
confusable topic clusters with hard-negative distractors, 25 indirect
queries with known gold passages), indexes it with the real RAGPipeline
(FAISS), and reports Hit@k / MRR@5 plus average query latency for
several embedding models.

Queries deliberately avoid the gold passage's vocabulary, and each
cluster contains near-miss distractors (Mercury vs Venus, JavaScript's
1995 release vs Java's, npm vs PyPI, black tea's caffeine vs coffee's),
so scores are meaningfully below 1.0 and differentiate models.

The corpus lives in this file so the eval is fully reproducible with no
downloads beyond the embedding models themselves:

    python examples/eval_retrieval.py

Results are printed as a Markdown table (see README "Evaluation").
"""
import importlib.util
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_rag_pipeline():
    """Import the repo as a package regardless of its directory name.

    pipeline.py uses relative imports, so it must be loaded as a package;
    this registers the repo root as `rag_pipeline` explicitly.
    """
    spec = importlib.util.spec_from_file_location(
        "rag_pipeline",
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["rag_pipeline"] = module
    spec.loader.exec_module(module)
    return module


rag = _load_rag_pipeline()

# 40 passages in confusable clusters (Python vs Java, FAISS vs Milvus,
# Mars vs Venus, coffee vs tea, ...) so retrieval is not trivially easy.
CORPUS = [
    # Python
    "Python was created by Guido van Rossum and first released in 1991. "
    "It emphasizes code readability through significant indentation.",
    "Python is dynamically typed and garbage-collected. Its global "
    "interpreter lock (GIL) prevents multiple native threads from "
    "executing Python bytecode at the same time.",
    "The Python Package Index (PyPI) hosts hundreds of thousands of "
    "third-party packages that can be installed with pip.",
    # Java
    "Java was developed by James Gosling at Sun Microsystems and released "
    "in 1995. Its slogan was 'write once, run anywhere'.",
    "Java is statically typed and compiles to bytecode that runs on the "
    "Java Virtual Machine (JVM), which uses just-in-time compilation.",
    "Maven and Gradle are the dominant build and dependency management "
    "tools in the Java ecosystem.",
    # Vector search
    "FAISS is a library developed by Meta AI for efficient similarity "
    "search and clustering of dense vectors, supporting both exact and "
    "approximate indexes.",
    "IVF (inverted file) indexes speed up vector search by partitioning "
    "the space into clusters and probing only the nearest partitions at "
    "query time.",
    "HNSW builds a multi-layer proximity graph that enables fast "
    "approximate nearest-neighbor search with high recall.",
    "Milvus is an open-source distributed vector database designed to "
    "scale similarity search across billions of vectors with replication "
    "and sharding.",
    # Embeddings / transformers
    "Sentence embeddings map variable-length text to fixed-size dense "
    "vectors such that semantically similar sentences are close in "
    "vector space.",
    "Cosine similarity measures the angle between two vectors and is "
    "commonly used to compare normalized text embeddings.",
    "The transformer architecture, introduced in the paper 'Attention Is "
    "All You Need', relies on self-attention instead of recurrence.",
    # RAG
    "Retrieval-Augmented Generation (RAG) grounds a language model's "
    "answers in documents fetched from an external knowledge base, "
    "reducing hallucinations.",
    "Chunking splits long documents into smaller overlapping pieces so "
    "that retrieval returns focused, relevant passages instead of entire "
    "files.",
    "Reranking applies a more expensive cross-encoder model to reorder an "
    "initial candidate list of retrieved passages, improving precision.",
    # Networking
    "HTTP is a stateless request-response protocol; HTTPS wraps it in TLS "
    "encryption to protect data in transit.",
    "DNS translates human-readable domain names into IP addresses using a "
    "hierarchy of authoritative name servers.",
    "TCP provides reliable, ordered delivery of a byte stream using "
    "acknowledgments and retransmissions, whereas UDP is connectionless "
    "and best-effort.",
    # Planets
    "Mars is the fourth planet from the Sun. Its red color comes from "
    "iron oxide dust, and it hosts Olympus Mons, the tallest volcano in "
    "the solar system.",
    "Venus is the second planet from the Sun and the hottest in the solar "
    "system due to a runaway greenhouse effect in its dense carbon "
    "dioxide atmosphere.",
    "Jupiter is the largest planet in the solar system; its Great Red "
    "Spot is a giant storm that has raged for centuries.",
    # Beverages
    "Coffee is brewed from roasted beans of the Coffea plant and "
    "typically contains more caffeine per cup than tea.",
    "Green tea is made from unoxidized Camellia sinensis leaves and is "
    "rich in catechin antioxidants.",
    "Espresso is a concentrated coffee brewed by forcing hot water "
    "through finely ground beans at high pressure.",
    # Biology
    "Photosynthesis converts carbon dioxide and water into glucose and "
    "oxygen using light energy captured by chlorophyll in chloroplasts.",
    "Mitochondria produce ATP through cellular respiration and are often "
    "called the powerhouse of the cell.",
    "DNA stores genetic information as sequences of four nucleotide "
    "bases: adenine, thymine, guanine, and cytosine.",
    # Machine learning
    "Overfitting occurs when a model memorizes training data noise and "
    "fails to generalize; regularization and early stopping mitigate it.",
    "Gradient descent iteratively updates model parameters in the "
    "direction that most reduces the loss function.",
    "A confusion matrix summarizes classification performance by counting "
    "true positives, false positives, true negatives, and false "
    "negatives.",
    # Databases
    "PostgreSQL is an open-source relational database supporting ACID "
    "transactions, rich SQL features, and extensions like PostGIS.",
    "Redis is an in-memory key-value store often used for caching, "
    "session storage, and message brokering.",
    "B-tree indexes keep keys sorted to allow range scans and "
    "logarithmic-time lookups in relational databases.",
    # General knowledge distractors
    "The Great Wall of China was built over centuries by successive "
    "dynasties to protect against invasions from the north.",
    "The Amazon rainforest hosts unmatched biodiversity and plays a major "
    "role in the global carbon and water cycles.",
    "The printing press, invented by Johannes Gutenberg around 1440, "
    "revolutionized the spread of information in Europe.",
    "Ludwig van Beethoven composed his Ninth Symphony while almost "
    "completely deaf; its final movement sets Schiller's 'Ode to Joy'.",
    "Impressionism emerged in 19th-century France, with painters like "
    "Monet capturing fleeting effects of light with loose brushwork.",
    "The twelve-bar blues is a chord progression that underpins much of "
    "early rock and roll.",
    # Hard negatives: near-miss distractors for the clusters above
    "JavaScript was created by Brendan Eich in ten days and released in "
    "1995 as a scripting language for the Netscape browser.",
    "Ruby was designed by Yukihiro Matsumoto to make programmers happy, "
    "blending object-oriented and functional styles.",
    "npm is the default package registry for JavaScript, hosting "
    "millions of reusable modules installed via the npm command.",
    "Anaconda is a Python distribution for data science that manages "
    "packages and environments through the conda tool.",
    "Rust's borrow checker enforces memory safety at compile time "
    "without a garbage collector.",
    "Annoy and ScaNN are alternative libraries for approximate "
    "nearest-neighbor search over high-dimensional vectors.",
    "Elasticsearch is a distributed search engine built on Lucene, "
    "widely used for full-text search and log analytics.",
    "Word2vec learns static word embeddings from co-occurrence "
    "statistics; each word gets a single vector regardless of context.",
    "The TLS handshake uses certificates and asymmetric cryptography to "
    "negotiate a shared session key between client and server.",
    "Mercury is the closest planet to the Sun, yet it is not the "
    "hottest; it has almost no atmosphere to trap heat.",
    "Mauna Loa in Hawaii is the largest active volcano on Earth, rising "
    "about nine kilometers from the ocean floor.",
    "Black tea is fully oxidized during processing and contains more "
    "caffeine than green tea, though still less than coffee.",
    "Decaffeinated coffee has most of its caffeine removed using water "
    "or solvent-based extraction before roasting.",
    "Memcached is a simple distributed in-memory cache with a flat "
    "key-value model and no persistence.",
    "Chloroplasts contain the green pigment chlorophyll and are found "
    "in plant cells but not animal cells.",
]

# (query, index of the gold passage in CORPUS)
# Queries avoid the gold passage's vocabulary where possible, and most
# have a hard-negative distractor passage in the corpus.
QUERIES = [
    ("Which language, released in the early nineties, was designed around "
     "enforced indentation for readability?", 0),
    ("What stops a multithreaded script in this interpreted language from "
     "using more than one CPU core at once?", 1),
    ("What is the main public repository of installable add-ons for the "
     "Python ecosystem?", 2),
    ("Which language did a Sun Microsystems engineer design to run "
     "unchanged on any device?", 3),
    ("How can one compiled program execute on different operating systems "
     "without recompilation?", 4),
    ("Which build tools resolve and download a Java project's "
     "dependencies?", 5),
    ("Which library from a large social media company's research lab "
     "performs fast similarity search over embeddings?", 6),
    ("How does partitioning vectors into clusters and searching only a "
     "few of them speed up retrieval?", 7),
    ("Which approach uses a layered graph of neighbors for quick "
     "approximate matching?", 8),
    ("Which purpose-built database shards and replicates embedding "
     "collections at billion scale?", 9),
    ("How can whole sentences be represented so that ones with similar "
     "meaning end up numerically close?", 10),
    ("Which measure of vector similarity depends only on direction, not "
     "magnitude?", 11),
    ("What architecture dropped recurrent networks entirely in favor of "
     "attention mechanisms?", 12),
    ("How can a chatbot cite real source material instead of inventing "
     "facts?", 13),
    ("Why break a long file into small overlapping segments before "
     "putting it in a search index?", 14),
    ("What second-stage step re-scores candidate passages with a slower, "
     "more accurate model?", 15),
    ("How is web traffic protected from eavesdropping while it travels "
     "across the network?", 16),
    ("What system converts a memorable website name into the numeric "
     "address computers use?", 17),
    ("Which transport guarantees ordered delivery, and which one just "
     "sends and hopes?", 18),
    ("Which world looks rusty red and hosts the solar system's tallest "
     "peak?", 19),
    ("Which planet is the warmest, even though another orbits closer to "
     "the Sun?", 20),
    ("What is the centuries-old storm visible on the gas giant?", 21),
    ("Which everyday hot drink delivers the strongest dose of stimulant "
     "per cup?", 22),
    ("Which beverage keeps its leaves unoxidized and is prized for its "
     "antioxidants?", 23),
    ("What brewing method forces pressurized hot water through a fine "
     "grind for a concentrated shot?", 24),
]

MODELS = [
    "sentence-transformers/paraphrase-MiniLM-L3-v2",
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-MiniLM-L12-v2",
]

TOP_K = 5


def evaluate(model_name: str, hybrid: bool = False) -> dict:
    """Index the corpus and score retrieval for one configuration."""
    pipeline = rag.RAGPipeline(
        embedding_model=model_name,
        vector_store_type="faiss",
        top_k=TOP_K,
    )
    pipeline.add_documents(
        [{"content": text, "doc_id": i} for i, text in enumerate(CORPUS)],
        chunk=False,
    )

    ranks = []
    start = time.perf_counter()
    for query, gold in QUERIES:
        if hybrid:
            results = pipeline.retriever.hybrid_retrieve(query, top_k=TOP_K)
        else:
            results = pipeline.retrieve(query, top_k=TOP_K)
        rank = next(
            (pos + 1 for pos, r in enumerate(results)
             if r.metadata.get("doc_id") == gold),
            None,
        )
        ranks.append(rank)
    latency_ms = (time.perf_counter() - start) / len(QUERIES) * 1000

    def hit_at(k: int) -> float:
        return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)

    mrr = sum(1 / r for r in ranks if r is not None) / len(ranks)

    return {
        "model": model_name.split("/")[-1] + (" (hybrid)" if hybrid else ""),
        "dim": pipeline.embedding.dimension,
        "hit1": hit_at(1),
        "hit3": hit_at(3),
        "hit5": hit_at(5),
        "mrr": mrr,
        "latency_ms": latency_ms,
    }


def main():
    rows = []
    for model in MODELS:
        print(f"Evaluating {model} ...", file=sys.stderr)
        rows.append(evaluate(model))
    # Hybrid (vector + keyword boosting) with the default model
    print("Evaluating all-MiniLM-L6-v2 with hybrid retrieval ...", file=sys.stderr)
    rows.append(evaluate("sentence-transformers/all-MiniLM-L6-v2", hybrid=True))

    print(f"\n{len(CORPUS)} passages, {len(QUERIES)} queries, top_k={TOP_K}\n")
    print("| Embedding model | Dim | Hit@1 | Hit@3 | Hit@5 | MRR@5 | ms/query |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print(
            f"| {r['model']} | {r['dim']} "
            f"| {r['hit1']:.2f} | {r['hit3']:.2f} | {r['hit5']:.2f} "
            f"| {r['mrr']:.2f} | {r['latency_ms']:.1f} |"
        )


if __name__ == "__main__":
    main()
