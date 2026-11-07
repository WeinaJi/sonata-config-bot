"""
RAG module — Retrieval-Augmented Generation for SONATA spec context.

Loads the SONATA simulation config spec, chunks it, embeds it locally with
FastEmbed (ONNX, no API calls), stores in FAISS, and exposes a `retrieve(query)`
function that returns the most relevant spec chunks for a given query.

First call downloads the embedding model (~65 MB, cached) and builds the index
in a few seconds. Subsequent calls load the index from disk.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SPEC_URL = "https://sonata-extension.readthedocs.io/en/latest/sonata_simulation.html"
SPEC_LOCAL_FALLBACK = Path(__file__).parent / "spec_cache.txt"
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
# One index per embedding model — vectors from different models are not compatible
FAISS_INDEX_DIR = Path(__file__).parent / "faiss_index" / EMBEDDING_MODEL.replace("/", "__")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 4  # number of chunks to retrieve per query

# ---------------------------------------------------------------------------
# Module-level state (lazy init)
# ---------------------------------------------------------------------------

_store: FAISS | None = None
_embeddings: FastEmbedEmbeddings | None = None


def _get_embeddings() -> FastEmbedEmbeddings:
    global _embeddings  # noqa: PLW0603
    if _embeddings is None:
        logger.info("Using local embedding model: %s", EMBEDDING_MODEL)
        # cache_dir=None uses FastEmbed's default; set FASTEMBED_CACHE_DIR to keep the model somewhere persistent
        _embeddings = FastEmbedEmbeddings(
            model_name=EMBEDDING_MODEL,
            cache_dir=os.environ.get("FASTEMBED_CACHE_DIR"),
        )
    return _embeddings


def _load_spec_text() -> str:
    """Load the SONATA spec from the web, or fall back to a local cache."""
    # Try local cache first (faster, works offline)
    if SPEC_LOCAL_FALLBACK.exists():
        logger.info("Loading spec from local cache: %s", SPEC_LOCAL_FALLBACK)
        return SPEC_LOCAL_FALLBACK.read_text(encoding="utf-8")

    # Fetch from the web
    logger.info("Fetching spec from: %s", SPEC_URL)
    loader = WebBaseLoader(SPEC_URL)
    docs = loader.load()
    text = "\n\n".join(doc.page_content for doc in docs)

    # Cache locally for next time
    SPEC_LOCAL_FALLBACK.write_text(text, encoding="utf-8")
    logger.info("Spec cached to: %s", SPEC_LOCAL_FALLBACK)

    return text


def _build_index() -> FAISS:
    """Build the FAISS vector store from spec chunks."""
    text = _load_spec_text()

    # Also include the Pydantic model JSON schema as extra context
    from data_model import SimulationConfig

    schema_text = "SONATA SimulationConfig JSON Schema:\n" + repr(
        SimulationConfig.model_json_schema(mode="serialization"),
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.create_documents([text, schema_text])
    logger.info("Built %d chunks from spec + schema", len(chunks))

    # Local embedding: no rate limits, so embed all chunks in one go
    store = FAISS.from_documents(chunks, _get_embeddings())

    # Save to disk for next time
    store.save_local(str(FAISS_INDEX_DIR))
    logger.info("FAISS index saved to: %s", FAISS_INDEX_DIR)

    return store


def _get_store() -> FAISS:
    global _store  # noqa: PLW0603
    if _store is None:
        # Try loading from disk first (check the file, not the dir — a mounted volume creates an empty dir)
        if (FAISS_INDEX_DIR / "index.faiss").exists():
            logger.info("Loading FAISS index from disk: %s", FAISS_INDEX_DIR)
            embeddings = _get_embeddings()
            _store = FAISS.load_local(
                str(FAISS_INDEX_DIR),
                embeddings,
                allow_dangerous_deserialization=True,
            )
        else:
            _store = _build_index()
    return _store


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def retrieve(query: str, top_k: int = TOP_K) -> str:
    """
    Retrieve the most relevant spec chunks for a given query.

    Returns a single string with the concatenated chunks, ready to inject
    into the LLM prompt as context.
    """
    store = _get_store()
    docs = store.similarity_search(query, k=top_k)
    return "\n\n---\n\n".join(doc.page_content for doc in docs)
