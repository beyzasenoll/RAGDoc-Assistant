"""
RAG-based operational document assistant — FastAPI service.

Endpoints:
  GET  /health
  POST /ask

LLM and embeddings: local Ollama.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.errors import InvalidCollectionException
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ollama_utils import ollama_chat, ollama_chat_model, ollama_host, ollama_reachable


DEFAULT_CHROMA_PATH = Path(__file__).resolve().parent / "chroma_db"
DEFAULT_COLLECTION = "operational_docs"
DEFAULT_TOP_K = 4


class AskRequest(BaseModel):
    """Request body for a question."""

    question: str = Field(..., min_length=1, description="User question")


class SourceInfo(BaseModel):
    """Source summary."""

    source: str
    page: int


class AskResponse(BaseModel):
    """API response."""

    answer: str
    sources: list[SourceInfo]


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    return Path(raw).expanduser().resolve() if raw else default


def _env_int(name: str, default: int, *, min_value: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer: {raw}") from exc
    if value < min_value:
        raise ValueError(f"{name} must be at least {min_value}: {value}")
    return value


def _ollama_embed_fn() -> OllamaEmbeddingFunction:
    embed_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    return OllamaEmbeddingFunction(url=ollama_host(), model_name=embed_model)


def _load_chroma_collection() -> Collection | None:
    """
    Open the persisted ChromaDB collection.
    The same embedding settings used during ingest must be used here as well.
    """
    chroma_path = _env_path("CHROMA_PATH", DEFAULT_CHROMA_PATH)
    collection_name = os.getenv("COLLECTION_NAME", DEFAULT_COLLECTION)

    if not chroma_path.exists():
        return None

    client = chromadb.PersistentClient(path=str(chroma_path))
    try:
        return client.get_collection(
            name=collection_name,
            embedding_function=_ollama_embed_fn(),
        )
    except (InvalidCollectionException, ValueError):
        return None


def _build_context_and_sources(
    collection: Collection,
    question: str,
    top_k: int,
) -> tuple[str, list[SourceInfo]]:
    """
    Retrieve similar chunks and build the context and source list for the LLM.
    """
    result = collection.query(
        query_texts=[question],
        n_results=top_k,
        include=["documents", "metadatas"],
    )

    docs = (result.get("documents") or [[]])[0]
    metas = (result.get("metadatas") or [[]])[0]

    context_blocks: list[str] = []
    source_pairs: set[tuple[str, int]] = set()

    for i, doc in enumerate(docs):
        meta = metas[i] if i < len(metas) else {}
        meta = meta or {}

        source = str(meta.get("source", "unknown"))
        page_raw = meta.get("page", 0)

        try:
            page = int(page_raw)
        except (TypeError, ValueError):
            page = 0

        if not doc or not str(doc).strip():
            continue

        context_blocks.append(f"[Source: {source} — page {page}]\n{doc}")
        source_pairs.add((source, page))

    context = "\n\n---\n\n".join(context_blocks)
    sources = [
        SourceInfo(source=source, page=page)
        for source, page in sorted(source_pairs, key=lambda item: (item[0], item[1]))
    ]
    return context, sources


def _grounded_answer(system_prompt: str, user_content: str, model: str) -> str:
    """Generate a grounded answer using Ollama."""
    return ollama_chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        model=model,
        temperature=0.2,
    )


SYSTEM_PROMPT = """You are an operational document assistant.

Rules:
- Answer ONLY using the provided CONTEXT. Do not use outside knowledge.
- If the CONTEXT is insufficient to answer, say clearly that you cannot answer from the documents and what is missing.
- Be concise and clear.
- Do not invent facts, page numbers, or document names that are not supported by the CONTEXT.
- Answer in the same language as the user's question when possible."""


def _format_user_prompt(question: str, context: str) -> str:
    return f"""CONTEXT:
{context}

QUESTION:
{question}

Answer based strictly on the CONTEXT."""


load_dotenv()

app = FastAPI(
    title="RAG-Based Operational Document Assistant",
    version="0.2.0",
    description="Question answering over PDFs using RAG (MVP, Ollama).",
)

_chroma_collection: Collection | None = None
_chroma_error: str | None = None


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "RAG-Based Operational Document Assistant",
        "endpoints": {
            "health": "/health",
            "ask": "POST /ask",
            "swagger": "/docs",
        },
        "hint": "Start Ollama first, then run ingest and the API.",
    }


@app.on_event("startup")
def on_startup() -> None:
    """Open the Chroma collection and store an error message if something fails."""
    global _chroma_collection, _chroma_error
    _chroma_collection = None
    _chroma_error = None

    chroma_path = _env_path("CHROMA_PATH", DEFAULT_CHROMA_PATH)

    if not chroma_path.exists():
        _chroma_error = (
            f"ChromaDB directory not found: '{chroma_path}'. "
            "Run `python ingest.py` first to create the index."
        )
        return

    if not ollama_reachable():
        _chroma_error = (
            f"Ollama is not reachable ({ollama_host()}). "
            "Start the Ollama service or fix the OLLAMA_HOST value."
        )
        return

    collection = _load_chroma_collection()
    if collection is None:
        _chroma_error = (
            "ChromaDB collection could not be opened or does not exist. "
            "Make sure `python ingest.py` has been run and that COLLECTION_NAME "
            "matches the ingest configuration."
        )
        return

    _chroma_collection = collection


@app.get("/health")
def health() -> dict[str, Any]:
    ready = _chroma_collection is not None
    ollama_ok = ollama_reachable()

    return {
        "status": "ok" if ready else "degraded",
        "chroma_ready": ready,
        "ollama_reachable": ollama_ok,
        "detail": None if ready else (_chroma_error or "Chroma collection is not ready."),
    }


@app.post("/ask", response_model=AskResponse)
def ask(body: AskRequest) -> AskResponse:
    if _chroma_collection is None:
        raise HTTPException(
            status_code=503,
            detail=_chroma_error or "Vector database is unavailable.",
        )

    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="question cannot be empty.")

    try:
        top_k = _env_int("TOP_K", DEFAULT_TOP_K, min_value=1)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=f"Invalid TOP_K value: {exc}") from exc

    chat_model = ollama_chat_model()

    try:
        context, sources = _build_context_and_sources(
            _chroma_collection,
            question,
            top_k=top_k,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error during retrieval: {exc}",
        ) from exc

    if not context.strip():
        return AskResponse(
            answer="No sufficient context was found in the documents for this question.",
            sources=[],
        )

    user_prompt = _format_user_prompt(question, context)

    try:
        answer = _grounded_answer(SYSTEM_PROMPT, user_prompt, chat_model)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                f"Ollama LLM call failed (model: {chat_model}). "
                f"Try `ollama pull {chat_model}`. Details: {exc}"
            ),
        ) from exc

    if not answer:
        answer = "The model returned an empty response."

    return AskResponse(answer=answer, sources=sources)