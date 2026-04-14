"""
A feed script that indexes PDFs by splitting them into chunks and storing them in ChromaDB.

Usage:
  python ingest.py
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from dotenv import load_dotenv
from pypdf import PdfReader

from ollama_utils import ollama_host, ollama_reachable


DEFAULT_DOCS_PATH = Path(__file__).resolve().parent / "data" / "docs"
DEFAULT_CHROMA_PATH = Path(__file__).resolve().parent / "chroma_db"
DEFAULT_COLLECTION = "operational_docs"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200


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


def list_pdf_files(docs_dir: Path) -> list[Path]:
    """Return all PDF files in the directory in alphabetical order."""
    if not docs_dir.is_dir():
        return []
    return sorted(p for p in docs_dir.glob("*.pdf") if p.is_file())


def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """
    Return page number (1-based) and extracted text from the PDF.
    Unreadable or empty pages are skipped.
    """
    reader = PdfReader(str(pdf_path))
    pages: list[tuple[int, str]] = []

    for idx, page in enumerate(reader.pages, start=1):
        try:
            text = (page.extract_text() or "").strip()
        except Exception as exc:
            print(
                f"Warning: failed to read page {idx} in {pdf_path.name}: {exc}",
                file=sys.stderr,
            )
            continue

        if text:
            pages.append((idx, text))

    return pages


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Create chunks using a simple character-based sliding window.
    overlap must be smaller than chunk_size.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)

        if end >= text_len:
            break

        start = end - overlap

    return chunks


def build_documents(
    pdf_files: list[Path],
    chunk_size: int,
    overlap: int,
) -> tuple[list[str], list[dict[str, Any]], list[str]]:
    """
    Build document texts, metadata, and unique IDs for all PDFs.
    """
    texts: list[str] = []
    metadatas: list[dict[str, Any]] = []
    ids: list[str] = []

    for pdf_path in pdf_files:
        source_name = pdf_path.name
        for page_num, page_text in extract_pages(pdf_path):
            chunks = chunk_text(page_text, chunk_size, overlap)
            for chunk_index, chunk in enumerate(chunks):
                texts.append(chunk)
                metadatas.append(
                    {
                        "source": source_name,
                        "page": page_num,
                        "chunk_index": chunk_index,
                    }
                )
                ids.append(str(uuid.uuid4()))

    return texts, metadatas, ids


def run_ingest() -> None:
    """Main ingest flow."""
    load_dotenv()

    if not ollama_reachable():
        print(
            f"Error: could not connect to Ollama ({ollama_host()}). "
            "Make sure Ollama is running and OLLAMA_HOST is set correctly.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        docs_dir = _env_path("DOCS_PATH", DEFAULT_DOCS_PATH)
        chroma_path = _env_path("CHROMA_PATH", DEFAULT_CHROMA_PATH)
        collection_name = os.getenv("COLLECTION_NAME", DEFAULT_COLLECTION)
        embed_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        chunk_size = _env_int("CHUNK_SIZE", DEFAULT_CHUNK_SIZE, min_value=1)
        overlap = _env_int("CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP, min_value=0)
    except ValueError as exc:
        print(f"Error: invalid environment variable: {exc}", file=sys.stderr)
        sys.exit(1)

    pdf_files = list_pdf_files(docs_dir)
    if not pdf_files:
        print(
            f"Error: no PDF files found in '{docs_dir}'. "
            "Please add .pdf files to this folder and try again.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Documents directory: {docs_dir}")
    print(f"Number of PDFs found: {len(pdf_files)}")
    print("Extracting text and creating chunks...")

    texts, metadatas, ids = build_documents(pdf_files, chunk_size, overlap)
    if not texts:
        print(
            "Error: no extractable text found in the PDFs "
            "(files may be empty, encrypted, or image-only).",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Total chunks: {len(texts)}")
    print(f"ChromaDB path: {chroma_path}")
    print(f"Ollama embedding model: {embed_model}")

    chroma_path.mkdir(parents=True, exist_ok=True)

    embedding_fn = OllamaEmbeddingFunction(url=ollama_host(), model_name=embed_model)
    client = chromadb.PersistentClient(path=str(chroma_path))

    try:
        client.delete_collection(name=collection_name)
        print(f"Existing collection deleted: {collection_name}")
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        metadata={"description": "Operational PDF chunks (Ollama embeddings)"},
    )

    batch_size = 128
    try:
        for i in range(0, len(texts), batch_size):
            collection.add(
                ids=ids[i:i + batch_size],
                documents=texts[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size],
            )
    except Exception as exc:
        print(
            "Error: a problem occurred while generating embeddings or writing to Chroma. "
            f"Is the model downloaded? Try: ollama pull {embed_model}\n"
            f"Details: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Ingest completed. You can now start the API.")


if __name__ == "__main__":
    run_ingest()