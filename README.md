# DocuRAG — RAG-Based Operational Document Assistant (MVP)

A lightweight **Retrieval-Augmented Generation (RAG)** service for answering questions over PDF operational documents.  
Built with **local LLM and embeddings via Ollama** (no OpenAI required) and **ChromaDB** as a persistent vector store.

---

## 🚀 Features

- Reads PDF documents from `data/docs` using `pypdf`
- Splits documents into chunks with metadata (file name + page number)
- Indexes content into ChromaDB using Ollama embeddings
- Retrieves relevant chunks and generates contextual answers via `/ask`
- Returns answers with **source file names** and **page references**

---

## 🧰 Tech Stack

| Component        | Description                        |
|-----------------|------------------------------------|
| Python 3.11+     | Runtime environment                |
| FastAPI          | HTTP API                           |
| ChromaDB         | Persistent vector database         |
| pypdf            | PDF text extraction                |
| Ollama           | Local embeddings + LLM             |
| python-dotenv    | Environment variable management    |
| Docker           | Containerized API (Ollama external)|

---

## ⚙️ Prerequisites

### Install Ollama

https://ollama.com

Run:

ollama serve

Pull models:

ollama pull nomic-embed-text
ollama pull llama3.2

---

## 📁 Project Structure

rag-app/
  app.py
  ingest.py
  ollama_utils.py
  requirements.txt
  Dockerfile
  .env.example
  README.md
  data/docs/
  chroma_db/

---

## 🛠️ Setup

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

---

## 🔐 .env

OLLAMA_HOST=http://127.0.0.1:11434
OLLAMA_EMBED_MODEL=nomic-embed-text
OLLAMA_CHAT_MODEL=llama3.2

---

## 📄 Add PDFs

Put files into:

data/docs/

---

## 📥 Ingest

python ingest.py

---

## ▶️ Run API

uvicorn app:app --reload --host 0.0.0.0 --port 8000

---

## 📚 Docs

http://localhost:8000/docs

---

## 🔌 Endpoints

GET /health

POST /ask

---

## 📌 Example

curl -X POST "http://localhost:8000/ask" -H "Content-Type: application/json" -d '{"question":"When is backup performed?"}'

---

## 🐳 Docker

docker build -t docurag .

docker run --rm -p 8000:8000 --env-file .env -v "$(pwd)/data/docs:/app/data/docs" -v "$(pwd)/chroma_db:/app/chroma_db" docurag

---

## 📄 License

MVP project.
