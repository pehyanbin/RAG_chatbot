from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Any
from pypdf import PdfReader
from io import BytesIO
from pathlib import Path
from bs4 import BeautifulSoup
import json

from rag_agent.agent import RAGAgent
from rag_agent.config import settings

app = FastAPI(title="Multi-provider RAG Agent", version="0.2.0")
app.mount("/static", StaticFiles(directory="static"), name="static")

agent = RAGAgent.from_settings(settings)


class IngestTextRequest(BaseModel):
    text: str = Field(..., min_length=1)
    source: str = "manual"
    metadata: dict[str, Any] = Field(default_factory=dict)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    retrieval_mode: str | None = Field(default=None, description="vector, keyword, or hybrid")


def _extract_pdf(data: bytes) -> str:
    reader = PdfReader(BytesIO(data))
    pages: list[str] = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"\n\n[Page {i}]\n{text}")
    return "\n".join(pages).strip()


def _extract_text_file(filename: str, data: bytes) -> str:
    name = filename.lower()

    if name.endswith(".pdf"):
        return _extract_pdf(data)

    text = data.decode("utf-8", errors="ignore")

    if name.endswith((".html", ".htm")):
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text("\n").strip()

    if name.endswith(".json"):
        try:
            return json.dumps(json.loads(text), indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            return text

    return text


@app.get("/")
def root():
    return {
        "name": "Multi-provider RAG Agent",
        "docs": "/docs",
        "chat_ui": "/chat",
        "llm_provider": settings.llm_provider,
        "embedding_provider": settings.embedding_provider,
        "retrieval_mode": settings.retrieval_mode,
    }


@app.get("/chat")
def chat_ui():
    return FileResponse(Path("static") / "index.html")


@app.get("/providers")
def providers():
    return {
        "active_llm_provider": settings.llm_provider,
        "active_embedding_provider": settings.embedding_provider,
        "active_retrieval_mode": settings.retrieval_mode,
        "supported_llm_providers": [
            "openai_api_key",
            "openai_compatible_api",
            "nvidia_nim_api",
            "ollama_api",
            "openrouter_api",
            "gemini_api_key",
            "anthropic_api_key",
            "deepseek_api",
            "xai_api",
            "moonshot_ai",
            "codex_oauth",
            "openai_plus_pro_subscription",
            "gemini_cli_oauth",
            "google_antigravity_oauth",
            "opencode",
        ],
        "supported_embedding_providers": [
            "hash",
            "local_sentence_transformers",
            "openai_compatible",
            "gemini",
        ],
        "supported_retrieval_modes": ["vector", "keyword", "hybrid"],
    }


@app.post("/ingest/text")
def ingest_text(req: IngestTextRequest):
    count = agent.ingest_text(req.text, source=req.source, metadata=req.metadata)
    return {"chunks_inserted": count}


@app.post("/ingest/pdf")
async def ingest_pdf(file: UploadFile = File(...), source: str | None = None):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported by this endpoint.")

    data = await file.read()
    text = _extract_pdf(data)
    if not text:
        raise HTTPException(status_code=400, detail="No extractable text found in PDF.")

    count = agent.ingest_text(
        text,
        source=source or file.filename,
        metadata={"filename": file.filename, "content_type": file.content_type},
    )
    return {"chunks_inserted": count, "source": source or file.filename}


@app.post("/ingest/files")
async def ingest_files(files: list[UploadFile] = File(...)):
    """
    Upload and ingest multiple documents automatically from the Chat UI or Swagger.
    Supports PDF, TXT, MD, CSV, JSON, HTML.
    """
    results = []
    total = 0
    supported_suffixes = (".pdf", ".txt", ".md", ".csv", ".json", ".html", ".htm")

    for file in files:
        filename = file.filename or "uploaded-file"
        if not filename.lower().endswith(supported_suffixes):
            results.append({
                "filename": filename,
                "chunks_inserted": 0,
                "status": "skipped",
                "reason": "unsupported file type",
            })
            continue

        data = await file.read()
        text = _extract_text_file(filename, data)

        if not text.strip():
            results.append({
                "filename": filename,
                "chunks_inserted": 0,
                "status": "skipped",
                "reason": "no extractable text",
            })
            continue

        count = agent.ingest_text(
            text,
            source=filename,
            metadata={"filename": filename, "content_type": file.content_type},
        )
        total += count
        results.append({
            "filename": filename,
            "chunks_inserted": count,
            "status": "ingested",
        })

    return {
        "files": results,
        "total_chunks_inserted": total,
    }


@app.post("/ask")
def ask(req: AskRequest):
    return agent.ask(req.question, top_k=req.top_k, retrieval_mode=req.retrieval_mode)


@app.post("/reset")
def reset():
    agent.reset()
    return {"status": "database reset"}
