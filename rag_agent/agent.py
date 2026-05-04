from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .chunking import chunk_text
from .config import Settings
from .embeddings import build_embedder, Embedder
from .llm import build_llm, GenerationConfig, LLM
from .vector_store import SQLiteVectorStore


SYSTEM_PROMPT = """You are a careful RAG assistant.
Use the retrieved context to answer the user's question.
If the context does not contain the answer, say that you do not know from the indexed documents.
Cite sources using bracket IDs like [1], [2].
Be concise but complete.
"""


@dataclass
class RAGAgent:
    store: SQLiteVectorStore
    embedder: Embedder
    llm: LLM
    default_retrieval_mode: str = "hybrid"

    @classmethod
    def from_settings(cls, settings: Settings) -> "RAGAgent":
        store = SQLiteVectorStore(settings.rag_db_path)
        embedder = build_embedder(settings.embedding_provider, dim=settings.embedding_dim)
        llm = build_llm(
            settings.llm_provider,
            GenerationConfig(
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
            ),
        )
        return cls(store=store, embedder=embedder, llm=llm, default_retrieval_mode=settings.retrieval_mode)

    def reset(self):
        self.store.reset()

    def ingest_text(self, text: str, source: str, metadata: dict[str, Any] | None = None) -> int:
        metadata = metadata or {}
        chunks = chunk_text(text)
        if not chunks:
            return 0

        vectors = self.embedder.embed(chunks)
        for i, (chunk, vector) in enumerate(zip(chunks, vectors), start=1):
            self.store.add(
                content=chunk,
                source=source,
                metadata={**metadata, "chunk_index": i},
                embedding=vector,
            )
        return len(chunks)

    def ask(self, question: str, top_k: int = 5, retrieval_mode: str | None = None) -> dict[str, Any]:
        query_vector = self.embedder.embed([question])[0]
        mode = retrieval_mode or self.default_retrieval_mode
        retrieved = self.store.search(query_vector, query_text=question, top_k=top_k, mode=mode)

        context_blocks = []
        citations = []
        for idx, chunk in enumerate(retrieved, start=1):
            context_blocks.append(
                f"[{idx}] Source: {chunk.source}; score={chunk.score:.3f}\n{chunk.content}"
            )
            citations.append(
                {
                    "id": idx,
                    "source": chunk.source,
                    "score": round(chunk.score, 4),
                    "metadata": chunk.metadata,
                    "preview": chunk.content[:240],
                }
            )

        user_prompt = f"""Question:
{question}

Retrieved context:
{chr(10).join(context_blocks) if context_blocks else "(no context found)"}

Answer with source citations like [1] where relevant.
"""

        answer = self.llm.generate(SYSTEM_PROMPT, user_prompt)

        return {
            "answer": answer,
            "citations": citations,
            "top_k": top_k,
            "retrieval_mode": mode,
        }
