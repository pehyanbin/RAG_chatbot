import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


def _float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value in (None, "") else float(value)


def _int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value in (None, "") else int(value)


@dataclass(frozen=True)
class Settings:
    rag_db_path: str = os.getenv("RAG_DB_PATH", "rag.sqlite3")

    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama_api")
    llm_temperature: float = _float("LLM_TEMPERATURE", 0.2)
    llm_max_tokens: int = _int("LLM_MAX_TOKENS", 900)

    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "hash")
    embedding_dim: int = _int("EMBEDDING_DIM", 512)

    retrieval_mode: str = os.getenv("RETRIEVAL_MODE", "hybrid")


settings = Settings()
