from __future__ import annotations

import hashlib
import math
import os
from typing import Protocol
import requests


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class DeterministicHashEmbedder:
    """
    Free dev embedder. Good enough to prove the pipeline works, not ideal for production quality.
    """

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = text.lower().split()
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign

        norm = math.sqrt(sum(x * x for x in vec))
        if norm:
            vec = [x / norm for x in vec]
        return vec

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]


class LocalSentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. Run: pip install sentence-transformers"
            ) from exc
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()


class OpenAICompatibleEmbedder:
    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not self.api_key:
            raise RuntimeError("Missing EMBEDDING_API_KEY.")
        response = requests.post(
            f"{self.base_url}/embeddings",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={"model": self.model, "input": texts},
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        return [item["embedding"] for item in payload["data"]]


class GeminiEmbedder:
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not self.api_key:
            raise RuntimeError("Missing GEMINI_API_KEY.")

        vectors: list[list[float]] = []
        for text in texts:
            url = (
                "https://generativelanguage.googleapis.com/v1beta/"
                f"models/{self.model}:embedContent?key={self.api_key}"
            )
            response = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json={"content": {"parts": [{"text": text}]}},
                timeout=60,
            )
            response.raise_for_status()
            vectors.append(response.json()["embedding"]["values"])
        return vectors


def build_embedder(provider: str, dim: int = 512) -> Embedder:
    provider = provider.lower().strip()

    if provider == "hash":
        return DeterministicHashEmbedder(dim=dim)

    if provider == "local_sentence_transformers":
        return LocalSentenceTransformerEmbedder(
            model_name=os.getenv("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        )

    if provider == "openai_compatible":
        return OpenAICompatibleEmbedder(
            base_url=os.getenv("EMBEDDING_BASE_URL", "https://api.openai.com/v1"),
            api_key=os.getenv("EMBEDDING_API_KEY", ""),
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        )

    if provider == "gemini":
        return GeminiEmbedder(
            api_key=os.getenv("GEMINI_API_KEY", ""),
            model=os.getenv("GEMINI_EMBEDDING_MODEL", "text-embedding-004"),
        )

    raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {provider}")
