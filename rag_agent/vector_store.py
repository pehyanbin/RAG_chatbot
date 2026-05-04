from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass
class RetrievedChunk:
    id: int
    content: str
    source: str
    metadata: dict[str, Any]
    score: float


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


class SQLiteVectorStore:
    """
    Tiny vector store for starter RAG projects.

    Includes vector, keyword, and hybrid retrieval. This is easy to run anywhere.
    For production-scale search, use pgvector, Qdrant, Weaviate, Pinecone, LanceDB, etc.
    """

    def __init__(self, path: str):
        self.path = path
        self._init()

    def _connect(self):
        return sqlite3.connect(self.path)

    def _init(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    source TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    embedding_json TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def reset(self):
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks")
            conn.commit()

    def add(self, content: str, source: str, metadata: dict[str, Any], embedding: list[float]):
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO chunks (content, source, metadata_json, embedding_json)
                VALUES (?, ?, ?, ?)
                """,
                (content, source, json.dumps(metadata), json.dumps(embedding)),
            )
            conn.commit()

    def _all_rows(self) -> list[tuple[int, str, str, dict[str, Any], list[float]]]:
        rows = []
        with self._connect() as conn:
            for id_, content, source, metadata_json, embedding_json in conn.execute(
                "SELECT id, content, source, metadata_json, embedding_json FROM chunks"
            ):
                rows.append((id_, content, source, json.loads(metadata_json), json.loads(embedding_json)))
        return rows

    def _vector_scores(self, query_embedding: list[float], rows) -> dict[int, float]:
        q = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return {}
        q = q / q_norm

        scores = {}
        for id_, _content, _source, _metadata, embedding in rows:
            emb = np.array(embedding, dtype=np.float32)
            emb_norm = np.linalg.norm(emb)
            if emb_norm == 0:
                scores[id_] = 0.0
            else:
                scores[id_] = float(np.dot(q, emb / emb_norm))
        return scores

    def _keyword_scores(self, query: str, rows) -> dict[int, float]:
        """
        Lightweight BM25-like scoring without requiring a separate index.
        """
        query_terms = _tokenize(query)
        if not query_terms:
            return {row[0]: 0.0 for row in rows}

        docs = [_tokenize(row[1]) for row in rows]
        n_docs = len(docs)
        avg_len = sum(len(doc) for doc in docs) / max(1, n_docs)
        df = {}
        for doc in docs:
            for term in set(doc):
                df[term] = df.get(term, 0) + 1

        k1 = 1.5
        b = 0.75
        scores = {}
        for row, doc in zip(rows, docs):
            id_ = row[0]
            doc_len = len(doc) or 1
            freqs = {}
            for term in doc:
                freqs[term] = freqs.get(term, 0) + 1

            score = 0.0
            for term in query_terms:
                if term not in freqs:
                    continue
                idf = np.log(1 + (n_docs - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5))
                tf = freqs[term]
                denom = tf + k1 * (1 - b + b * doc_len / max(avg_len, 1))
                score += float(idf * (tf * (k1 + 1)) / denom)
            scores[id_] = score
        return scores

    @staticmethod
    def _minmax(scores: dict[int, float]) -> dict[int, float]:
        if not scores:
            return {}
        values = list(scores.values())
        lo, hi = min(values), max(values)
        if hi - lo < 1e-9:
            return {k: 0.0 for k in scores}
        return {k: (v - lo) / (hi - lo) for k, v in scores.items()}

    def search(
        self,
        query_embedding: list[float],
        query_text: str = "",
        top_k: int = 5,
        mode: str = "hybrid",
    ) -> list[RetrievedChunk]:
        rows = self._all_rows()
        if not rows:
            return []

        mode = (mode or "hybrid").lower()
        vector_scores = self._vector_scores(query_embedding, rows)
        keyword_scores = self._keyword_scores(query_text, rows)

        vector_norm = self._minmax(vector_scores)
        keyword_norm = self._minmax(keyword_scores)

        final_scores = {}
        for id_, *_ in rows:
            if mode == "vector":
                final_scores[id_] = vector_scores.get(id_, 0.0)
            elif mode == "keyword":
                final_scores[id_] = keyword_scores.get(id_, 0.0)
            else:
                final_scores[id_] = 0.72 * vector_norm.get(id_, 0.0) + 0.28 * keyword_norm.get(id_, 0.0)

        by_id = {row[0]: row for row in rows}
        ranked_ids = sorted(final_scores, key=lambda id_: final_scores[id_], reverse=True)[:top_k]

        results = []
        for id_ in ranked_ids:
            row_id, content, source, metadata, _embedding = by_id[id_]
            results.append(
                RetrievedChunk(
                    id=row_id,
                    content=content,
                    source=source,
                    metadata=metadata,
                    score=float(final_scores[id_]),
                )
            )
        return results
