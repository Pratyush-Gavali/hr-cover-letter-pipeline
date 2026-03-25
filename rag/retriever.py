"""
Qdrant retriever + BGE CrossEncoder reranker.

Two-stage retrieval
--------------------
Stage 1 — Hybrid search (fast, ~10ms)
  Qdrant RRF-fused dense + sparse ANN search.
  Payload pre-filters applied before ANN (ai_probability, jd_match_score).
  Returns top-20 chunks from across all applicants.

Stage 2 — CrossEncoder rerank (precise, ~150ms)
  BGE CrossEncoder (ms-marco-MiniLM-L-6-v2) jointly encodes
  (query, passage) pairs — cross-attention captures entailment
  that bi-encoder retrieval cannot (bi-encoders encode query and
  passage independently; CrossEncoders see both at once).
  Runs only over the 20 shortlisted chunks, not the full collection.
  Deduplicates by applicant_id, returning top-5 unique applicants.

Why two stages?
---------------
CrossEncoder accuracy >> bi-encoder, but CrossEncoder latency scales
linearly with collection size (can't ANN-search with a CrossEncoder).
The two-stage approach gets you ANN speed for candidate selection,
CrossEncoder precision for the final ranking — standard RAG pattern.
"""

from __future__ import annotations
import asyncio
from concurrent.futures import ThreadPoolExecutor

from sentence_transformers import CrossEncoder

from storage.qdrant_store import QdrantCoverStore


_RERANK_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_executor = ThreadPoolExecutor(max_workers=4)


class CoverLetterRetriever:
    """
    Async-compatible retriever wrapping Qdrant + CrossEncoder.

    Parameters
    ----------
    store
        Initialised QdrantCoverStore.
    rerank_model_id
        HuggingFace CrossEncoder model ID.
    """

    def __init__(
        self,
        store: QdrantCoverStore,
        rerank_model_id: str = _RERANK_MODEL_ID,
    ):
        self._store = store
        self._reranker = CrossEncoder(rerank_model_id)

    async def search(
        self,
        job_id: str,
        query: str,
        top_k: int = 20,
        ai_prob_max: float = 1.0,
        min_match_score: float = 0.0,
    ) -> list[dict]:
        """Async wrapper for Qdrant hybrid search (runs in thread pool)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _executor,
            lambda: self._store.hybrid_search(
                job_id=job_id,
                query_text=query,
                top_k=top_k,
                ai_prob_max=ai_prob_max,
                min_match_score=min_match_score,
            ),
        )

    async def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_n: int = 5,
    ) -> list[dict]:
        """
        CrossEncoder reranking with applicant-level deduplication.

        Parameters
        ----------
        query
            HR requirement string (same as Stage 1 query).
        candidates
            Qdrant payloads from hybrid_search (typically 20 chunks).
        top_n
            Unique applicants to return after reranking.

        Returns
        -------
        list[dict]
            Top-n payloads, deduplicated by applicant_id,
            sorted by CrossEncoder score descending.
            Each dict includes 'rerank_score' added by this function.
        """
        if not candidates:
            return []

        pairs = [(query, c.get("chunk_text", "")) for c in candidates]

        loop = asyncio.get_event_loop()
        scores: list[float] = await loop.run_in_executor(
            _executor,
            lambda: self._reranker.predict(pairs).tolist(),
        )

        for doc, score in zip(candidates, scores):
            doc["rerank_score"] = round(float(score), 4)

        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        # Deduplicate: keep highest-scoring chunk per applicant
        seen: set[str] = set()
        unique: list[dict] = []
        for doc in candidates:
            aid = doc.get("applicant_id", "")
            if aid not in seen:
                seen.add(aid)
                unique.append(doc)
            if len(unique) >= top_n:
                break

        return unique
