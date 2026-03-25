"""
Qdrant vector store — collection schema and upsert helpers.

Collection design
-----------------
One collection per job posting: covers_{job_id}
  - Dense vector:  BGE-M3 1024-dim  (semantic matching)
  - Sparse vector: BM25 sparse      (keyword recall)
  - Payload:       all SVA scores + metadata for pre-filtering

Hybrid retrieval uses Reciprocal Rank Fusion (RRF) to fuse
dense and sparse ranked lists before the CrossEncoder reranker.

Why one collection per job (not one global collection)?
-------------------------------------------------------
A global collection with a job_id filter works fine at 10k docs.
At 100k+ docs, payload filtering before ANN search still scans
all vectors. Per-job collections keep each search space bounded
to the actual applicant pool (typically 200-2000 points).
Qdrant collection creation is fast (<50ms) so creating one per
job posting adds negligible overhead.

Why Qdrant over ChromaDB?
--------------------------
ChromaDB is excellent for prototyping. Qdrant wins here for:
1. Native named multi-vector support (dense + sparse in one point)
2. Built-in hybrid search with RRF fusion (no custom merge code)
3. Payload indexes with pre-filtering before ANN — critical for
   filtering on ai_probability and jd_match_score without full scan
4. Production-grade performance with horizontal scaling support
"""

from __future__ import annotations
from dataclasses import asdict

from qdrant_client import QdrantClient, models as qm
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    PointStruct,
    SparseVector,
    Filter,
    FieldCondition,
    Range,
)
from FlagEmbedding import BGEM3FlagModel

from sva.engine import SVAScores


_DENSE_DIM = 1024
_COLLECTION_PREFIX = "covers_"


class QdrantCoverStore:
    """
    Manage Qdrant collections and point upserts for cover letters.

    Parameters
    ----------
    url
        Qdrant server URL.
    api_key
        Qdrant API key (for cloud/hosted deployments; None for local).
    embed_model
        Pre-initialised BGE-M3 model (shared with SVA engine to avoid
        loading two copies of the same 570MB model into VRAM).
    """

    def __init__(
        self,
        url: str,
        api_key: str | None,
        embed_model: BGEM3FlagModel,
    ):
        # ":memory:" is the qdrant-client in-memory mode; it must be passed
        # positionally — passing it as url= makes urllib3 try to parse it as
        # a network address and raises LocationParseError.
        if url == ":memory:":
            self._client = QdrantClient(":memory:")
        else:
            self._client = QdrantClient(url=url, api_key=api_key)
        self._model = embed_model

    # ── Collection lifecycle ───────────────────────────────────────────────────

    def ensure_collection(self, job_id: str) -> None:
        """
        Idempotently create a per-job collection with hybrid vector config.
        Safe to call on every ingestion request.
        """
        name = self._col(job_id)
        if self._client.collection_exists(name):
            return

        self._client.create_collection(
            collection_name=name,
            vectors_config={
                "dense": VectorParams(
                    size=_DENSE_DIM,
                    distance=Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams()
            },
        )

        # Create payload indexes for pre-filtering
        # Without these, Qdrant performs a full collection scan before ANN
        for field, schema in [
            ("jd_match_score", "float"),
            ("ai_probability", "float"),
            ("applicant_id",   "keyword"),
            ("chunk_index",    "integer"),
        ]:
            self._client.create_payload_index(
                collection_name=name,
                field_name=field,
                field_schema=schema,
            )

    # ── Upsert ─────────────────────────────────────────────────────────────────

    def upsert_chunk(
        self,
        job_id: str,
        applicant_id: str,
        chunk_index: int,
        chunk_text: str,
        scores: SVAScores,
        blob_path: str,
    ) -> str:
        """
        Embed a single masked paragraph chunk and upsert to Qdrant.

        BGE-M3 produces both dense (1024-d) and sparse (BM25 lexical weights)
        in a single forward pass — no need to run two separate models.

        Returns
        -------
        str
            Point ID (deterministic from applicant_id + chunk_index).
        """
        self.ensure_collection(job_id)
        point_id = self._make_id(applicant_id, chunk_index)

        encoding = self._model.encode(
            [chunk_text],
            return_dense=True,
            return_sparse=True,
            batch_size=1,
        )
        dense_vec: list[float] = encoding["dense_vecs"][0].tolist()
        sparse_data = encoding["lexical_weights"][0]
        sparse_vec = SparseVector(
            indices=list(sparse_data.keys()),
            values=list(sparse_data.values()),
        )

        payload = {
            "applicant_id": applicant_id,
            "job_id":        job_id,
            "chunk_index":   chunk_index,
            "chunk_text":    chunk_text,   # Stored masked — never raw PII
            "blob_path":     blob_path,
            **asdict(scores),
        }
        payload["human_confidence"] = scores.human_confidence

        self._client.upsert(
            collection_name=self._col(job_id),
            points=[
                PointStruct(
                    id=point_id,
                    vector={
                        "dense":  dense_vec,
                        "sparse": sparse_vec,
                    },
                    payload=payload,
                )
            ],
        )
        return str(point_id)

    # ── Hybrid retrieval ───────────────────────────────────────────────────────

    def hybrid_search(
        self,
        job_id: str,
        query_text: str,
        top_k: int = 20,
        ai_prob_max: float = 1.0,
        min_match_score: float = 0.0,
    ) -> list[dict]:
        """
        Hybrid dense+sparse search with payload pre-filtering.

        The payload filter runs BEFORE the ANN search, dramatically
        reducing the search space when filtering on ai_probability
        or jd_match_score.

        Parameters
        ----------
        query_text
            HR requirement string.
        top_k
            Number of results before CrossEncoder reranking.
        ai_prob_max
            Pre-filter: exclude likely AI-written letters (0.3 is a good default).
        min_match_score
            Pre-filter: exclude letters below this JD match threshold.

        Returns
        -------
        list[dict]
            Qdrant point payloads, sorted by RRF-fused score.
        """
        col = self._col(job_id)
        enc = self._model.encode(
            [query_text], return_dense=True, return_sparse=True
        )
        dense_q = enc["dense_vecs"][0].tolist()
        sparse_data = enc["lexical_weights"][0]
        sparse_q = SparseVector(
            indices=list(sparse_data.keys()),
            values=list(sparse_data.values()),
        )

        conditions = []
        if ai_prob_max < 1.0:
            conditions.append(FieldCondition(
                key="ai_probability",
                range=Range(lte=ai_prob_max),
            ))
        if min_match_score > 0.0:
            conditions.append(FieldCondition(
                key="jd_match_score",
                range=Range(gte=min_match_score),
            ))
        payload_filter = Filter(must=conditions) if conditions else None

        # Qdrant native hybrid search with built-in RRF fusion (v1.7+).
        # The filter is applied inside each Prefetch so that it is respected
        # by both the in-memory local client and the remote server.
        results = self._client.query_points(
            collection_name=col,
            prefetch=[
                qm.Prefetch(query=dense_q,  using="dense",  limit=top_k * 2,
                            filter=payload_filter),
                qm.Prefetch(query=sparse_q, using="sparse", limit=top_k * 2,
                            filter=payload_filter),
            ],
            query=qm.FusionQuery(fusion=qm.Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )
        return [r.payload for r in results.points]

    # ── Utilities ──────────────────────────────────────────────────────────────

    @staticmethod
    def _col(job_id: str) -> str:
        return f"{_COLLECTION_PREFIX}{job_id}"

    @staticmethod
    def _make_id(applicant_id: str, chunk_index: int) -> int:
        """Deterministic int ID from applicant_id + chunk_index."""
        import hashlib
        h = hashlib.md5(f"{applicant_id}_{chunk_index}".encode()).hexdigest()
        return int(h[:15], 16)
