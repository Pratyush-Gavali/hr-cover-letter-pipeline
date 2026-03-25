"""
monitoring/patches.py

Copy-paste patches for main.py and api/routes.py.
These are the minimal changes needed to wire up monitoring.

1. main.py  — add middleware + /metrics endpoint
2. routes.py — instrument upload and query handlers
"""


# ═══════════════════════════════════════════════════════════════════════════════
# PATCH 1 — main.py
# Add these two lines inside create_app() or at the bottom of lifespan()
# ═══════════════════════════════════════════════════════════════════════════════

MAIN_PY_PATCH = """
# Add to imports at top of main.py:
from monitoring.middleware import PrometheusMiddleware

# Add inside lifespan(), after app is created but before yield:
app.add_middleware(PrometheusMiddleware)
"""


# ═══════════════════════════════════════════════════════════════════════════════
# PATCH 2 — api/routes.py
# Add imports, the /metrics endpoint, and instrument upload + query handlers
# ═══════════════════════════════════════════════════════════════════════════════

ROUTES_PY_IMPORTS = """
import time
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response as FastAPIResponse

from monitoring.metrics import (
    INGESTION_TOTAL, INGESTION_DURATION_SECONDS,
    CHUNK_COUNT, PII_ENTITIES_REDACTED,
    AI_PROBABILITY, AI_DETECTION_DURATION_SECONDS,
    HR_QUERIES_TOTAL, HR_QUERY_DURATION_SECONDS,
    PIPELINE_STAGE_DURATION_SECONDS,
    CANDIDATES_RETRIEVED, DTW_COMPOSITE_SCORE,
    LLM_RESPONSE_TOKENS, ACTIVE_APPLICANTS,
    QDRANT_UPSERT_DURATION_SECONDS, BGE_INFERENCE_DURATION_SECONDS,
)
"""

METRICS_ENDPOINT = """
@router.get("/metrics", include_in_schema=False)
async def metrics():
    \"\"\"Prometheus scrape endpoint. Mount behind internal network in prod.\"\"\"
    return FastAPIResponse(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
"""

# ── Upload handler instrumentation ─────────────────────────────────────────────
# Wrap your existing upload_cover_letter body with this pattern:

UPLOAD_INSTRUMENTED = """
@router.post("/covers/{job_id}", response_model=UploadResponse, status_code=201)
async def upload_cover_letter(job_id: str, file: UploadFile = File(...), ...):
    t_start = time.perf_counter()

    try:
        file_bytes = await file.read()

        # --- extraction, chunking, masking (existing code) ---
        chunks = chunker.chunk(file_bytes, file.content_type)
        masked_docs = [masker.mask(...) for ...]
        masked_texts = [m.masked_text for m in masked_docs]

        CHUNK_COUNT.observe(len(chunks))
        PII_ENTITIES_REDACTED.observe(sum(m.entity_count for m in masked_docs))

        # --- AI detection ---
        t_ai = time.perf_counter()
        ai_result = detector.detect(masked_texts)
        AI_DETECTION_DURATION_SECONDS.observe(time.perf_counter() - t_ai)
        AI_PROBABILITY.observe(ai_result.ai_probability)

        # --- Qdrant upsert loop ---
        for i, (masked, chunk) in enumerate(zip(masked_docs, chunks)):
            t_upsert = time.perf_counter()
            qdrant.upsert_chunk(ChunkMetadata(...))
            QDRANT_UPSERT_DURATION_SECONDS.observe(time.perf_counter() - t_upsert)

        # --- success instrumentation ---
        INGESTION_TOTAL.labels(job_id=job_id, status="success").inc()
        INGESTION_DURATION_SECONDS.labels(job_id=job_id).observe(
            time.perf_counter() - t_start
        )
        ACTIVE_APPLICANTS.labels(job_id=job_id).inc()

        return UploadResponse(...)

    except Exception:
        INGESTION_TOTAL.labels(job_id=job_id, status="error").inc()
        raise
"""

# ── Query handler instrumentation ──────────────────────────────────────────────

QUERY_INSTRUMENTED = """
@router.post("/query", response_model=QueryResponse)
async def query_talent(body: QueryRequest, rag=Depends(get_rag)):
    t_start = time.perf_counter()
    try:
        result = await rag.ainvoke({
            "hr_prompt": body.prompt,
            "job_id":    body.job_id,
            "filters":   {"ai_prob_max": body.ai_prob_max},
        })

        raw       = result.get("raw_results", [])
        reranked  = result.get("reranked", [])
        response  = result.get("response", "")

        CANDIDATES_RETRIEVED.observe(len(raw))
        for ts in reranked:
            DTW_COMPOSITE_SCORE.observe(ts.composite_score)
        LLM_RESPONSE_TOKENS.observe(len(response.split()))

        HR_QUERIES_TOTAL.labels(job_id=body.job_id, status="success").inc()
        HR_QUERY_DURATION_SECONDS.labels(job_id=body.job_id).observe(
            time.perf_counter() - t_start
        )

        return QueryResponse(
            job_id=body.job_id,
            response=response,
            candidate_count=len(reranked),
            top_candidates=[
                {"applicant_id": ts.applicant_id,
                 "composite_score": ts.composite_score,
                 "thematic_coverage": ts.thematic_coverage,
                 "best_chunk": ts.best_chunk}
                for ts in reranked
            ],
        )
    except Exception:
        HR_QUERIES_TOTAL.labels(job_id=body.job_id, status="error").inc()
        raise
"""


# ═══════════════════════════════════════════════════════════════════════════════
# PATCH 3 — rag/graph.py
# Wrap each node with a stage timer
# ═══════════════════════════════════════════════════════════════════════════════

GRAPH_STAGE_TIMER = """
import time
from monitoring.metrics import PIPELINE_STAGE_DURATION_SECONDS


def _timed(stage: str, fn):
    \"\"\"Wraps an async graph node to record its latency.\"\"\"
    async def wrapper(state):
        t = time.perf_counter()
        result = await fn(state)
        PIPELINE_STAGE_DURATION_SECONDS.labels(stage=stage).observe(
            time.perf_counter() - t
        )
        return result
    return wrapper


# In build_talent_copilot_graph(), replace the add_node calls with:
builder.add_node("query_parse", _timed("query_parse", make_query_parse_node(llm, jd_store)))
builder.add_node("retrieve",    _timed("retrieve",    make_retrieve_node(retriever)))
builder.add_node("rerank",      _timed("rerank",      make_rerank_node(retriever)))
builder.add_node("synthesise",  _timed("synthesise",  make_synthesise_node(llm)))
"""


# ═══════════════════════════════════════════════════════════════════════════════
# PATCH 4 — storage/qdrant_store.py and sva/engine.py
# Instrument BGE-M3 calls with call_site label
# ═══════════════════════════════════════════════════════════════════════════════

BGE_INSTRUMENTATION = """
# Wherever you call _embed_model().encode(), wrap with timing:

from monitoring.metrics import BGE_INFERENCE_DURATION_SECONDS

# Example in qdrant_store.upsert_chunk():
t = time.perf_counter()
dense_vec = self._model.encode([meta.chunk_text], ...)["dense_vecs"][0].tolist()
BGE_INFERENCE_DURATION_SECONDS.labels(call_site="ingestion").observe(
    time.perf_counter() - t
)

# Example in sva/engine.py SVATrajectoryReranker._score_applicant():
t = time.perf_counter()
chunk_embs = np.array(_embed_model().encode(texts, ...)["dense_vecs"])
BGE_INFERENCE_DURATION_SECONDS.labels(call_site="rerank").observe(
    time.perf_counter() - t
)

# Example in qdrant_store.retrieve():
t = time.perf_counter()
query_vec = self._model.encode([query_text], ...)["dense_vecs"][0].tolist()
BGE_INFERENCE_DURATION_SECONDS.labels(call_site="retrieve").observe(
    time.perf_counter() - t
)
"""
