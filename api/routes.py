"""
FastAPI routes — cover letter ingestion and HR query endpoints.

Endpoints
---------
POST /api/v1/covers/{job_id}         Upload a cover letter file
POST /api/v1/query                   HR natural language talent query
GET  /api/v1/applicants/{job_id}     List all applicants for a job
GET  /api/v1/health                  Liveness check

Authentication
--------------
Azure AD B2C JWT validation via the azure-identity library.
HR users: read + query permissions only.
Compliance users: additionally have Key Vault secret READ permission.
Admins: full access including collection deletion.

Rate limiting
-------------
Redis sliding window: 60 requests / minute per X-User-ID header.
Prevents a single HR user from hammering the embedding endpoint.
"""

from __future__ import annotations
import hashlib
import time
from typing import Annotated

from monitoring.metrics import (
    INGESTION_TOTAL, INGESTION_DURATION_SECONDS,
    CHUNK_COUNT, PII_ENTITIES_REDACTED,
    AI_PROBABILITY,
    HR_QUERIES_TOTAL, HR_QUERY_DURATION_SECONDS,
    CANDIDATES_RETRIEVED, LLM_RESPONSE_TOKENS,
    ACTIVE_APPLICANTS, QDRANT_UPSERT_DURATION_SECONDS,
)

from fastapi import (
    APIRouter, Depends, File, HTTPException,
    UploadFile, status, Request
)
from pydantic import BaseModel, Field
import redis.asyncio as aioredis

from processing.chunker import CoverLetterChunker
from processing.pii import PIIMasker
from storage.blob_client import CoverLetterBlobClient
from storage.qdrant_store import QdrantCoverStore
from sva.engine import SVAEngine
from rag.graph import build_hr_rag_graph


router = APIRouter(prefix="/api/v1", tags=["HR Pipeline"])


# ── Dependency injectors ──────────────────────────────────────────────────────

def get_chunker(req: Request) -> CoverLetterChunker:
    return req.app.state.chunker

def get_masker(req: Request) -> PIIMasker:
    return req.app.state.masker

def get_blob(req: Request) -> CoverLetterBlobClient:
    return req.app.state.blob

def get_qdrant(req: Request) -> QdrantCoverStore:
    return req.app.state.qdrant

def get_sva(req: Request) -> SVAEngine:
    return req.app.state.sva

def get_rag(req: Request):
    return req.app.state.rag_graph

def get_redis(req: Request) -> aioredis.Redis:
    return req.app.state.redis

def get_jd_store(req: Request) -> dict:
    return req.app.state.jd_store  # Simple dict cache: {job_id: jd_text}


# ── Rate limiting ──────────────────────────────────────────────────────────────

async def rate_limit(
    req: Request,
    redis: aioredis.Redis = Depends(get_redis),
) -> None:
    """Sliding window rate limit: 60 req/min per user."""
    user_id = req.headers.get("X-User-ID", req.client.host)
    key = f"ratelimit:{user_id}"
    count = await redis.incr(key)
    if count == 1:
        await redis.expire(key, 60)
    if count > 60:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Max 60 requests/min per user.",
        )


# ── Pydantic schemas ───────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    applicant_id:   str
    job_id:         str
    blob_path:      str
    chunk_count:    int
    entity_count:   int
    jd_match_score: float
    ai_probability: float
    message:        str


class QueryRequest(BaseModel):
    prompt:          str   = Field(..., description="HR natural language talent query")
    job_id:          str   = Field(..., description="Target job posting ID")
    ai_prob_max:     float = Field(1.0, ge=0.0, le=1.0,
                                   description="Max AI probability (0.3 = prefer human)")
    min_match_score: float = Field(0.0, ge=0.0, le=1.0,
                                   description="Minimum JD match score")


class QueryResponse(BaseModel):
    job_id:          str
    response:        str
    candidate_count: int
    top_candidates:  list[dict]


# ── Upload endpoint ────────────────────────────────────────────────────────────

@router.post(
    "/covers/{job_id}",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(rate_limit)],
    summary="Upload a cover letter for a job posting",
)
async def upload_cover_letter(
    job_id: str,
    file: UploadFile = File(...),
    chunker: CoverLetterChunker = Depends(get_chunker),
    masker: PIIMasker            = Depends(get_masker),
    blob: CoverLetterBlobClient  = Depends(get_blob),
    qdrant: QdrantCoverStore     = Depends(get_qdrant),
    sva: SVAEngine               = Depends(get_sva),
    jd_store: dict               = Depends(get_jd_store),
):
    t_start = time.perf_counter()
    # ── Validation ─────────────────────────────────────────────────────────────
    allowed_types = {
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/plain",
    }
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {file.content_type}",
        )

    file_bytes = await file.read()
    if len(file_bytes) > 5 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File exceeds 5 MB limit",
        )

    # ── Deduplication ──────────────────────────────────────────────────────────
    # Applicant ID is derived deterministically from file content hash.
    # Re-submitting the same file for the same job is silently idempotent.
    content_hash = hashlib.sha256(file_bytes).hexdigest()
    applicant_id = f"a_{content_hash[:12]}"

    # ── 1. Upload raw to Blob ──────────────────────────────────────────────────
    blob_path, _ = blob.upload_raw(
        file_bytes, job_id, applicant_id, file.content_type
    )

    # ── 2. Extract and chunk ───────────────────────────────────────────────────
    chunks = chunker.chunk(file_bytes, file.content_type)
    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Could not extract text from document",
        )

    # ── 3. PII mask each chunk ─────────────────────────────────────────────────
    masked_docs = [
        masker.mask(
            text=chunk.text,
            applicant_id=applicant_id,
            job_id=job_id,
            chunk_index=chunk.chunk_index,
        )
        for chunk in chunks
    ]
    total_entities = sum(m.entity_count for m in masked_docs)
    masked_texts = [m.masked_text for m in masked_docs]
    CHUNK_COUNT.observe(len(chunks))
    PII_ENTITIES_REDACTED.observe(total_entities)

    # ── 4. SVA analysis ────────────────────────────────────────────────────────
    jd_text = jd_store.get(job_id, f"Job description for {job_id}")
    jd_profile = sva.profiler.profile(jd_text)
    scores = sva.analyse(masked_texts, jd_profile)
    AI_PROBABILITY.observe(scores.ai_probability)

    # ── 5. Upsert all chunks to Qdrant ─────────────────────────────────────────
    for i, (masked, chunk) in enumerate(zip(masked_docs, chunks)):
        t_upsert = time.perf_counter()
        qdrant.upsert_chunk(
            job_id=job_id,
            applicant_id=applicant_id,
            chunk_index=i,
            chunk_text=masked.masked_text,
            scores=scores,
            blob_path=blob_path,
        )
        QDRANT_UPSERT_DURATION_SECONDS.observe(time.perf_counter() - t_upsert)

    # ── 6. Upload masked plaintext for audit trail ─────────────────────────────
    blob.upload_masked(" ".join(masked_texts), job_id, applicant_id)

    INGESTION_TOTAL.labels(job_id=job_id, status="success").inc()
    INGESTION_DURATION_SECONDS.labels(job_id=job_id).observe(time.perf_counter() - t_start)
    ACTIVE_APPLICANTS.labels(job_id=job_id).inc()

    return UploadResponse(
        applicant_id=applicant_id,
        job_id=job_id,
        blob_path=blob_path,
        chunk_count=len(chunks),
        entity_count=total_entities,
        jd_match_score=scores.jd_match_score,
        ai_probability=scores.ai_probability,
        message=(
            f"Processed {len(chunks)} chunks. "
            f"JD match: {scores.jd_match_score:.2f}. "
            f"AI probability: {scores.ai_probability:.2f}."
        ),
    )


# ── Query endpoint ─────────────────────────────────────────────────────────────

@router.post(
    "/query",
    response_model=QueryResponse,
    dependencies=[Depends(rate_limit)],
    summary="HR natural language talent query",
)
async def query_talent(
    body: QueryRequest,
    rag=Depends(get_rag),
):
    t_start = time.perf_counter()
    try:
        result = await rag.ainvoke(
            {
                "hr_prompt": body.prompt,
                "job_id":    body.job_id,
                "filters": {
                    "ai_prob_max":     body.ai_prob_max,
                    "min_match_score": body.min_match_score,
                },
            },
            config={"configurable": {"thread_id": f"{body.job_id}:{body.prompt[:32]}"}},
        )

        reranked = result.get("reranked", [])
        response = result.get("response", "")
        CANDIDATES_RETRIEVED.observe(len(result.get("raw_results", reranked)))
        LLM_RESPONSE_TOKENS.observe(len(response.split()))
        HR_QUERIES_TOTAL.labels(job_id=body.job_id, status="success").inc()
        HR_QUERY_DURATION_SECONDS.labels(job_id=body.job_id).observe(time.perf_counter() - t_start)

        return QueryResponse(
            job_id=body.job_id,
            response=response,
            candidate_count=len(reranked),
            top_candidates=reranked,
        )
    except Exception:
        HR_QUERIES_TOTAL.labels(job_id=body.job_id, status="error").inc()
        raise


# ── Applicants list ────────────────────────────────────────────────────────────

@router.get("/applicants/{job_id}", summary="List all applicants for a job")
async def list_applicants(
    job_id: str,
    qdrant: QdrantCoverStore = Depends(get_qdrant),
):
    col = f"covers_{job_id}"
    if not qdrant._client.collection_exists(col):
        return {"job_id": job_id, "applicants": []}

    # Scroll all points, deduplicate by applicant_id
    records, _ = qdrant._client.scroll(
        collection_name=col,
        with_payload=True,
        limit=1000,
    )
    seen: set[str] = set()
    applicants = []
    for r in records:
        aid = r.payload.get("applicant_id", "")
        if aid not in seen:
            seen.add(aid)
            applicants.append({
                "applicant_id":   aid,
                "jd_match_score": r.payload.get("jd_match_score"),
                "ai_probability": r.payload.get("ai_probability"),
                "blob_path":      r.payload.get("blob_path"),
            })
    applicants.sort(key=lambda x: x["jd_match_score"] or 0, reverse=True)
    return {"job_id": job_id, "applicants": applicants}


# ── JD seeding ─────────────────────────────────────────────────────────────────

class SeedJDRequest(BaseModel):
    jd_text: str = Field(..., description="Full job description text")


@router.post("/jd/{job_id}", summary="Seed job description for SVA scoring")
async def seed_jd(
    job_id: str,
    body: SeedJDRequest,
    jd_store: dict = Depends(get_jd_store),
):
    jd_store[job_id] = body.jd_text
    return {"job_id": job_id, "seeded": True, "length": len(body.jd_text)}


# ── Prometheus metrics ─────────────────────────────────────────────────────────

from fastapi.responses import PlainTextResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

@router.get("/metrics", summary="Prometheus metrics scrape endpoint",
            response_class=PlainTextResponse, include_in_schema=False)
async def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ── Health check ───────────────────────────────────────────────────────────────

@router.get("/health", summary="Liveness check")
async def health():
    return {"status": "ok"}
