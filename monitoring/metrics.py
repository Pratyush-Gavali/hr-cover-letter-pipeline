"""
monitoring/metrics.py

All Prometheus metrics for the HR talent search copilot.
Import from here — never instantiate metrics elsewhere.

Three layers:
  HTTP layer      — request counts, latency, errors
  Pipeline layer  — ingestion, AI detection, DTW, query
  System layer    — BGE-M3 inference, Qdrant ops, queue depth
"""

from prometheus_client import Counter, Histogram, Gauge

# ── HTTP ───────────────────────────────────────────────────────────────────────

HTTP_REQUESTS_TOTAL = Counter(
    "hr_pipeline_http_requests_total",
    "Total HTTP requests by method, endpoint and status",
    ["method", "endpoint", "status_code"],
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "hr_pipeline_http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

# ── Ingestion ──────────────────────────────────────────────────────────────────

INGESTION_TOTAL = Counter(
    "hr_pipeline_ingestion_total",
    "Cover letters processed",
    ["job_id", "status"],           # status: success | error | duplicate
)

INGESTION_DURATION_SECONDS = Histogram(
    "hr_pipeline_ingestion_duration_seconds",
    "End-to-end ingestion latency per cover letter",
    ["job_id"],
    buckets=[0.5, 1.0, 2.0, 4.0, 8.0, 15.0, 30.0],
)

CHUNK_COUNT = Histogram(
    "hr_pipeline_chunk_count",
    "Paragraph chunks extracted per cover letter",
    buckets=[1, 2, 3, 4, 5, 6, 8, 10, 15],
)

PII_ENTITIES_REDACTED = Histogram(
    "hr_pipeline_pii_entities_redacted",
    "PII entities redacted per cover letter",
    buckets=[0, 1, 2, 3, 5, 8, 12, 20],
)

# ── AI detection ───────────────────────────────────────────────────────────────

AI_PROBABILITY = Histogram(
    "hr_pipeline_ai_probability",
    "Distribution of P(AI-written) across ingested letters",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

AI_DETECTION_DURATION_SECONDS = Histogram(
    "hr_pipeline_ai_detection_duration_seconds",
    "AIDetector.detect() latency per letter",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
)

# ── Query / RAG ────────────────────────────────────────────────────────────────

HR_QUERIES_TOTAL = Counter(
    "hr_pipeline_hr_queries_total",
    "Total HR talent queries",
    ["job_id", "status"],           # status: success | error
)

HR_QUERY_DURATION_SECONDS = Histogram(
    "hr_pipeline_hr_query_duration_seconds",
    "End-to-end HR query latency",
    ["job_id"],
    buckets=[0.5, 1.0, 2.0, 4.0, 8.0, 15.0],
)

PIPELINE_STAGE_DURATION_SECONDS = Histogram(
    "hr_pipeline_stage_duration_seconds",
    "Per-stage LangGraph latency",
    ["stage"],                      # query_parse | retrieve | rerank | synthesise
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

CANDIDATES_RETRIEVED = Histogram(
    "hr_pipeline_candidates_retrieved",
    "Raw candidates from Qdrant before DTW reranking",
    buckets=[0, 5, 10, 15, 20, 25, 30],
)

DTW_COMPOSITE_SCORE = Histogram(
    "hr_pipeline_dtw_composite_score",
    "DTW composite score for top-ranked candidates",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

LLM_RESPONSE_TOKENS = Histogram(
    "hr_pipeline_llm_response_tokens",
    "Approximate token count of LLM synthesised response",
    buckets=[50, 100, 200, 300, 500, 800, 1200],
)

# ── ML model ───────────────────────────────────────────────────────────────────

BGE_INFERENCE_DURATION_SECONDS = Histogram(
    "hr_pipeline_bge_inference_duration_seconds",
    "BGE-M3 embedding latency per batch",
    ["call_site"],                  # ingestion | rerank | retrieve
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
)

QDRANT_UPSERT_DURATION_SECONDS = Histogram(
    "hr_pipeline_qdrant_upsert_duration_seconds",
    "Qdrant upsert latency per chunk",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
)

# ── Gauges (point-in-time values) ──────────────────────────────────────────────

ACTIVE_APPLICANTS = Gauge(
    "hr_pipeline_active_applicants",
    "Indexed applicants per job",
    ["job_id"],
)

QUEUE_DEPTH = Gauge(
    "hr_pipeline_queue_depth",
    "Pending tasks in the local ingestion queue",
)
