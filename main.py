"""
FastAPI application entry point.

Uses lifespan context manager (FastAPI 0.95+) to initialise
all heavy resources once at startup and share them via app.state.

Resource initialisation order
-------------------------------
1. Redis connection (rate limiting)
2. Azure Blob client (file storage)
3. BGE-M3 embed model (shared by SVAEngine and QdrantCoverStore)
4. SVA engine (wraps BGE-M3)
5. Qdrant store (wraps BGE-M3)
6. PII masker (Presidio + spaCy + Key Vault)
7. Text chunker (stateless, cheap)
8. LangGraph RAG graph (wraps LLM + retriever)
9. Local queue worker (async background task)

All resources are shared across requests via dependency injection
from app.state — no per-request model loading.
"""

from __future__ import annotations
import asyncio
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from fastapi import FastAPI
from langgraph.checkpoint.memory import MemorySaver
from FlagEmbedding import BGEM3FlagModel

from api.routes import router
from ingestion.worker import LocalQueueWorker
from monitoring.middleware import PrometheusMiddleware
from processing.chunker import CoverLetterChunker
from processing.pii import PIIMasker, LocalPIIMasker
from rag.graph import build_hr_rag_graph
from rag.retriever import CoverLetterRetriever
from storage.blob_client import CoverLetterBlobClient
from storage.qdrant_store import QdrantCoverStore
from sva.engine import SVAEngine

import os


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ────────────────────────────────────────────────────────────────
    app.state.redis = aioredis.from_url(
        os.environ["REDIS_URL"], decode_responses=True
    )

    # CoverLetterBlobClient reads AZURE_STORAGE_CONNECTION_STRING from env
    app.state.blob = CoverLetterBlobClient()

    # BGE-M3 loaded once — shared by SVAEngine and QdrantCoverStore
    embed_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

    app.state.sva = SVAEngine()

    app.state.qdrant = QdrantCoverStore(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ.get("QDRANT_API_KEY"),
        embed_model=embed_model,
    )

    if os.environ.get("USE_LOCAL_PII"):
        app.state.masker = LocalPIIMasker()          # No Key Vault — local dev only
    else:
        app.state.masker = PIIMasker(
            key_vault_url=os.environ["AZURE_KEY_VAULT_URL"],
        )

    app.state.chunker = CoverLetterChunker()

    # JD text cache — seed from your JD database at startup
    app.state.jd_store: dict[str, str] = {}

    # LLM provider — only used by the /query endpoint.
    # LLM_PROVIDER=stub  → no credentials needed, returns a fixed message (local dev)
    # LLM_PROVIDER=openai → OPENAI_API_KEY required
    # LLM_PROVIDER=ollama → run `ollama pull llama3.1:8b` first
    # LLM_PROVIDER=azure  → AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_DEPLOYMENT required
    llm_provider = os.environ.get("LLM_PROVIDER", "azure")

    if llm_provider == "stub":
        from langchain_core.messages import AIMessage
        from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
        llm = FakeMessagesListChatModel(responses=[
            AIMessage(content="[Stub LLM] Set LLM_PROVIDER=openai or ollama for real responses.")
        ])
    elif llm_provider == "ollama":
        from langchain_ollama import ChatOllama  # pip install langchain-ollama
        llm = ChatOllama(
            model="llama3.1:8b",
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0,
        )
    elif llm_provider == "openai":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=os.environ["OPENAI_API_KEY"],
            temperature=0,
        )
    else:  # azure (default)
        from langchain_openai import AzureChatOpenAI
        llm = AzureChatOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
            api_version="2024-05-01-preview",
            temperature=0,
        )

    retriever = CoverLetterRetriever(store=app.state.qdrant)

    app.state.rag_graph = build_hr_rag_graph(
        llm=llm,
        retriever=retriever,
        checkpointer=MemorySaver(),
    )

    # Local async queue worker — drop-in for Azure Service Bus in dev
    worker = LocalQueueWorker(pipeline_fn=_make_pipeline_fn(app))
    worker_task = asyncio.create_task(worker.run())
    app.state.worker = worker
    app.state.queue = worker.queue  # Expose for enqueueing from routes

    yield

    # ── Shutdown ───────────────────────────────────────────────────────────────
    await worker.stop()
    worker_task.cancel()
    await app.state.redis.aclose()


def _make_pipeline_fn(app: FastAPI):
    """Create the pipeline function injected into the queue worker."""
    async def process_message(msg: dict) -> None:
        # Download raw file from Blob
        ext = msg["content_type"].split("/")[-1]
        file_bytes = app.state.blob.download_raw(
            job_id=msg["job_id"],
            applicant_id=msg["applicant_id"],
            ext=ext,
        )
        chunks = app.state.chunker.chunk(file_bytes, msg["content_type"])
        masked_docs = [
            app.state.masker.mask(
                text=c.text,
                applicant_id=msg["applicant_id"],
                job_id=msg["job_id"],
                chunk_index=c.chunk_index,
            )
            for c in chunks
        ]
        jd_text = app.state.jd_store.get(msg["job_id"], "")
        jd_profile = app.state.sva.profiler.profile(jd_text)
        scores = app.state.sva.analyse([m.masked_text for m in masked_docs], jd_profile)
        for i, (masked, chunk) in enumerate(zip(masked_docs, chunks)):
            app.state.qdrant.upsert_chunk(
                job_id=msg["job_id"],
                applicant_id=msg["applicant_id"],
                chunk_index=i,
                chunk_text=masked.masked_text,
                scores=scores,
                blob_path=msg["blob_uri"],
            )
    return process_message


app = FastAPI(
    title="HR Cover Letter Intelligence Pipeline",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(PrometheusMiddleware)
app.include_router(router)
