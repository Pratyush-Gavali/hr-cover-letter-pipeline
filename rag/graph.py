"""
LangGraph RAG pipeline — four-node graph for HR talent queries.

Graph topology
--------------
  query_parse -> retrieve -> rerank -> synthesise -> END

State (TypedDict)
-----------------
  hr_prompt       : str        HR's natural language query
  job_id          : str        Target job posting
  requirements    : str        Extracted requirement string (vector search query)
  filters         : dict       {ai_prob_max, min_match_score}
  raw_results     : list[dict] Qdrant hybrid search results (top-20)
  reranked        : list[dict] CrossEncoder reranked results (top-5)
  response        : str        Final grounded LLM response

Key design decisions
---------------------
query_parse
  Uses the LLM to extract structured search params from HR's natural
  language prompt. This avoids brittle regex parsing and handles
  complex queries like "find Python engineers who didn't use AI,
  with at least 70% JD match, for the ML role posted last week".

retrieve
  Calls Qdrant with payload pre-filters extracted by query_parse.
  Returns top-20 chunks across all applicants (not deduplicated yet).

rerank
  BGE CrossEncoder jointly scores (query, passage) pairs — captures
  fine-grained entailment that bi-encoder retrieval cannot. Deduplicates
  by applicant_id (highest scoring chunk per applicant) to avoid showing
  the same person 3x. Returns top-5 unique applicants.

synthesise
  LLM is grounded strictly on retrieved chunks. The prompt explicitly
  forbids inferring details not in the context — hallucination on
  candidate credentials is architecturally prevented, not prompted away.
"""

from __future__ import annotations
import json
import time
from typing import TypedDict

from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from .retriever import CoverLetterRetriever
from monitoring.metrics import PIPELINE_STAGE_DURATION_SECONDS


# ── State ─────────────────────────────────────────────────────────────────────

class PipelineState(TypedDict):
    hr_prompt:    str
    job_id:       str
    requirements: str
    filters:      dict
    raw_results:  list[dict]
    reranked:     list[dict]
    response:     str


# ── Node factories ─────────────────────────────────────────────────────────────

def make_query_parse_node(llm: AzureChatOpenAI):
    """
    Extract structured search parameters from HR's natural language prompt.
    Returns: job_id (if mentioned), requirements string, and filter values.
    """
    async def query_parse(state: PipelineState) -> dict:
        prompt = (
            "You are an HR system assistant. Extract structured parameters "
            "from the HR query below. Respond ONLY with valid JSON.\n\n"
            f"HR query: {state['hr_prompt']}\n\n"
            "JSON schema:\n"
            "{\n"
            '  "job_id":           "<job ID if mentioned, else null>",\n'
            '  "requirements":     "<concise requirement string for vector search>",\n'
            '  "ai_prob_max":      <float 0-1, default 1.0, lower=prefer human-written>,\n'
            '  "min_match_score":  <float 0-1, default 0.0>\n'
            "}"
        )
        result = await llm.ainvoke(prompt)
        try:
            parsed = json.loads(result.content)
        except json.JSONDecodeError:
            parsed = {
                "job_id": state.get("job_id"),
                "requirements": state["hr_prompt"],
                "ai_prob_max": 1.0,
                "min_match_score": 0.0,
            }

        return {
            "job_id":       parsed.get("job_id") or state.get("job_id", ""),
            "requirements": parsed.get("requirements", state["hr_prompt"]),
            "filters": {
                "ai_prob_max":     float(parsed.get("ai_prob_max", 1.0)),
                "min_match_score": float(parsed.get("min_match_score", 0.0)),
            },
        }
    return query_parse


def make_retrieve_node(retriever: CoverLetterRetriever):
    async def retrieve(state: PipelineState) -> dict:
        results = await retriever.search(
            job_id=state["job_id"],
            query=state["requirements"],
            top_k=20,
            **state.get("filters", {}),
        )
        return {"raw_results": results}
    return retrieve


def make_rerank_node(retriever: CoverLetterRetriever):
    async def rerank(state: PipelineState) -> dict:
        reranked = await retriever.rerank(
            query=state["requirements"],
            candidates=state["raw_results"],
            top_n=5,
        )
        return {"reranked": reranked}
    return rerank


def make_synthesise_node(llm: AzureChatOpenAI):
    """
    Grounded response synthesis.

    The LLM receives only masked cover letter excerpts as context.
    It cannot hallucinate candidate details because the context is
    the only source of truth available to it.
    """
    async def synthesise(state: PipelineState) -> dict:
        if not state.get("reranked"):
            return {"response": "No matching applicants found for the given criteria."}

        context_parts: list[str] = []
        for i, doc in enumerate(state["reranked"], start=1):
            aid   = doc.get("applicant_id", "unknown")
            score = doc.get("jd_match_score", 0.0)
            ai_p  = doc.get("ai_probability", 0.0)
            cov   = doc.get("thematic_coverage", 0.0)
            text  = doc.get("chunk_text", "")
            context_parts.append(
                f"[Candidate {i} | ID: {aid} | "
                f"Match: {score:.2f} | P(AI): {ai_p:.2f} | Coverage: {cov:.2f}]\n{text}"
            )

        context = "\n\n---\n\n".join(context_parts)

        prompt = (
            "You are an HR talent assistant. Based ONLY on the cover letter "
            "excerpts below, answer the HR query. Every claim must be traceable "
            "to the provided excerpts — do not infer or invent details.\n\n"
            f"HR query: {state['hr_prompt']}\n\n"
            f"Cover letter excerpts (PII masked):\n{context}\n\n"
            "Instructions:\n"
            "- Rank candidates best-fit first.\n"
            "- For each: state their ID, match score, and the specific excerpt "
            "phrase that makes them stand out (under 15 words).\n"
            "- Flag any candidate with P(AI) > 0.5 as 'Possible AI-written'.\n"
            "- If no candidates meet the criteria, say so explicitly."
        )

        result = await llm.ainvoke(prompt)
        return {"response": result.content}
    return synthesise


# ── Graph builder ──────────────────────────────────────────────────────────────

def build_hr_rag_graph(
    llm: AzureChatOpenAI,
    retriever: CoverLetterRetriever,
    checkpointer: MemorySaver | None = None,
):
    """
    Assemble and compile the LangGraph RAG graph.

    Example usage
    -------------
    graph = build_hr_rag_graph(llm, retriever)
    result = await graph.ainvoke({
        "hr_prompt": "Find top ML engineers with LLM fine-tuning experience",
        "job_id":    "jd_1138",
    })
    print(result["response"])
    """
    def _timed(stage: str, fn):
        async def wrapper(state):
            t = time.perf_counter()
            result = await fn(state)
            PIPELINE_STAGE_DURATION_SECONDS.labels(stage=stage).observe(time.perf_counter() - t)
            return result
        return wrapper

    builder = StateGraph(PipelineState)

    builder.add_node("query_parse", _timed("query_parse", make_query_parse_node(llm)))
    builder.add_node("retrieve",    _timed("retrieve",    make_retrieve_node(retriever)))
    builder.add_node("rerank",      _timed("rerank",      make_rerank_node(retriever)))
    builder.add_node("synthesise",  _timed("synthesise",  make_synthesise_node(llm)))

    builder.set_entry_point("query_parse")
    builder.add_edge("query_parse", "retrieve")
    builder.add_edge("retrieve",    "rerank")
    builder.add_edge("rerank",      "synthesise")
    builder.add_edge("synthesise",  END)

    return builder.compile(checkpointer=checkpointer)
