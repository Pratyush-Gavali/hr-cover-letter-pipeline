"""
Semantic Velocity Analysis (SVA) Engine
========================================
Novel 2026 method combining:
  1. Per-paragraph trajectory embedding + DTW alignment to JD
  2. Four-dimensional AI authorship detection

Core insight
------------
Most HR tech encodes a cover letter as a single flat vector and computes
cosine similarity against the JD. SVA treats each document as an *ordered
trajectory through semantic space*, not a point. Dynamic Time Warping (DTW)
then measures how well the cover letter's narrative arc aligns with the JD's
thematic progression — capturing structure and intent, not just vocabulary.

AI detection exploits a consistent property of LLM-generated text: it is
*uniformly competent*. Human writers have bursts of domain vocabulary, topic
pivots, and stylometric variation. AI writers produce smooth, homogeneous
text across all four measured dimensions.
"""

from __future__ import annotations
import re
import math
import statistics
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from FlagEmbedding import BGEM3FlagModel
import torch

from .dtw import dtw_distance


# ── Singleton model loader ────────────────────────────────────────────────────

_EMBED_MODEL: BGEM3FlagModel | None = None
_PPL_TOKENIZER = None
_PPL_MODEL = None


def _embed_model() -> BGEM3FlagModel:
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        _EMBED_MODEL = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    return _EMBED_MODEL


def _ppl_model():
    """Small causal LM for token entropy estimation (perplexity proxy)."""
    global _PPL_TOKENIZER, _PPL_MODEL
    if _PPL_MODEL is None:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        model_id = "distilgpt2"
        _PPL_TOKENIZER = AutoTokenizer.from_pretrained(model_id)
        _PPL_MODEL = AutoModelForCausalLM.from_pretrained(model_id)
        _PPL_MODEL.eval()
    return _PPL_TOKENIZER, _PPL_MODEL


# ── Data containers ────────────────────────────────────────────────────────────

@dataclass
class SVAScores:
    """All per-document scores produced by the SVA engine."""

    # JD alignment
    jd_match_score: float         # Composite 0-1, higher = better match
    trajectory_dtw: float         # Raw DTW distance (lower = better alignment)
    thematic_coverage: float      # Fraction of JD themes addressed (0-1)
    domain_density: float         # Role-specific term density (0-1)

    # AI authorship detection
    ai_probability: float         # P(AI-written) output by logistic classifier
    sem_velocity_sigma: float     # sigma of |delta_embed| per sentence
    lexical_burstiness: float     # Variance of type-token ratio across windows
    token_entropy_sigma: float    # sigma of per-sentence perplexity
    stylometric_drift: float      # sigma of sentence-length distribution

    # Derived
    human_confidence: float = field(init=False)

    def __post_init__(self):
        self.human_confidence = round(1.0 - self.ai_probability, 4)


@dataclass
class JDProfile:
    """
    Decomposed job description ready for DTW alignment.
    Cache and reuse per job posting — profiling is expensive.
    """
    raw_text: str
    phases: list[str]             # Ordered thematic segments
    phase_embeddings: np.ndarray  # (n_phases, 1024) BGE-M3 dense
    required_terms: set[str]      # Domain-specific terms for density scoring


# ── JD Profiler ───────────────────────────────────────────────────────────────

class JDProfiler:
    """
    Decomposes a job description into ordered thematic phases and
    extracts domain-specific terminology for domain density scoring.
    """

    def profile(self, jd_text: str) -> JDProfile:
        phases = self._segment_phases(jd_text)
        embeddings = _embed_model().encode(
            phases, batch_size=8, max_length=512
        )["dense_vecs"]
        terms = self._extract_domain_terms(jd_text)
        return JDProfile(
            raw_text=jd_text,
            phases=phases,
            phase_embeddings=np.array(embeddings),
            required_terms=terms,
        )

    def _segment_phases(self, text: str) -> list[str]:
        """Split JD into paragraphs; cap at 10 phases."""
        paras = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 40]
        if not paras:
            chunk = max(1, len(text) // 5)
            paras = [text[i:i+chunk] for i in range(0, len(text), chunk)]
        return paras[:10]

    def _extract_domain_terms(self, text: str) -> set[str]:
        """Simple regex: extract PascalCase / camelCase / ALLCAPS tokens."""
        tokens = re.findall(
            r"\b[A-Z][a-zA-Z0-9+#.]{2,}\b|\b[a-z]+[A-Z][a-zA-Z0-9]+\b",
            text,
        )
        return {t.lower() for t in tokens}


# ── SVA Engine (core) ─────────────────────────────────────────────────────────

class SVAEngine:
    """
    Semantic Velocity Analysis engine.

    Usage
    -----
    engine = SVAEngine()
    jd_profile = engine.profiler.profile(jd_text)
    scores = engine.analyse(cover_letter_chunks, jd_profile)
    """

    def __init__(self):
        self.profiler = JDProfiler()

    # ── Public interface ───────────────────────────────────────────────────────

    def analyse(
        self,
        chunks: list[str],
        jd_profile: JDProfile,
    ) -> SVAScores:
        """
        Full SVA analysis for a single cover letter.

        Parameters
        ----------
        chunks
            Paragraph-level text chunks (PII already masked).
        jd_profile
            Pre-computed JD profile (cache and reuse per job posting).

        Returns
        -------
        SVAScores
            All scores ready to be persisted as Qdrant point payload.
        """
        chunk_embeddings: np.ndarray = np.array(
            _embed_model().encode(
                chunks, batch_size=8, max_length=512
            )["dense_vecs"]
        )  # (n_chunks, 1024)

        full_text = " ".join(chunks)
        sentences = self._split_sentences(full_text)

        # ── 1. JD trajectory matching ──────────────────────────────────────────
        dtw_dist = dtw_distance(chunk_embeddings, jd_profile.phase_embeddings)
        norm_dtw = dtw_dist / (len(chunks) + len(jd_profile.phases))
        thematic_cov = self._thematic_coverage(chunk_embeddings, jd_profile)
        domain_dens = self._domain_density(full_text, jd_profile.required_terms)
        jd_match = self._composite_match(norm_dtw, thematic_cov, domain_dens)

        # ── 2. AI authorship detection ─────────────────────────────────────────
        sem_vel_sigma = self._semantic_velocity_sigma(sentences)
        lex_burst = self._lexical_burstiness(sentences)
        tok_ent_sigma = self._token_entropy_sigma(sentences)
        stylo_drift = self._stylometric_drift(sentences)

        ai_prob = self._classify_ai(
            sem_vel_sigma, lex_burst, tok_ent_sigma, stylo_drift
        )

        return SVAScores(
            jd_match_score=round(jd_match, 4),
            trajectory_dtw=round(norm_dtw, 4),
            thematic_coverage=round(thematic_cov, 4),
            domain_density=round(domain_dens, 4),
            ai_probability=round(ai_prob, 4),
            sem_velocity_sigma=round(sem_vel_sigma, 4),
            lexical_burstiness=round(lex_burst, 4),
            token_entropy_sigma=round(tok_ent_sigma, 4),
            stylometric_drift=round(stylo_drift, 4),
        )

    # ── JD Matching helpers ────────────────────────────────────────────────────

    def _thematic_coverage(
        self, chunk_embs: np.ndarray, jd_profile: JDProfile
    ) -> float:
        """
        For each JD phase, check whether any chunk embedding exceeds a
        cosine similarity threshold (0.55). Returns fraction of phases covered.
        """
        threshold = 0.55
        covered = 0
        for phase_emb in jd_profile.phase_embeddings:
            sims = self._cosine_row(chunk_embs, phase_emb)
            if sims.max() >= threshold:
                covered += 1
        return covered / max(1, len(jd_profile.phase_embeddings))

    def _domain_density(self, text: str, required_terms: set[str]) -> float:
        """Ratio of JD-required terms present in the cover letter."""
        if not required_terms:
            return 0.0
        lower = text.lower()
        hit = sum(1 for t in required_terms if t in lower)
        return hit / len(required_terms)

    def _composite_match(
        self, norm_dtw: float, thematic_cov: float, domain_dens: float
    ) -> float:
        """
        Weighted composite. DTW is converted to 0-1 via exponential decay;
        lower DTW distance = higher similarity.
        """
        dtw_sim = math.exp(-norm_dtw * 3)
        return 0.4 * dtw_sim + 0.4 * thematic_cov + 0.2 * domain_dens

    # ── AI Detection helpers ───────────────────────────────────────────────────

    def _semantic_velocity_sigma(self, sentences: list[str]) -> float:
        """
        Sigma of |delta_embed| between consecutive sentences.

        AI text: smooth semantic transitions → low sigma.
        Human text: topic pivots, anecdotes, hedging → high sigma.
        """
        if len(sentences) < 3:
            return 0.0
        embs = np.array(
            _embed_model().encode(sentences, batch_size=16)["dense_vecs"]
        )
        deltas = [
            float(np.linalg.norm(embs[i+1] - embs[i]))
            for i in range(len(embs) - 1)
        ]
        return statistics.stdev(deltas) if len(deltas) > 1 else 0.0

    def _lexical_burstiness(self, sentences: list[str], window: int = 5) -> float:
        """
        Variance of type-token ratio (TTR) in sliding windows.

        AI text: evenly distributed vocabulary → low TTR variance.
        Human text: domain bursts + filler phrases → high TTR variance.
        """
        if len(sentences) < window:
            return 0.0
        ttrs: list[float] = []
        for i in range(len(sentences) - window + 1):
            tokens = " ".join(sentences[i:i+window]).lower().split()
            if tokens:
                ttrs.append(len(set(tokens)) / len(tokens))
        return statistics.variance(ttrs) if len(ttrs) > 1 else 0.0

    def _token_entropy_sigma(self, sentences: list[str]) -> float:
        """
        Sigma of per-sentence pseudo-perplexity under distilgpt2.

        AI text: near-optimal token predictions → uniformly low perplexity.
        Human text: unconventional word choices → high perplexity variance.
        """
        if len(sentences) < 3:
            return 0.0
        tokenizer, model = _ppl_model()
        ppl_scores: list[float] = []
        with torch.no_grad():
            for sent in sentences[:20]:  # Cap for performance
                enc = tokenizer(
                    sent, return_tensors="pt", truncation=True, max_length=64
                )
                if enc["input_ids"].shape[1] < 2:
                    continue
                out = model(**enc, labels=enc["input_ids"])
                ppl_scores.append(float(torch.exp(out.loss)))
        return statistics.stdev(ppl_scores) if len(ppl_scores) > 1 else 0.0

    def _stylometric_drift(self, sentences: list[str]) -> float:
        """
        Sigma of word-count per sentence.

        AI text: remarkably consistent sentence length.
        Human text: mix of punchy short sentences and long flowing ones.
        """
        lengths = [len(s.split()) for s in sentences if s.strip()]
        return statistics.stdev(lengths) if len(lengths) > 1 else 0.0

    def _classify_ai(
        self,
        sem_v: float,
        lex_b: float,
        tok_e: float,
        stylo: float,
    ) -> float:
        """
        Logistic classifier over the 4 SVA dimensions → P(AI-written).

        In production, replace the illustrative rule below with:
            import joblib
            clf = joblib.load('models/sva_classifier.pkl')
            return clf.predict_proba([[sem_v, lex_b, tok_e, stylo]])[0][1]

        The classifier should be trained on a labelled corpus of:
          - Human-written: SHRM cover letters, LinkedIn samples
          - AI-written:    GPT-4, Claude, Gemini generated letters
        """
        features = np.array([sem_v, lex_b, tok_e, stylo])
        # Expected human-range maxima for normalisation
        norms = np.array([0.15, 0.04, 25.0, 8.0])
        normed = np.clip(features / norms, 0, 1)
        # Weighted human-likeness score (high = human-like, low = AI-like)
        human_score = float(np.dot(normed, [0.35, 0.25, 0.25, 0.15]))
        # Sigmoid inversion: low human_score → high P(AI)
        p_ai = 1.0 / (1.0 + math.exp(6 * (human_score - 0.4)))
        return round(p_ai, 4)

    # ── Utilities ──────────────────────────────────────────────────────────────

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [p.strip() for p in parts if len(p.strip()) > 10]

    @staticmethod
    def _cosine_row(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
        m_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
        v_norm = vector / (np.linalg.norm(vector) + 1e-9)
        return m_norm @ v_norm
