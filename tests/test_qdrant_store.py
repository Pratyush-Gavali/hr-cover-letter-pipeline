"""
Tests for QdrantCoverStore — uses qdrant-client in-memory mode.
No external Qdrant server or GPU required.
BGE-M3 is mocked to return deterministic fake embeddings.
"""
import numpy as np
import pytest
from unittest.mock import MagicMock

from storage.qdrant_store import QdrantCoverStore
from sva.engine import SVAScores


def make_scores(**overrides) -> SVAScores:
    defaults = dict(
        jd_match_score=0.75, trajectory_dtw=0.22, thematic_coverage=0.80,
        domain_density=0.55, ai_probability=0.10, sem_velocity_sigma=0.12,
        lexical_burstiness=0.03, token_entropy_sigma=18.0, stylometric_drift=6.5,
    )
    defaults.update(overrides)
    return SVAScores(**defaults)


@pytest.fixture
def store():
    """
    QdrantCoverStore backed by in-memory Qdrant.
    BGE-M3 model is mocked — returns random vectors of correct shape.
    """
    mock_model = MagicMock()
    # Return numpy arrays (not lists) — the real BGE-M3 returns np.ndarray,
    # and upsert_chunk calls .tolist() on the result.
    mock_model.encode.return_value = {
        "dense_vecs":      [np.random.rand(1024)],
        "lexical_weights": [{1: 0.5, 42: 0.3, 100: 0.8}],
    }
    from qdrant_client import QdrantClient
    real_client = QdrantClient(":memory:")
    s = QdrantCoverStore(url=":memory:", api_key=None, embed_model=mock_model)
    s._client = real_client
    return s


class TestQdrantCoverStore:
    def test_ensure_collection_idempotent(self, store):
        store.ensure_collection("jd_test")
        store.ensure_collection("jd_test")  # Should not raise
        assert store._client.collection_exists("covers_jd_test")

    def test_upsert_returns_string_id(self, store):
        pid = store.upsert_chunk(
            job_id="jd_001", applicant_id="a_abc123", chunk_index=0,
            chunk_text="Experience with transformer fine-tuning.",
            scores=make_scores(), blob_path="jd_001/a_abc123/raw.pdf",
        )
        assert isinstance(pid, str)

    def test_make_id_is_deterministic(self):
        assert QdrantCoverStore._make_id("a_abc", 0) == QdrantCoverStore._make_id("a_abc", 0)
        assert QdrantCoverStore._make_id("a_abc", 0) != QdrantCoverStore._make_id("a_abc", 1)

    def test_make_id_fits_int64(self):
        pid = QdrantCoverStore._make_id("a_long_applicant_id_xyz", 99)
        assert 0 < pid < 2**63

    def test_collection_name_prefixed(self):
        assert QdrantCoverStore._col("jd_1138") == "covers_jd_1138"

    def test_multiple_chunks_same_applicant(self, store):
        for i in range(4):
            store.upsert_chunk(
                job_id="jd_multi", applicant_id="a_multi", chunk_index=i,
                chunk_text=f"Paragraph {i} discussing ML engineering skills.",
                scores=make_scores(), blob_path="jd_multi/a_multi/raw.pdf",
            )
        info = store._client.get_collection("covers_jd_multi")
        assert info.points_count == 4

    def test_ai_probability_filter_excludes_high_ai(self, store):
        """
        Insert one low-AI and one high-AI chunk.
        Search with ai_prob_max=0.3 should only return the human letter.
        Validates that payload pre-filtering works correctly.
        """
        store.upsert_chunk(
            job_id="jd_filter", applicant_id="a_human", chunk_index=0,
            chunk_text="Human-written cover letter content with varied vocabulary.",
            scores=make_scores(ai_probability=0.08),
            blob_path="jd_filter/a_human/raw.pdf",
        )
        store.upsert_chunk(
            job_id="jd_filter", applicant_id="a_robot", chunk_index=0,
            chunk_text="I am writing to express my strong interest in this position.",
            scores=make_scores(ai_probability=0.92),
            blob_path="jd_filter/a_robot/raw.pdf",
        )
        results = store.hybrid_search(
            job_id="jd_filter",
            query_text="machine learning engineer",
            ai_prob_max=0.3,
        )
        applicant_ids = [r["applicant_id"] for r in results]
        assert "a_human" in applicant_ids
        assert "a_robot" not in applicant_ids
