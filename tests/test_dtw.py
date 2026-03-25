"""Tests for the DTW implementation — pure NumPy, no model loading."""
import numpy as np
import pytest
from sva.dtw import dtw_distance, dtw_path


class TestDTWDistance:
    def test_identical_sequences_near_zero(self):
        seq = np.random.rand(4, 8)
        assert dtw_distance(seq, seq) == pytest.approx(0.0, abs=1e-6)

    def test_empty_sequence_returns_inf(self):
        a = np.random.rand(4, 8)
        b = np.zeros((0, 8))
        assert dtw_distance(a, b) == float("inf")

    def test_unequal_length_sequences(self):
        a = np.random.rand(5, 8)
        b = np.random.rand(3, 8)
        dist = dtw_distance(a, b)
        assert 0.0 <= dist < float("inf")

    def test_cosine_and_euclidean_both_finite_non_negative(self):
        a = np.random.rand(4, 16)
        b = np.random.rand(4, 16)
        assert dtw_distance(a, b, dist_fn="cosine") >= 0
        assert dtw_distance(a, b, dist_fn="euclidean") >= 0

    def test_single_element_sequences(self):
        a = np.array([[1.0, 0.0, 0.0]])
        b = np.array([[0.0, 1.0, 0.0]])
        dist = dtw_distance(a, b)
        assert 0.0 < dist <= 2.0

    def test_aligned_lower_than_misaligned(self):
        """
        A cover letter that follows the JD's thematic order should produce
        lower DTW distance than one that addresses themes in reverse order.
        This validates the key SVA thesis.
        """
        np.random.seed(0)
        d = 16

        # JD: 4 distinct orthogonal phases
        jd = np.eye(4, d)
        jd /= np.linalg.norm(jd, axis=1, keepdims=True)

        aligned   = jd + np.random.rand(4, d) * 0.05
        aligned  /= np.linalg.norm(aligned, axis=1, keepdims=True)

        misaligned   = jd[::-1] + np.random.rand(4, d) * 0.3
        misaligned  /= np.linalg.norm(misaligned, axis=1, keepdims=True)

        assert dtw_distance(aligned, jd) < dtw_distance(misaligned, jd)


class TestDTWPath:
    def test_starts_at_origin(self):
        a = np.random.rand(3, 8)
        b = np.random.rand(3, 8)
        assert dtw_path(a, b)[0] == (0, 0)

    def test_ends_at_terminal(self):
        a = np.random.rand(4, 8)
        b = np.random.rand(3, 8)
        assert dtw_path(a, b)[-1] == (3, 2)

    def test_monotonically_non_decreasing(self):
        a = np.random.rand(5, 8)
        b = np.random.rand(4, 8)
        path = dtw_path(a, b)
        for k in range(1, len(path)):
            assert path[k][0] >= path[k-1][0]
            assert path[k][1] >= path[k-1][1]
