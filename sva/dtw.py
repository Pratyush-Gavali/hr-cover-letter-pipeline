"""
Dynamic Time Warping (DTW) for embedding sequence alignment.

Used by SVA to measure how well a cover letter's semantic trajectory
aligns with the job description's thematic progression.

Why DTW over plain cosine similarity
--------------------------------------
Modern HR professionals value a compelling narrative flow in cover letters. 
DTW captures this by allowing flexible "warping" of the cover letter's embedding sequence to
best match the JD's embedding sequence, rewarding coherent storytelling over scattered keyword matching.
"""

from __future__ import annotations
import numpy as np


def dtw_distance(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    dist_fn: str = "cosine",
) -> float:
    """
    Compute DTW distance between two embedding sequences.

    Parameters
    ----------
    seq_a : (n, d) ndarray
        Cover letter paragraph embeddings.
    seq_b : (m, d) ndarray
        JD phase embeddings.
    dist_fn : {"cosine", "euclidean"}
        Distance function for pairwise costs.

    Returns
    -------
    float
        DTW alignment distance. Lower = better narrative alignment.
    """
    n, m = len(seq_a), len(seq_b)
    if n == 0 or m == 0:
        return float("inf")

    cost_matrix = _pairwise_cost(seq_a, seq_b, dist_fn)
    dtw = _accumulate(cost_matrix, n, m)
    return float(dtw[n - 1, m - 1])


def _pairwise_cost(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    dist_fn: str,
) -> np.ndarray:
    """Build the n x m cost matrix between all pairs of embeddings."""
    if dist_fn == "cosine":
        # Cosine distance = 1 - cosine_similarity
        a_norm = seq_a / (np.linalg.norm(seq_a, axis=1, keepdims=True) + 1e-9)
        b_norm = seq_b / (np.linalg.norm(seq_b, axis=1, keepdims=True) + 1e-9)
        sim = a_norm @ b_norm.T          # (n, m) cosine similarities
        return 1.0 - sim                 # Cosine distances
    else:
        # Euclidean distance via broadcasting
        diff = seq_a[:, np.newaxis, :] - seq_b[np.newaxis, :, :]  # (n, m, d)
        return np.linalg.norm(diff, axis=2)                         # (n, m)


def _accumulate(cost: np.ndarray, n: int, m: int) -> np.ndarray:
    """Standard DP accumulation."""
    INF = float("inf")
    dtw = np.full((n, m), INF)
    dtw[0, 0] = cost[0, 0]

    for i in range(1, n):
        dtw[i, 0] = dtw[i - 1, 0] + cost[i, 0]
    for j in range(1, m):
        dtw[0, j] = dtw[0, j - 1] + cost[0, j]

    for i in range(1, n):
        for j in range(1, m):
            dtw[i, j] = cost[i, j] + min(
                dtw[i - 1, j],      # insertion
                dtw[i, j - 1],      # deletion
                dtw[i - 1, j - 1],  # match
            )
    return dtw


def dtw_path(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    dist_fn: str = "cosine",
) -> list[tuple[int, int]]:
    """
    Return the optimal warping path for interpretability.
    Each tuple (i, j) means cover letter chunk i aligns to JD phase j.
    Useful for generating HR-facing explanations of why a letter scored highly.
    """
    n, m = len(seq_a), len(seq_b)
    cost_matrix = _pairwise_cost(seq_a, seq_b, dist_fn)
    dtw = _accumulate(cost_matrix, n, m)

    path: list[tuple[int, int]] = []
    i, j = n - 1, m - 1
    while i > 0 or j > 0:
        path.append((i, j))
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            move = np.argmin([dtw[i-1, j-1], dtw[i-1, j], dtw[i, j-1]])
            if move == 0:
                i -= 1; j -= 1
            elif move == 1:
                i -= 1
            else:
                j -= 1
    path.append((0, 0))
    return list(reversed(path))
