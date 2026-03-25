# tests/conftest.py
"""
Shared pytest fixtures and configuration.

Model loading (BGE-M3, distilgpt2, CrossEncoder) is mocked by default
so the test suite runs in CI without a GPU or network access.
Set INTEGRATION_TESTS=1 to use real models in integration tests.
"""
import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch


# ── Autouse mock: BGE-M3 ──────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def mock_bge_model(monkeypatch):
    """Replace BGE-M3 with a fast mock returning random 1024-d vectors."""
    if os.environ.get("INTEGRATION_TESTS"):
        yield  # Use real model in integration tests
        return

    mock = MagicMock()
    mock.encode.side_effect = lambda texts, **kw: {
        "dense_vecs":      [np.random.rand(1024).tolist() for _ in texts],
        "lexical_weights": [{i: float(i) / 100 for i in range(1, 11)} for _ in texts],
    }
    with patch("sva.engine._embed_model", return_value=mock):
        yield mock


# ── Autouse mock: distilgpt2 (token entropy) ──────────────────────────────────

@pytest.fixture(autouse=True)
def mock_ppl_model(monkeypatch):
    """Replace distilgpt2 perplexity calculation with random values."""
    if os.environ.get("INTEGRATION_TESTS"):
        yield
        return

    import torch

    mock_tokenizer = MagicMock()
    mock_tokenizer.side_effect = lambda text, **kw: {
        "input_ids": torch.randint(0, 50000, (1, 20))
    }

    mock_model = MagicMock()
    mock_loss = MagicMock()
    mock_loss.return_value = torch.tensor(2.5 + np.random.rand() * 2)
    mock_model.return_value.loss = mock_loss()

    with patch("sva.engine._ppl_model", return_value=(mock_tokenizer, mock_model)):
        yield
