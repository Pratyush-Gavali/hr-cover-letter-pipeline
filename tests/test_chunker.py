"""Tests for the cover letter text chunker."""
import pytest
from processing.chunker import CoverLetterChunker, MIN_CHUNK_CHARS, MAX_CHUNK_CHARS


SAMPLE_LETTER = b"""Dear Hiring Manager,

I am writing to apply for the Senior Machine Learning Engineer role.

My background spans five years of applied ML research and engineering.
During this time I built production-grade NLP pipelines serving millions of users.
The technical challenges I faced required deep understanding of transformer architectures.

I led the development of a retrieval-augmented generation system at my previous company.
This system reduced hallucination rates by 40 percent across customer-facing LLM features.
I collaborated closely with product teams to align technical outputs with business outcomes.

What draws me to your company is the emphasis on applied research.
Your recent work on sparse attention mechanisms directly influenced my own projects.
I believe my expertise in efficient inference and model quantisation would be immediately valuable.

Kind regards,
"""


@pytest.fixture
def chunker():
    return CoverLetterChunker()


class TestCoverLetterChunker:
    def test_plain_text_returns_chunks(self, chunker):
        chunks = chunker.chunk(SAMPLE_LETTER, "text/plain")
        assert len(chunks) >= 2

    def test_boilerplate_stripped(self, chunker):
        chunks = chunker.chunk(SAMPLE_LETTER, "text/plain")
        combined = " ".join(c.text.lower() for c in chunks)
        assert "dear hiring manager" not in combined
        assert "i am writing to apply" not in combined
        assert "kind regards" not in combined

    def test_chunk_indices_sequential(self, chunker):
        chunks = chunker.chunk(SAMPLE_LETTER, "text/plain")
        indices = [c.chunk_index for c in chunks]
        assert indices == sorted(indices)

    def test_chunks_above_min_length(self, chunker):
        for chunk in chunker.chunk(SAMPLE_LETTER, "text/plain"):
            assert len(chunk.text) >= MIN_CHUNK_CHARS

    def test_long_paragraph_split_below_max(self, chunker):
        long_para = ("The quick brown fox jumps over the lazy dog. " * 60).encode()
        letter = b"Opening paragraph with substantial content.\n\n" + long_para
        for chunk in chunker.chunk(letter, "text/plain"):
            assert len(chunk.text) <= MAX_CHUNK_CHARS * 1.1  # 10% buffer

    def test_char_start_end_sensible(self, chunker):
        for chunk in chunker.chunk(SAMPLE_LETTER, "text/plain"):
            assert chunk.char_start >= 0
            assert chunk.char_end > chunk.char_start

    def test_empty_document_returns_empty_list(self, chunker):
        assert chunker.chunk(b"", "text/plain") == []

    def test_unknown_mime_falls_back_to_text(self, chunker):
        result = chunker.chunk(SAMPLE_LETTER, "application/octet-stream")
        assert isinstance(result, list)
