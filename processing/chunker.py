"""
Text extraction and semantic paragraph chunking.

Supports PDF (pdfplumber), DOCX (python-docx), and plain TXT.
Boilerplate openers are stripped before chunking to reduce embedding noise.

Why paragraph-level chunking (not fixed token-window)?
-------------------------------------------------------
A paragraph in a cover letter is a natural semantic unit — it typically
addresses exactly one theme: motivation, technical skills, a specific
achievement, cultural fit, or growth intent. Fixed-window chunking
(e.g. 256 tokens with 50-token overlap) breaks these units arbitrarily,
producing chunks that mix two themes, which degrades DTW accuracy
because the resulting embedding is the *average* of two semantic directions
rather than a clean signal for one JD phase.

Paragraph-level chunking produces 4-8 chunks for a typical cover letter —
exactly the right granularity for DTW alignment with a 5-10 phase JD.
"""

from __future__ import annotations
import re
import io
from dataclasses import dataclass

import pdfplumber
from docx import Document as DocxDocument


# ── Boilerplate patterns to strip ─────────────────────────────────────────────

BOILERPLATE_PATTERNS = [
    re.compile(r"^dear\s+(hiring|recruiter|team|sir|madam)[^\n]*\n", re.I | re.M),
    re.compile(r"^to whom it may concern[^\n]*\n", re.I | re.M),
    re.compile(r"^i am writing to (apply|express)[^\n]*\n", re.I | re.M),
    re.compile(r"^sincerely[,\s]*\n", re.I | re.M),
    re.compile(r"^kind\s?regards[,\s]*\n", re.I | re.M),
    re.compile(r"^yours\s+(faithfully|sincerely)[,\s]*\n", re.I | re.M),
    re.compile(r"^\[applicant name\][^\n]*\n", re.I | re.M),
]

MIN_CHUNK_CHARS = 80   # Below this: too short to embed meaningfully
MAX_CHUNK_CHARS = 1200 # Above this: split to avoid truncation at max_length=512


@dataclass
class TextChunk:
    text: str
    chunk_index: int
    char_start: int
    char_end: int


class CoverLetterChunker:
    """
    Extracts text from PDF / DOCX / TXT and returns semantic paragraph chunks.
    """

    def chunk(
        self,
        file_bytes: bytes,
        content_type: str,
    ) -> list[TextChunk]:
        """
        Extract and chunk a cover letter file.

        Parameters
        ----------
        file_bytes
            Raw file content.
        content_type
            MIME type: "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            or "text/plain".

        Returns
        -------
        list[TextChunk]
            Ordered paragraph chunks, boilerplate stripped.
        """
        raw = self._extract_text(file_bytes, content_type)
        cleaned = self._strip_boilerplate(raw)
        return self._paragraph_chunks(cleaned)

    # ── Extraction ─────────────────────────────────────────────────────────────

    def _extract_text(self, data: bytes, content_type: str) -> str:
        ct = content_type.lower()
        if "pdf" in ct:
            return self._extract_pdf(data)
        elif "wordprocessing" in ct or "docx" in ct:
            return self._extract_docx(data)
        else:
            return data.decode("utf-8", errors="replace")

    def _extract_pdf(self, data: bytes) -> str:
        parts: list[str] = []
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages:
                # x_tolerance/y_tolerance tuned for typical cover letter fonts
                text = page.extract_text(x_tolerance=2, y_tolerance=3)
                if text:
                    parts.append(text)
        return "\n\n".join(parts)

    def _extract_docx(self, data: bytes) -> str:
        doc = DocxDocument(io.BytesIO(data))
        lines: list[str] = []
        for para in doc.paragraphs:
            stripped = para.text.strip()
            if stripped:
                lines.append(stripped)
            elif lines and lines[-1] != "":
                lines.append("")  # Preserve paragraph break
        return "\n".join(lines)

    # ── Cleaning ───────────────────────────────────────────────────────────────

    def _strip_boilerplate(self, text: str) -> str:
        for pattern in BOILERPLATE_PATTERNS:
            text = pattern.sub("", text)
        # Normalise: 3+ newlines -> 2
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    # ── Chunking ───────────────────────────────────────────────────────────────

    def _paragraph_chunks(self, text: str) -> list[TextChunk]:
        """
        Split on double-newline paragraph boundaries.

        Strategy:
        1. Adjacent short paragraphs are merged to stay above MIN_CHUNK_CHARS
        2. Long paragraphs are split at sentence boundaries to stay below
           MAX_CHUNK_CHARS (avoids embedding truncation at 512 tokens)
        """
        raw_paras = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
        merged = self._merge_short(raw_paras)
        split = self._split_long(merged)

        chunks: list[TextChunk] = []
        cursor = 0
        for i, para in enumerate(split):
            if len(para) >= MIN_CHUNK_CHARS:
                chunks.append(TextChunk(
                    text=para,
                    chunk_index=i,
                    char_start=cursor,
                    char_end=cursor + len(para),
                ))
            cursor += len(para) + 2
        return chunks

    def _merge_short(self, paras: list[str]) -> list[str]:
        """Merge consecutive short paragraphs into single chunks."""
        merged: list[str] = []
        buffer = ""
        for para in paras:
            if buffer:
                candidate = buffer + " " + para
                if len(candidate) < MIN_CHUNK_CHARS * 2:
                    buffer = candidate
                else:
                    merged.append(buffer)
                    buffer = para
            else:
                buffer = para
        if buffer:
            merged.append(buffer)
        return merged

    def _split_long(self, paras: list[str]) -> list[str]:
        """Split paragraphs exceeding MAX_CHUNK_CHARS at sentence boundaries."""
        result: list[str] = []
        for para in paras:
            if len(para) <= MAX_CHUNK_CHARS:
                result.append(para)
                continue
            sentences = re.split(r"(?<=[.!?])\s+", para)
            chunk = ""
            for sent in sentences:
                if len(chunk) + len(sent) <= MAX_CHUNK_CHARS:
                    chunk = (chunk + " " + sent).strip()
                else:
                    if chunk:
                        result.append(chunk)
                    chunk = sent
            if chunk:
                result.append(chunk)
        return result
