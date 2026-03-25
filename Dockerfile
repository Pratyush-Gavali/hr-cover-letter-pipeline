# ── Build stage ────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System dependencies for pdfplumber (poppler) and spaCy (gcc)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc \
    poppler-utils libpoppler-cpp-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Runtime stage ──────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Runtime system libs (no compiler toolchain)
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Download spaCy model
RUN python -m spacy download en_core_web_lg

# Copy source
COPY src/ /app/

# Non-root user for security
RUN useradd --create-home --uid 1001 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Pre-warm: trigger BGE-M3 download on first build
# Comment out if images are built in an offline environment —
# the model will be downloaded on first container start instead.
# RUN python -c "from FlagEmbedding import BGEM3FlagModel; BGEM3FlagModel('BAAI/bge-m3')"

EXPOSE 8000

# Gunicorn for production (single worker per replica, use horizontal scaling)
# --workers 1: BGE-M3 is loaded once per process; multiple workers waste VRAM
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--loop", "uvloop", "--http", "httptools"]
