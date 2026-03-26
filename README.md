# Cover Letter Intelligence Pipeline

An end-to-end HR talent acquisition system that ingests cover letters, masks PII, scores candidates against a job description using Semantic Velocity Analysis (SVA), and surfaces ranked results through a natural language RAG interface.

---

## Architecture Overview

```
Cover Letter (PDF/DOCX/TXT)
        │
        ▼
  Text Extraction (pdfplumber / python-docx)
        │
        ▼
  Paragraph Chunking
        │
        ▼
  PII Masking (Presidio + spaCy en_core_web_lg)
        │
        ▼
  SVA Engine
  ├── JD Trajectory Alignment (DTW)
  ├── Thematic Coverage
  ├── Domain Density
  └── AI Authorship Detection (stylometric drift, token entropy, lexical burstiness)
        │
        ▼
  Qdrant Vector Store (BGE-M3 dense + BM25 sparse, per-job collection)
        │
        ▼
  HR Query → LangGraph RAG Pipeline
  └── query_parse → retrieve → CrossEncoder rerank → LLM synthesise
```

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.13 |
| Docker Desktop | Running |
| spaCy model | `en_core_web_lg` |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

### 2. Start infrastructure (Qdrant, Redis, Azurite)

```bash
docker compose up -d qdrant redis azurite
```

Wait for all three to be healthy:

```bash
docker compose ps
```

### 3. Configure environment variables

For local development, copy the values below into your shell or a `.env` file:

```bash
export QDRANT_URL=http://localhost:6333
export REDIS_URL=redis://localhost:6379
export AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;"

# Skip Azure Key Vault in local dev — uses an in-memory PII key store instead
export USE_LOCAL_PII=1

# LLM provider: stub | openai | ollama | azure
# "stub" returns a fixed response with no API key required
export LLM_PROVIDER=stub
```

**LLM provider options:**

| `LLM_PROVIDER` | Additional env vars required |
|---|---|
| `ollama` | Install Ollama, run `ollama pull llama3.1:8b`; optionally set `OLLAMA_BASE_URL` (default: `http://localhost:11434`) |
| `stub` | None — returns a fixed placeholder response, no API key required |
| `openai` | `OPENAI_API_KEY` |
| `azure` | `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT` |

**Recommended for local development:** `ollama` — fully offline, no API costs.

Install Ollama on macOS:
```bash
brew install ollama
brew services start ollama
ollama pull llama3.1:8b
```

### 4. Start the API server

```bash
uvicorn main:app --port 8000
```

Verify it is running:

```bash
curl http://localhost:8000/api/v1/health
# {"status":"ok"}
```

### 5. Start the Gradio UI

In a second terminal:

```bash
API_BASE=http://localhost:8000/api/v1 python ui.py
```

Open **http://localhost:7860** in your browser.

---

## Using the UI

The portal has four tabs:

### Job Setup
Register a job description before uploading cover letters. The SVA engine uses this text to score semantic alignment and thematic coverage.

1. Enter a **Job ID** (e.g. `job_ba_pnbank`)
2. Paste the full job description text
3. Click **Save Job Description**

### Upload Cover Letter
Upload a PDF, DOCX, or TXT cover letter. The pipeline extracts text, masks PII, runs SVA analysis, and stores the result in Qdrant.

Returns:
- **JD Match Score** — semantic similarity to the job description (0–1)
- **AI Probability** — likelihood the letter was AI-generated (0–1)
- **Human Confidence** — complement of AI probability
- **Chunks** — number of paragraphs extracted
- **PII Entities Masked** — count of names, dates, locations, etc. redacted

### Applicant Overview
Lists all indexed applicants for a job, sorted by JD match score, with authorship assessments.

### Talent Search
Ask a natural language question against the indexed applicant pool. The RAG pipeline parses the query, retrieves candidates from Qdrant via hybrid search, reranks with a CrossEncoder, and synthesises a grounded response using the configured LLM.

Filter controls:
- **Maximum AI Probability** — lower this to exclude likely AI-written submissions
- **Minimum JD Match Score** — raise this to show only high-relevance candidates

---

## API Reference

The FastAPI server exposes the following endpoints. Interactive docs are available at **http://localhost:8000/docs**.

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/jd/{job_id}` | Seed a job description |
| `POST` | `/api/v1/covers/{job_id}` | Upload and analyse a cover letter |
| `GET` | `/api/v1/applicants/{job_id}` | List applicants for a job |
| `POST` | `/api/v1/query` | Natural language talent search |
| `GET` | `/api/v1/metrics` | Prometheus metrics scrape endpoint |
| `GET` | `/api/v1/health` | Liveness check |

---

## Monitoring

### Prometheus

Start a local Prometheus instance:

```bash
docker run -d --name prometheus-local -p 9090:9090 \
  -v "$(pwd)/prometheus-local.yml:/etc/prometheus/prometheus.yml:ro" \
  prom/prometheus:v2.51.0 --config.file=/etc/prometheus/prometheus.yml
```

Open **http://localhost:9090**.

### Grafana

```bash
docker run -d --name grafana-local -p 3000:3000 \
  -e GF_SECURITY_ADMIN_PASSWORD=hrpipeline \
  grafana/grafana:10.4.0
```

Open **http://localhost:3000** (login: `admin` / `hrpipeline`), then:
1. Add a Prometheus data source pointing to `http://host.docker.internal:9090`
2. Import the dashboard JSON from `monitoring/grafana/dashboards/hr_pipeline.json`

---

## Running Tests

```bash
pytest
```

The test suite (42 tests) runs fully offline — no GPU, Qdrant server, or API keys required. BGE-M3 is mocked in store tests; the SVA engine and DTW tests use pure CPU computation.

---

## Project Structure

```
.
├── main.py                      # FastAPI app entry point + lifespan startup
├── ui.py                        # Gradio portal
├── requirements.txt
├── docker-compose.yml           # Qdrant, Redis, Azurite
│
├── api/
│   └── routes.py                # HTTP endpoints
│
├── sva/
│   ├── engine.py                # SVA scoring: DTW, AI detection, JD profiling
│   └── dtw.py                   # Dynamic Time Warping implementation
│
├── processing/
│   ├── chunker.py               # PDF/DOCX/TXT paragraph extraction
│   └── pii.py                   # Presidio PII masking (Azure KV + local mode)
│
├── storage/
│   ├── qdrant_store.py          # Hybrid vector store (BGE-M3 dense + BM25 sparse)
│   └── blob_client.py           # Azure Blob Storage client
│
├── rag/
│   ├── graph.py                 # LangGraph RAG pipeline
│   └── retriever.py             # Qdrant retrieval + CrossEncoder reranking
│
├── ingestion/
│   └── worker.py                # Async queue worker
│
├── monitoring/
│   ├── metrics.py               # Prometheus metric definitions
│   ├── middleware.py            # HTTP metrics middleware
│   └── prometheus.yml           # Prometheus scrape config
│
└── tests/
    ├── test_chunker.py
    ├── test_dtw.py
    ├── test_qdrant_store.py
    └── test_sva_engine.py
```
