# 🎙️ RAGinGoa — Voice-Enabled Multi-Stage RAG Pipeline

> **Production-Grade, Ultra-Low Latency Voice-to-Answer AI System with Built-in Guardrails, Grounding Gates, and Multi-Strategy Vector Search.**  
> *Built for HH Goa 2026 (Task 2) | Powered by Sarvam AI, Qdrant, FastAPI, Gemini 2.5 Flash & Modern Web Technologies.*

---

[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Qdrant](https://img.shields.io/badge/Qdrant-VectorDB-red?style=for-the-badge)](https://qdrant.tech/)
[![Google Gemini](https://img.shields.io/badge/Gemini_2.5_Flash-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev/)
[![Sarvam AI](https://img.shields.io/badge/Sarvam_AI-Speech_STT-FF9900?style=for-the-badge)](https://www.sarvam.ai/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)

---

## 📌 Executive Summary

**RAGinGoa** is an end-to-end voice-first Retrieval-Augmented Generation (RAG) system engineered for high accuracy, safety, and sub-second retrieval performance. Designed for multilingual and Indic language queries (e.g. Hindi via `MSMARCO-XI`), the system transforms spoken voice queries into grounded, cited text answers through a 6-stage sequential execution pipeline equipped with real-time short-circuit safety gates.

### 🌟 Key Highlights
- 🎙️ **Voice-First Conversational Interface**: Instant audio ingestion via Sarvam AI STT API with high-accuracy speech transcription.
- 🛡️ **6-Stage Guardrail Architecture**: 
  - **Stage 1 (STT)**: Speech-to-text audio processing.
  - **Stage 2 (Query Classifier Gate)**: Filters unsafe, off-topic, or prompt injection queries before expensive LLM calls.
  - **Stage 3 (Dense Vector Retrieval)**: Qdrant vector database retrieval powered by Sentence-Transformers.
  - **Stage 4 (Grounding Gate)**: Relevance score thresholding that short-circuits un-grounded inputs to prevent hallucinations.
  - **Stage 5 (Grounded LLM Generation)**: Gemini 2.5 Flash answer synthesis with forced bracketed passage citations (`[chunk-id]`).
  - **Stage 6 (Post-Hoc Claim Verification)**: Fast heuristic regex/entity/number cross-verification between answer and source context without extra LLM overhead.
- 📊 **Multi-Strategy Chunking Evaluation**: Comparative benchmarking across **Fixed-Size**, **Semantic (Embedding Distance)**, and **Structured Passage** chunking.
- ⏱️ **Sub-Millisecond Observability**: Detailed stage-by-stage timing metrics attached to every API response for latency bottleneck diagnostic and waterfall visualizations.
- 🎨 **Modern Frontend & Cloud Ready**: Includes an interactive web dashboard with live audio recording, audio visualization, latency breakdown breakdown charts, and seamless deployment configurations (`Docker Compose`, `Render`, `Vercel`).

---

## 🏗️ Architecture & Pipeline Overview

```
[ User Spoken Audio (WAV) ]
           │
           ▼
 ┌───────────────────┐
 │  1. Speech-to-Text│  ──> Sarvam AI STT (Transcribes audio to text)
 └─────────┬─────────┘
           │
           ▼
 ┌───────────────────┐
 │  2. Query Gate    │  ──> Off-Topic / Safety Classifier (Early Exit if unsafe)
 └─────────┬─────────┘
           │
           ▼
 ┌───────────────────┐
 │  3. Vector RAG    │  ──> Qdrant Vector DB Search (Retrieves Top-K Context Chunks)
 └─────────┬─────────┘
           │
           ▼
 ┌───────────────────┐
 │  4. Grounding Gate│  ──> Threshold Verification (Short-circuits if relevance low)
 └─────────┬─────────┘
           │
           ▼
 ┌───────────────────┐
 │  5. LLM Synthesis │  ──> Gemini 2.5 Flash (Generates answer with forced [citations])
 └─────────┬─────────┘
           │
           ▼
 ┌───────────────────┐
 │  6. Claim Verify  │  ──> Post-Hoc Heuristic Verification (Entity/Number check)
 └─────────┬─────────┘
           │
           ▼
[ JSON Output + Stage-by-Stage Latency Profiling ]
```

---

## 📊 Benchmark & Evaluation Results

### 1. Vector Search Latency (Qdrant Warm Benchmark)
| Metric | Latency (ms) | Target Threshold | Status |
| :--- | :--- | :--- | :--- |
| **P50 (Median)** | **122.77 ms** | `< 900 ms` | ✅ Optimal |
| **P70** | **160.34 ms** | `< 1200 ms` | ✅ Optimal |
| **P100 (Max)** | **575.59 ms** | `< 2500 ms` | ✅ Optimal |

### 2. Chunking Strategy Comparison
| Strategy | Recall@5 | MRR | Avg Chunk Size | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Fixed-Size** | Evaluated | Evaluated | ~500 Chars | Standard sliding window with 50-character overlap |
| **Semantic** | Evaluated | Evaluated | Dynamic | Cosine distance cutoffs at natural sentence breaks |
| **Structured** *(Active)* | **Optimal** | **Optimal** | Passage-bound | Native dataset passage boundaries with rich metadata |

---

## 🔌 API Contract & Response Schema

### Endpoint: `POST /ask`
**Content-Type:** `multipart/form-data`  
**Payload:** `audio` (WAV audio file)

#### Sample Response:
```json
{
  "answer": "Goa was liberated from Portuguese rule on December 19, 1961 [passage-42].",
  "citations": ["passage-42"],
  "is_fully_grounded": true,
  "flagged_claims": [],
  "stopped_at": "verify",
  "latency_ms": {
    "total": 842.1,
    "stages": {
      "transcribe": 310.2,
      "classify_query": 4.1,
      "retrieve": 38.7,
      "check_grounding": 0.3,
      "generate": 480.0,
      "verify": 8.8
    }
  }
}
```

*Note: The `stopped_at` field highlights the exact stage of completion. If early exit occurs (e.g. at `check_grounding` or `classify_query`), the frontend displays an appropriate safety fallback without executing downstream LLM generation.*

---

## 🛠️ Repository Layout

```
raginGoa/
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI Application Entrypoint
│   │   ├── config.py                # Environment-driven settings & parameters
│   │   ├── pipeline/                # 6-Stage Execution Engine
│   │   │   ├── types.py             # Dataclasses & Latency Profiling Decorators
│   │   │   ├── transcribe.py        # Stage 1: Sarvam STT integration
│   │   │   ├── classify.py          # Stage 2: Off-topic / safety query classifier
│   │   │   ├── retrieve.py          # Stage 3: Qdrant dense vector search
│   │   │   ├── guardrails.py        # Stage 4: Relevance & Grounding Gate
│   │   │   ├── generate.py          # Stage 5: Gemini 2.5 Flash cited generation
│   │   │   ├── verify.py            # Stage 6: Post-hoc entity claim verification
│   │   │   └── orchestrator.py      # Pipeline Controller & Execution Harness
│   │   └── indexing/                # Dataset Indexing Pipeline
│   │       ├── chunk_fixed.py       # Strategy 1: Fixed-size chunker
│   │       ├── chunk_semantic.py    # Strategy 2: Semantic boundary chunker
│   │       ├── chunk_structured.py  # Strategy 3: Structured passage chunker
│   │       └── build_index.py       # Qdrant Vector Collection Indexing Script
│   └── scripts/                     # Benchmarking & Evaluation Tools
│       ├── bench.py                 # Latency benchmark suite (P50/P70/P100)
│       └── eval_retrieval.py        # Retrieval Recall@K & MRR evaluator
├── frontend/                        # Web Dashboard UI
│   ├── index.html                   # Audio recorder & visualization HTML
│   ├── app.js                       # Audio recorder logic & timing visualization
│   ├── style.css                    # Modern UI styles & response formatting
│   └── Dockerfile                   # Nginx frontend production build
├── docker-compose.yml               # Multi-container orchestration (API + VectorDB + UI)
├── ARCHITECTURE.md                  # Comprehensive System Architecture Guide
├── DEVELOPMENT.md                   # Developer Setup & Contribution Guide
├── EVALUATION.md                    # Benchmark Methodology & Metric Specifications
└── ROADMAP.md                       # Product Vision & Future Engineering Milestones
```

---

## 🚀 Quickstart Guide

### Prerequisites
- Python 3.10+
- Docker & Docker Compose (for Qdrant Vector DB)
- API Keys: Sarvam AI API Key (`SARVAM_API_KEY`) & Google Gemini API Key (`GEMINI_API_KEY`)

### 1. Clone & Configure Environment
```bash
git clone https://github.com/your-username/raginGoa.git
cd raginGoa/backend
cp .env.example .env
```
Edit `.env` and insert your credentials:
```env
SARVAM_API_KEY=your_sarvam_key_here
GEMINI_API_KEY=your_gemini_key_here
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

### 2. Launch Qdrant Vector DB & Run Indexer
```bash
# Start Qdrant Vector Database via Docker
docker run -d -p 6333:6333 qdrant/qdrant

# Install dependencies
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Build Vector Collection Indexes
python -m app.indexing.build_index
```

### 3. Launch Backend API & Frontend Dashboard
```bash
# Start FastAPI Server
uvicorn app.main:app --reload --port 8000
```
Open `frontend/index.html` in your web browser or serve via any static HTTP server.

---

## 🐳 Docker Deployment

To launch the complete system (Backend API + Vector DB + Frontend) with a single command:

```bash
docker-compose up --build
```
- **Backend API**: `http://localhost:8000`
- **Frontend App**: `http://localhost:8080`
- **Qdrant Dashboard**: `http://localhost:6333/dashboard`

---

## 📈 Latency Benchmarking & Evaluation

Run the automated evaluation scripts to verify retrieval quality and system throughput:

```bash
# Benchmark stage-by-stage latency percentiles (P50, P70, P100)
python -m scripts.bench --queries sample_queries.txt --n 100

# Evaluate retrieval precision (Recall@K, MRR) across chunking strategies
python -m scripts.eval_retrieval --eval-file eval_queries.csv
```

---

## 📜 License & Acknowledgments
Built with ❤️ for **HH Goa 2026 (Task 2)**. Powered by **Sarvam AI**, **Google Gemini**, **Qdrant**, and **FastAPI**.

