# LitLens — Literature Intelligence for Researchers

A multi-agent research platform that helps PhD students and academic researchers synthesize 10–50 research papers, find contradictions, score evidence, identify gaps, and generate a literature review draft automatically.

![Architecture](assets/litlens_architecture.png)

## What It Does

Upload your research papers, define your research question, and LitLens delivers:

| Tab | What You Get |
|-----|-------------|
| **Overview** | Paper count, claim count, contradiction count, gap count, executive summary |
| **Contradictions** | Where papers disagree — with severity ratings and explanations |
| **Methodology** | Side-by-side comparison of study designs across all papers |
| **Evidence** | Every claim scored 0–100 based on supporting evidence quality |
| **Gaps** | What's missing in the literature, tied to your research question |
| **Literature Review** | A thematic, citation-ready draft you can edit and download |
| **Chat** | Ask follow-up questions grounded in your uploaded papers (RAG) |

## Architecture

- **Frontend**: React (single-file `LitLens.jsx`) with dark theme, 6 analysis tabs, drag-and-drop upload, and RAG chat
- **Backend**: FastAPI serving a LangGraph pipeline with 8 specialized agents
- **Vector Store**: FAISS with OpenAI embeddings for document-grounded responses
- **Models**: `gpt-4o-mini` for all agents — fast and cost-effective (~$0.01 per analysis)

### Agent Pipeline

```
PDF Upload → Text Extraction → FAISS Index
                                    ↓
Agent 1: Paper Ingestion (parallel, 12 threads)
Agent 2: Claim Extraction (parallel batches)
Agent 3+4+5: Contradictions + Methodology + Evidence (parallel)
Agent 6+7: Gap Analysis + Literature Review (parallel)
Agent 8: Report Generator
```

## Setup

### Prerequisites

- Python 3.10+
- Node.js 18+
- OpenAI API key

### Installation

```bash
git clone https://github.com/sjagannathan17/LitLens.git
cd LitLens

# Backend
pip install -r backend/requirements.txt

# Frontend
cd frontend && npm install && cd ..

# API key
echo "OPENAI_API_KEY=your-key-here" > .env
```

### Running

Start both servers:

```bash
# Terminal 1 — Backend
cd backend && uvicorn api:app --host 0.0.0.0 --port 8000

# Terminal 2 — Frontend
cd frontend && npx vite --host 0.0.0.0 --port 5173
```

Open **http://localhost:5173** in your browser.

## Usage

1. Upload 2+ research papers (PDF) via drag-and-drop
2. Enter your research question (e.g., "What are the key advances in large language models?")
3. Enter research domain (e.g., "AI", "Public Health")
4. Click **Analyze Literature**
5. Explore results across 6 tabs
6. Use the chat at the bottom to ask follow-up questions about your papers

## Project Structure

```
LitLens/
├── backend/
│   ├── api.py              # FastAPI server with /api/analyze and /api/chat
│   ├── pipeline.py          # 8 LangGraph agents, FAISS, pipeline runner
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── LitLens.jsx      # Complete React UI (single file)
│   │   └── main.jsx
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
├── midterm_solution (2).py   # Original Streamlit version (legacy)
├── .env                      # OpenAI API key (not committed)
└── README.md
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 19, Vite 6 |
| Backend | FastAPI, Uvicorn |
| Agent Framework | LangGraph, LangChain |
| LLM | OpenAI gpt-4o-mini |
| Embeddings | text-embedding-3-small |
| Vector Store | FAISS |
| PDF Parsing | PyPDFLoader |

## Cost

Each analysis run costs approximately **$0.01–$0.07** depending on paper count and length, using gpt-4o-mini for all agents.

## Disclaimer

LitLens is a research aid. Always verify claims, citations, and conclusions independently before academic use.
