"""
LitLens FastAPI Backend

Exposes the LangGraph pipeline as REST endpoints consumed by the React frontend.
    uvicorn api:app --host 0.0.0.0 --port 8000
"""

import os
import json
import asyncio
import tempfile
import shutil
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pipeline import (
    build_vector_store,
    extract_raw_texts_from_paths,
    run_pipeline,
    set_pipeline_data,
    transform_result_for_frontend,
    rag_answer,
    get_vector_store,
)

app = FastAPI(title="LitLens API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_executor = ThreadPoolExecutor(max_workers=2)


class ChatRequest(BaseModel):
    question: str
    research_question: Optional[str] = ""


def _run_analysis(file_paths, research_question, research_domain):
    """Runs the full pipeline synchronously — called in a thread."""
    raw_texts = extract_raw_texts_from_paths(file_paths)
    print(f"[API] Extracted text from {len(raw_texts)} files")

    vs = build_vector_store(raw_texts)
    print(f"[API] FAISS vector_store={'BUILT' if vs else 'FAILED'}")

    set_pipeline_data(raw_texts, vs)

    final = run_pipeline(research_question, research_domain)
    return transform_result_for_frontend(final)


@app.post("/api/analyze")
async def analyze(
    files: list[UploadFile] = File(...),
    research_question: str = Form(...),
    research_domain: str = Form("General"),
):
    print(f"\n{'='*60}")
    print(f"[API] /api/analyze — rq={research_question!r}, domain={research_domain!r}")
    print(f"[API] files={[f.filename for f in files]}")
    print(f"{'='*60}")

    tmp_dir = tempfile.mkdtemp()
    try:
        file_paths = []
        for f in files:
            path = os.path.join(tmp_dir, f.filename)
            with open(path, "wb") as out:
                content = await f.read()
                out.write(content)
            file_paths.append({"path": path, "filename": f.filename})

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor, _run_analysis, file_paths, research_question, research_domain
        )

        print(f"[API] Returning — papers={len(result.get('papers', []))}, "
              f"claims={len(result.get('claims', []))}")

        body = json.dumps(result, default=str)
        return JSONResponse(content=json.loads(body))

    except Exception as e:
        print(f"[API] ERROR: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/api/chat")
async def chat(req: ChatRequest):
    loop = asyncio.get_event_loop()
    answer, sources = await loop.run_in_executor(
        _executor, rag_answer, req.question, get_vector_store(), req.research_question
    )
    return {"answer": answer, "sources": sources}


@app.get("/api/health")
async def health():
    return {"status": "ok", "vector_store": get_vector_store() is not None}
