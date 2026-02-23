"""
LitLens Pipeline — Core agent logic with zero Streamlit dependency.

This module contains all 8 LangGraph agents, the FAISS vector store builder,
and the pipeline runner. It is imported by api.py (FastAPI) to serve the
React frontend.
"""

import os
import json
import re
import tempfile
from typing import TypedDict, List, Annotated
from operator import add
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.callbacks.manager import get_openai_callback
from langgraph.graph import StateGraph, END

# ─── Model Constants (DO NOT instantiate inside agents) ───────────────────────
LLM_MINI = ChatOpenAI(model="gpt-4o-mini", temperature=0)
LLM_FULL = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

BATCH_SIZE = 5
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5

_VECTOR_STORE = None
_RAW_TEXTS = []
_METHODOLOGY_PATTERNS = []


def _grounding_system(context: str) -> str:
    return (
        "You are analyzing ONLY the following document excerpts.\n"
        "You MUST NOT use any knowledge from your training data.\n"
        "If the information is not in the excerpts below, say 'Not found in documents.'\n\n"
        f"DOCUMENT EXCERPTS:\n{context}\n\n"
        "STRICT RULE: Your entire response must be grounded in the excerpts above.\n"
        "Any claim not directly supported by the excerpts is a hallucination.\n"
        "Do NOT use examples, analogies, or background knowledge from outside these excerpts."
    )


_GROUNDING_PREFIX = (
    "Based ONLY on the document excerpts provided in the system prompt, "
    "with zero use of outside knowledge:\n\n"
)


# ─── Agent State ──────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    research_question: str
    research_domain: str
    papers: List[dict]
    claim_map: List[dict]
    contradiction_map: List[dict]
    methodology_table: List[dict]
    evidence_scores: List[dict]
    gap_analysis: str
    literature_review_draft: str
    final_report: str
    current_agent: str
    agent_history: Annotated[List[str], add]


# ─── Helpers ──────────────────────────────────────────────────────────────────
def extract_json(text: str):
    for pattern in [r'```(?:json)?\s*([\s\S]*?)```', r'(\[[\s\S]*\])', r'(\{[\s\S]*\})']:
        match = re.search(pattern, text)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except (json.JSONDecodeError, IndexError):
                continue
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return None


def load_pdf_text(file_path: str) -> str:
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    return "\n\n".join(p.page_content for p in pages)


def extract_raw_texts_from_paths(file_paths: list[dict]) -> list:
    """Process PDF files from disk paths. Each item: {path, filename}."""
    results = []
    for item in file_paths:
        try:
            text = load_pdf_text(item["path"])
            results.append({"filename": item["filename"], "text": text})
        except Exception as e:
            results.append({"filename": item["filename"], "text": "", "error": str(e)})
    return results


def safe_int(val, default=0):
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def get_vector_store():
    global _VECTOR_STORE
    if not _VECTOR_STORE:
        print("[get_vector_store] WARNING: _VECTOR_STORE is None!")
    return _VECTOR_STORE


# ─── FAISS Vector Store ──────────────────────────────────────────────────────
def build_vector_store(raw_texts: list):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    all_chunks = []
    for item in raw_texts:
        if not item.get("text"):
            continue
        docs = splitter.create_documents(
            [item["text"]], metadatas=[{"source": item["filename"]}],
        )
        all_chunks.extend(docs)
    if not all_chunks:
        return None
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.from_documents(all_chunks, embeddings)


def rag_answer(question: str, vector_store, rq: str) -> tuple[str, list[str]]:
    if not vector_store:
        response = LLM_MINI.invoke(question)
        return response.content, ["General Knowledge"]
    docs = vector_store.similarity_search(question, k=TOP_K)
    context = "\n---\n".join(
        f"[{d.metadata.get('source', '?')}]\n{d.page_content}" for d in docs
    )
    prompt = (
        f"You are a research assistant. The user's research question is: {rq}\n"
        f"Answer based on the provided paper excerpts. Cite sources.\n\n"
        f"Question: {question}\n\nContext:\n{context}\n\nAnswer:"
    )
    response = LLM_MINI.invoke(prompt)
    sources = list(set(d.metadata.get("source", "?") for d in docs))
    return response.content, sources


# ─── Agent 1: Paper Ingestion (LLM_MINI) — PARALLEL ──────────────────────────
def _ingest_one_paper(item):
    """Process a single paper. Called concurrently by Agent 1."""
    if not item.get("text"):
        return None, item["filename"]
    text = item["text"][:6000]
    sys_msg = _grounding_system(text)
    usr_msg = (
        _GROUNDING_PREFIX +
        "Extract structured metadata from the paper in the system prompt. "
        "Extract only what is explicitly stated. Never infer or hallucinate. "
        "If a field is not found, set it to 'Not specified'.\n\n"
        "Return a single JSON object with exactly these keys:\n"
        '{"title":"...","authors":"...","year":"...","abstract":"...",'
        '"research_question":"...","methodology":"...","sample_size":"...",'
        '"data_collection":"...","key_findings":["..."],"key_claims":["..."],'
        '"limitations":["..."]}'
    )
    try:
        response = LLM_MINI.invoke([SystemMessage(content=sys_msg), HumanMessage(content=usr_msg)])
        paper = extract_json(response.content)
        if paper and isinstance(paper, dict):
            paper["source_file"] = item["filename"]
            print(f"[Agent1] ✓ {paper.get('title','?')!r}")
            return paper, None
    except Exception as e:
        print(f"[Agent1] ✗ {item['filename']}: {e}")
    return None, item["filename"]


def paper_ingestion_agent(state: AgentState) -> dict:
    global _RAW_TEXTS
    raw_texts = _RAW_TEXTS
    print(f"[Agent1-Ingestion] Processing {len(raw_texts)} papers in parallel...")
    papers, failed = [], []

    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = {pool.submit(_ingest_one_paper, item): item for item in raw_texts}
        for future in as_completed(futures):
            paper, fail_name = future.result()
            if paper:
                papers.append(paper)
            elif fail_name:
                failed.append(fail_name)

    msg = f"Paper Ingestion: Extracted metadata from {len(papers)}/{len(raw_texts)} papers"
    if failed:
        msg += f" | Failed: {', '.join(failed)}"
    return {"papers": papers, "current_agent": "Paper Ingestion", "agent_history": [msg]}


# ─── Agent 2: Claim Extraction (LLM_MINI) — PARALLEL ─────────────────────────
def _extract_claims_batch(batch, rq, vs):
    """Process one batch of papers for claims. Called concurrently."""
    faiss_context = ""
    if vs:
            docs = vs.similarity_search(rq, k=4)
            faiss_context = "\n---\n".join(d.page_content for d in docs)

    paper_data = json.dumps([
        {"title": p.get("title", "?"), "year": p.get("year", "?"),
         "authors": p.get("authors", "?"),
         "key_findings": p.get("key_findings", []),
         "key_claims": p.get("key_claims", []),
         "methodology": p.get("methodology", "?"),
         "sample_size": p.get("sample_size", "?"),
         "limitations": p.get("limitations", [])}
        for p in batch
    ], indent=2)

    combined = f"EXTRACTED PAPER DATA:\n{paper_data}\n\nRAW PAPER EXCERPTS:\n{faiss_context}"
    sys_msg = _grounding_system(combined)
    usr_msg = (
        _GROUNDING_PREFIX +
        "Extract and categorize all significant claims from the papers above. "
        "Use both the extracted paper data and raw excerpts. "
        "Rate evidence strength based on study design and sample size.\n\n"
        "For each claim return a JSON object with:\n"
        "  claim_text, source_paper (\"Title (Year)\"), "
        "claim_type (finding|hypothesis|methodology|limitation), "
        "evidence_strength (strong|moderate|weak), "
        "theme (short topic label for grouping).\n"
        "Return a JSON array."
    )
    try:
        response = LLM_MINI.invoke([SystemMessage(content=sys_msg), HumanMessage(content=usr_msg)])
        claims = extract_json(response.content)
        if isinstance(claims, list):
            print(f"[Agent2-Claims] Batch extracted {len(claims)} claims")
            return claims
    except Exception as e:
        print(f"[Agent2-Claims] ERROR: {e}")
    return []


def claim_extraction_agent(state: AgentState) -> dict:
    rq = state.get("research_question", "")
    papers = state.get("papers", [])
    print(f"[Agent2-Claims] Processing {len(papers)} papers in parallel batches...")
    if not papers:
        return {"claim_map": [], "current_agent": "Claim Extraction",
                "agent_history": ["Claim Extraction: No papers to process"]}

    vs = get_vector_store()
    batches = [papers[i:i+BATCH_SIZE] for i in range(0, len(papers), BATCH_SIZE)]

    all_claims = []
    with ThreadPoolExecutor(max_workers=len(batches)) as pool:
        futures = [pool.submit(_extract_claims_batch, batch, rq, vs) for batch in batches]
        for future in as_completed(futures):
            all_claims.extend(future.result())

    return {"claim_map": all_claims, "current_agent": "Claim Extraction",
            "agent_history": [f"Claim Extraction: Identified {len(all_claims)} claims across {len(papers)} papers"]}


# ─── Agent 3: Contradiction Detector (LLM_FULL) ──────────────────────────────
def contradiction_detector_agent(state: AgentState) -> dict:
    papers = state.get("papers", [])
    claims = state.get("claim_map", [])
    print(f"[Agent3-Contradictions] papers={len(papers)}, claims={len(claims)}")

    if len(papers) < 5:
        return {"contradiction_map": [], "current_agent": "Contradiction Detector",
                "agent_history": ["Contradiction Detector: Skipped — upload 5+ papers for contradiction analysis"]}
    if not claims:
        return {"contradiction_map": [], "current_agent": "Contradiction Detector",
                "agent_history": ["Contradiction Detector: No claims to analyze"]}

    rq = state.get("research_question", "")
    vs = get_vector_store()
    context = ""
    if vs:
        docs = vs.similarity_search(rq, k=4)
        context = "\n---\n".join(d.page_content for d in docs)

    claims_text = json.dumps(claims[:50], indent=2)
    combined = f"PAPER EXCERPTS:\n{context}\n\nEXTRACTED CLAIMS:\n{claims_text}"

    sys_msg = (
        "You are a critical peer reviewer analyzing research papers for contradictions. "
        "A contradiction exists when two papers make opposing claims about the same phenomenon. "
        "Use the paper excerpts and extracted claims below to find disagreements.\n\n"
        f"{combined}\n\n"
        "Look for: direct contradictions, partial disagreements, and findings that conflict "
        "due to different contexts or methodologies."
    )
    usr_msg = (
        "Identify all contradictions and disagreements between the papers.\n\n"
        "For each contradiction return a JSON object with:\n"
        "  topic, paper_a {title, year, position}, paper_b {title, year, position},\n"
        "  severity (direct contradiction|partial disagreement|different context),\n"
        "  possible_explanation.\n"
        "Return a JSON array. If none found, return []."
    )
    try:
        response = LLM_FULL.invoke([SystemMessage(content=sys_msg), HumanMessage(content=usr_msg)])
        contradictions = extract_json(response.content)
        if not isinstance(contradictions, list):
            contradictions = []
    except Exception:
        contradictions = []

    return {"contradiction_map": contradictions, "current_agent": "Contradiction Detector",
            "agent_history": [f"Contradiction Detector: Found {len(contradictions)} contradictions/disagreements"]}


# ─── Agent 4: Methodology Comparator (LLM_MINI) ─────────────────────────────
def methodology_comparator_agent(state: AgentState) -> dict:
    papers = state.get("papers", [])
    print(f"[Agent4-Methodology] papers={len(papers)}")
    if not papers:
        return {"methodology_table": [], "current_agent": "Methodology Comparator",
                "agent_history": ["Methodology Comparator: No papers to analyze"]}

    rq = state.get("research_question", "")
    vs = get_vector_store()
    context = ""
    if vs:
        docs = vs.similarity_search(rq + " methodology study design", k=4)
        context = "\n---\n".join(d.page_content for d in docs)

    paper_summary = json.dumps([
        {"paper_title": p.get("title", "?"), "year": p.get("year", "?"),
         "methodology": p.get("methodology", "?"), "sample_size": p.get("sample_size", "?"),
         "data_collection": p.get("data_collection", "?"), "limitations": p.get("limitations", [])}
        for p in papers
    ], indent=2)

    combined = f"PAPER METADATA:\n{paper_summary}\n\nRAW EXCERPTS:\n{context}" if context else f"PAPER METADATA:\n{paper_summary}"
    sys_msg = _grounding_system(combined)
    usr_msg = (
        _GROUNDING_PREFIX +
        "Compare study designs across the papers.\n\n"
        "For each paper return a JSON object with these EXACT keys:\n"
        '  paper_title, year, study_design, sample_size, data_collection_method, '
        'statistical_methods, key_strength, key_limitation\n\n'
        "Also list overall methodology patterns.\n\n"
        'Return JSON: {"comparisons": [{"paper_title":"...","year":"...","study_design":"...",'
        '"sample_size":"...","data_collection_method":"...","statistical_methods":"...",'
        '"key_strength":"...","key_limitation":"..."}], "patterns": ["..."]}'
    )
    try:
        response = LLM_MINI.invoke([SystemMessage(content=sys_msg), HumanMessage(content=usr_msg)])
        print(f"[Agent4-Methodology] RAW response length: {len(response.content)}")
        result = extract_json(response.content)
        if isinstance(result, dict):
            comparisons = result.get("comparisons", [])
            patterns = result.get("patterns", [])
        elif isinstance(result, list):
            comparisons, patterns = result, []
        else:
            print(f"[Agent4-Methodology] extract_json returned: {type(result)}")
            comparisons, patterns = [], []
    except Exception as e:
        print(f"[Agent4-Methodology] ERROR: {e}")
        comparisons, patterns = [], []

    if not comparisons:
        print("[Agent4-Methodology] LLM returned no comparisons, building from paper metadata")
        comparisons = [
            {"paper_title": p.get("title", "?"), "year": p.get("year", "?"),
             "study_design": p.get("methodology", "Not specified"),
             "sample_size": p.get("sample_size", "Not specified"),
             "data_collection_method": p.get("data_collection", "Not specified"),
             "statistical_methods": "Not specified",
             "key_strength": "Not specified", "key_limitation": "Not specified"}
            for p in papers
        ]

    global _METHODOLOGY_PATTERNS
    _METHODOLOGY_PATTERNS = patterns
    print(f"[Agent4-Methodology] FINAL comparisons={len(comparisons)}, patterns={len(patterns)}")
    if comparisons:
        print(f"[Agent4-Methodology] SAMPLE KEYS: {list(comparisons[0].keys())}")
    return {"methodology_table": comparisons, "current_agent": "Methodology Comparator",
            "agent_history": [f"Methodology Comparator: Compared {len(comparisons)} studies, found {len(patterns)} patterns"]}


# ─── Agent 5: Evidence Scorer (LLM_MINI) ─────────────────────────────────────
def evidence_scorer_agent(state: AgentState) -> dict:
    claims = state.get("claim_map", [])
    contradictions = state.get("contradiction_map", [])
    print(f"[Agent5-Evidence] claims={len(claims)}, contradictions={len(contradictions)}")
    if not claims:
        return {"evidence_scores": [], "current_agent": "Evidence Scorer",
                "agent_history": ["Evidence Scorer: No claims to score"]}

    rq = state.get("research_question", "")
    vs = get_vector_store()
    context = ""
    if vs:
        docs = vs.similarity_search(rq, k=4)
        context = "\n---\n".join(d.page_content for d in docs)

    sys_msg = _grounding_system(context)
    usr_msg = (
        _GROUNDING_PREFIX +
        "Score each claim based on evidence quality in the excerpts.\n\n"
        f"Extracted claims:\n{json.dumps(claims[:80], indent=2)}\n\n"
        f"Contradictions:\n{json.dumps(contradictions, indent=2)}\n\n"
        "For each major claim return a JSON object with:\n"
        "  claim_text, supporting_papers (count), evidence_quality (brief text),\n"
        "  contradicted (true/false), evidence_score (0-100),\n"
        "  flag (well-supported | needs more evidence | widely cited but poorly evidenced | null).\n"
        "Return a JSON array."
    )
    try:
        response = LLM_MINI.invoke([SystemMessage(content=sys_msg), HumanMessage(content=usr_msg)])
        scores = extract_json(response.content)
        if not isinstance(scores, list):
            scores = []
    except Exception:
        scores = []

    return {"evidence_scores": scores, "current_agent": "Evidence Scorer",
            "agent_history": [f"Evidence Scorer: Scored {len(scores)} claims"]}


# ─── Agent 6: Gap Analyzer (LLM_FULL) ────────────────────────────────────────
def gap_analyzer_agent(state: AgentState) -> dict:
    rq = state.get("research_question", "")
    rd = state.get("research_domain", "")
    papers = state.get("papers", [])
    claims = state.get("claim_map", [])
    contradictions = state.get("contradiction_map", [])
    print(f"[Agent6-Gaps] rq={rq!r}, papers={len(papers)}, claims={len(claims)}")

    papers_brief = "\n".join(
        f"- {p.get('title', '?')} ({p.get('year', '?')})" for p in papers
    )
    claims_brief = "\n".join(
        f"- [{c.get('evidence_strength', '?')}] {c.get('claim_text', '?')}" for c in claims[:50]
    )
    contra_brief = "\n".join(
        f"- {c.get('topic', '?')}: {c.get('severity', '?')}" for c in contradictions
    ) or "None identified"

    vs = get_vector_store()
    context = ""
    if vs:
        docs = vs.similarity_search(rq, k=4)
        context = "\n---\n".join(d.page_content for d in docs)

    sys_msg = _grounding_system(context)
    usr_msg = (
        _GROUNDING_PREFIX +
        "Identify research gaps. Think about what has NOT been studied.\n\n"
        f"Research Question: {rq}\nDomain: {rd}\n\n"
        f"Papers:\n{papers_brief}\n\nClaims:\n{claims_brief}\n\nContradictions:\n{contra_brief}\n\n"
        "For each gap: 1) Describe it, 2) Why it matters, 3) Suggest a study design.\n\n"
        f'Start with: "Given your research question \'{rq}\', the following gaps emerge:"'
    )
    try:
        response = LLM_FULL.invoke([SystemMessage(content=sys_msg), HumanMessage(content=usr_msg)])
        analysis = response.content
    except Exception as e:
        analysis = f"Gap analysis could not be completed: {e}"

    return {"gap_analysis": analysis, "current_agent": "Gap Analyzer",
            "agent_history": ["Gap Analyzer: Identified research gaps and opportunities"]}


# ─── Agent 7: Literature Review Writer (LLM_FULL) ────────────────────────────
def literature_review_writer_agent(state: AgentState) -> dict:
    rq = state.get("research_question", "")
    rd = state.get("research_domain", "")
    papers = state.get("papers", [])
    claims = state.get("claim_map", [])
    contradictions = state.get("contradiction_map", [])
    methodology = state.get("methodology_table", [])
    gaps = state.get("gap_analysis", "")
    print(f"[Agent7-LitReview] rq={rq!r}, papers={len(papers)}")

    papers_summary = json.dumps(
        [{"title": p.get("title"), "authors": p.get("authors"), "year": p.get("year"),
          "key_findings": p.get("key_findings")} for p in papers], indent=2)
    claims_summary = json.dumps(claims[:60], indent=2)
    contra_summary = json.dumps(contradictions, indent=2) if contradictions else "None"
    methods_summary = json.dumps(methodology, indent=2)

    vs = get_vector_store()
    context = ""
    if vs:
        docs = vs.similarity_search(rq, k=4)
        context = "\n---\n".join(d.page_content for d in docs)

    sys_msg = _grounding_system(context)
    usr_msg = (
        _GROUNDING_PREFIX +
        "Write a thematic literature review. Never summarize papers one by one. "
        "Every empirical claim must be cited (Author, Year). "
        f"Connect the conclusion to the research question: '{rq}'\n\n"
        f"Domain: {rd}\nPapers ({len(papers)} total):\n{papers_summary}\n\n"
        f"Claims:\n{claims_summary}\n\nContradictions:\n{contra_summary}\n\n"
        f"Methodology:\n{methods_summary}\n\nGaps:\n{gaps[:3000]}\n\n"
        "Sections: 1. Introduction, 2. Thematic Analysis (3-5 subsections), "
        "3. Methodological Considerations, 4. Contradictions and Debates, "
        "5. Research Gaps, 6. Conclusion.\n"
        "500-800 words. Only cite papers from the excerpts."
    )
    try:
        response = LLM_FULL.invoke([SystemMessage(content=sys_msg), HumanMessage(content=usr_msg)])
        review = response.content
    except Exception as e:
        review = f"Literature review generation failed: {e}"

    return {"literature_review_draft": review, "current_agent": "Literature Review Writer",
            "agent_history": ["Literature Review Writer: Drafted thematic literature review"]}


# ─── Agent 8: Report Generator (LLM_MINI) ────────────────────────────────────
def report_generator_agent(state: AgentState) -> dict:
    rq = state.get("research_question", "")
    rd = state.get("research_domain", "")
    papers = state.get("papers", [])
    claims = state.get("claim_map", [])
    contradictions = state.get("contradiction_map", [])
    scores = state.get("evidence_scores", [])
    gaps = state.get("gap_analysis", "")
    print(f"[Agent8-Report] papers={len(papers)}, claims={len(claims)}")

    n_papers, n_claims, n_contradictions = len(papers), len(claims), len(contradictions)
    gap_lines = [l.strip() for l in gaps.split("\n") if l.strip() and not l.strip().startswith("#")]
    n_gaps = sum(1 for l in gap_lines if l.startswith("-") or l.startswith("•") or (l and l[0].isdigit()))
    strong = sum(1 for s in scores if safe_int(s.get("evidence_score", 0)) >= 70)
    moderate = sum(1 for s in scores if 40 <= safe_int(s.get("evidence_score", 0)) < 70)
    weak = sum(1 for s in scores if safe_int(s.get("evidence_score", 0)) < 40)

    sys_msg = "You MUST use ONLY the statistics provided. Do not invent data."
    usr_msg = (
        _GROUNDING_PREFIX +
        f"Research Question: {rq}\nDomain: {rd}\n"
        f"Papers: {n_papers}, Claims: {n_claims}, Contradictions: {n_contradictions}, "
        f"Gaps: ~{n_gaps}\nEvidence: {strong} strong, {moderate} moderate, {weak} weak\n\n"
        "Write ONLY a 3-4 sentence executive summary paragraph."
    )
    try:
        response = LLM_MINI.invoke([SystemMessage(content=sys_msg), HumanMessage(content=usr_msg)])
        exec_summary = response.content.strip()
    except Exception:
        exec_summary = f"Analysis of {n_papers} papers yielded {n_claims} claims."

    report = json.dumps({
        "research_question": rq, "domain": rd,
        "n_papers": n_papers, "n_claims": n_claims,
        "n_contradictions": n_contradictions, "n_gaps": n_gaps,
        "strong": strong, "moderate": moderate, "weak": weak,
        "executive_summary": exec_summary,
    })
    return {"final_report": report, "current_agent": "Report Generator",
            "agent_history": [f"Report Generator: {n_papers} papers, {n_claims} claims, {n_contradictions} contradictions"]}


# ─── Parallel Combo Nodes ─────────────────────────────────────────────────────
def _parallel_analysis_node(state: AgentState) -> dict:
    """Run Agents 3 + 4 concurrently, then Agent 5 (needs contradictions from 3)."""
    print("[Parallel] Running Agents 3+4 concurrently...")
    with ThreadPoolExecutor(max_workers=2) as pool:
        f3 = pool.submit(contradiction_detector_agent, state)
        f4 = pool.submit(methodology_comparator_agent, state)
        r3 = f3.result()
        r4 = f4.result()

    state_for_5 = {**state, **r3}
    print("[Parallel] Running Agent 5 (needs contradictions)...")
    r5 = evidence_scorer_agent(state_for_5)

    merged = {
        "contradiction_map": r3.get("contradiction_map", []),
        "methodology_table": r4.get("methodology_table", []),
        "evidence_scores": r5.get("evidence_scores", []),
        "current_agent": "Analysis",
        "agent_history": (
            r3.get("agent_history", []) +
            r4.get("agent_history", []) +
            r5.get("agent_history", [])
        ),
    }
    print(f"[Parallel] Analysis done — contradictions={len(merged['contradiction_map'])}, "
          f"methodology={len(merged['methodology_table'])}, evidence={len(merged['evidence_scores'])}")
    return merged


def _parallel_synthesis_node(state: AgentState) -> dict:
    """Run Agents 6 + 7 concurrently in threads."""
    print("[Parallel] Running Agents 6+7 concurrently...")
    with ThreadPoolExecutor(max_workers=2) as pool:
        f6 = pool.submit(gap_analyzer_agent, state)
        f7 = pool.submit(literature_review_writer_agent, state)
        r6 = f6.result()
        r7 = f7.result()
    merged = {}
    merged.update(r6)
    merged.update(r7)
    merged["current_agent"] = "Synthesis"
    merged["agent_history"] = r6.get("agent_history", []) + r7.get("agent_history", [])
    print("[Parallel] Agents 6+7 done")
    return merged


# ─── LangGraph Pipeline ──────────────────────────────────────────────────────
AGENT_LABELS = [
    ("paper_ingestion", "Ingesting papers"),
    ("claim_extraction", "Extracting claims"),
    ("parallel_analysis", "Analyzing contradictions, methodology, evidence"),
    ("parallel_synthesis", "Writing gaps analysis and literature review"),
    ("report_generator", "Generating report"),
]


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("paper_ingestion", paper_ingestion_agent)
    g.add_node("claim_extraction", claim_extraction_agent)
    g.add_node("parallel_analysis", _parallel_analysis_node)
    g.add_node("parallel_synthesis", _parallel_synthesis_node)
    g.add_node("report_generator", report_generator_agent)

    g.set_entry_point("paper_ingestion")
    g.add_edge("paper_ingestion", "claim_extraction")
    g.add_edge("claim_extraction", "parallel_analysis")
    g.add_edge("parallel_analysis", "parallel_synthesis")
    g.add_edge("parallel_synthesis", "report_generator")
    g.add_edge("report_generator", END)
    return g.compile()


def run_pipeline(rq: str, rd: str, on_step=None) -> dict:
    """Run the full 8-agent pipeline. Returns the final state dict.

    on_step(step_number, agent_label, history_msg) is called after each agent
    completes, allowing the caller to stream progress.
    """
    if not rq:
        return {"error": "Research question is empty"}

    print(f"[Pipeline] START — rq={rq!r}, rd={rd!r}")
    graph = build_graph()
    initial: AgentState = {
        "research_question": rq, "research_domain": rd,
        "papers": [], "claim_map": [], "contradiction_map": [],
        "methodology_table": [], "evidence_scores": [],
        "gap_analysis": "", "literature_review_draft": "",
        "final_report": "", "current_agent": "", "agent_history": [],
    }

    final = dict(initial)
    with get_openai_callback() as cb:
        step = 0
        for output in graph.stream(initial):
            for node_name, update in output.items():
                step += 1
                for k, v in update.items():
                    if k == "agent_history":
                        final.setdefault(k, []).extend(v)
                    else:
                        final[k] = v
                history = update.get("agent_history", [])
                agent_label = dict(AGENT_LABELS).get(node_name, node_name)
                if on_step and history:
                    on_step(step, agent_label, history[-1])

    final["research_question"] = rq
    final["research_domain"] = rd
    final["total_cost"] = cb.total_cost
    final["total_tokens"] = cb.prompt_tokens + cb.completion_tokens
    print(f"[Pipeline] DONE — papers={len(final.get('papers', []))}, cost=${cb.total_cost:.4f}")
    return final


def set_pipeline_data(raw_texts: list, vector_store):
    """Set the module globals that agents read during pipeline execution."""
    global _VECTOR_STORE, _RAW_TEXTS
    _RAW_TEXTS = raw_texts
    _VECTOR_STORE = vector_store
    print(f"[set_pipeline_data] raw_texts={len(raw_texts)}, "
          f"vector_store={'OK' if vector_store else 'None'}")


def transform_result_for_frontend(final: dict) -> dict:
    """Map the pipeline output field names to what the React frontend expects."""
    report_raw = final.get("final_report", "{}")
    try:
        report = json.loads(report_raw)
    except (json.JSONDecodeError, TypeError):
        report = {}

    meth = final.get("methodology_table", [])
    print(f"[transform] methodology_table has {len(meth)} items")
    if meth:
        print(f"[transform] methodology sample keys: {list(meth[0].keys()) if isinstance(meth[0], dict) else type(meth[0])}")

    return {
        "researchQuestion": final.get("research_question", ""),
        "domain": final.get("research_domain", ""),
        "report": report,
        "papers": final.get("papers", []),
        "claims": final.get("claim_map", []),
        "contradictions": final.get("contradiction_map", []),
        "methodology": meth,
        "methodologyPatterns": _METHODOLOGY_PATTERNS,
        "evidence_scores": final.get("evidence_scores", []),
        "gap_analysis": final.get("gap_analysis", ""),
        "literature_review_draft": final.get("literature_review_draft", ""),
        "agent_history": final.get("agent_history", []),
        "total_cost": final.get("total_cost", 0),
        "total_tokens": final.get("total_tokens", 0),
    }
