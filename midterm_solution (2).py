"""
LitLens — Literature Intelligence for Researchers

A multi-agent research platform that helps PhD students and academic researchers
synthesize 10-50 research papers, find gaps and contradictions, and generate
literature review drafts automatically.

Setup:
    pip install streamlit langchain langchain-openai langchain-community \
                langchain-text-splitters langgraph faiss-cpu pypdf openai \
                tiktoken python-dotenv
    Create .env with: OPENAI_API_KEY=your-key
    streamlit run "midterm_solution (2).py"

Architecture: LangGraph pipeline with 8 specialized agents
Models: gpt-4o-mini (fast tasks) + gpt-4o (reasoning tasks)
"""

import os
import json
import re
import uuid
import tempfile
from typing import TypedDict, List, Annotated
from operator import add
from dotenv import load_dotenv

load_dotenv()

import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.callbacks.manager import get_openai_callback
from langgraph.graph import StateGraph, END

st.set_page_config(
    page_title="LitLens — Literature Intelligence",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

if not os.environ.get("OPENAI_API_KEY"):
    st.error("Please set your `OPENAI_API_KEY` in the `.env` file and restart.")
    st.stop()

# ─── Model Constants (DO NOT instantiate inside agents) ───────────────────────
LLM_MINI = ChatOpenAI(model="gpt-4o-mini", temperature=0)
LLM_FULL = ChatOpenAI(model="gpt-4o", temperature=0)

BATCH_SIZE = 5
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5

COST_PER_TOKEN_MINI = 0.00000015
COST_PER_TOKEN_FULL = 0.0000025

# Module-level holder so agents can access it inside graph.stream().
# st.session_state is NOT reliably accessible inside LangGraph execution.
_VECTOR_STORE = None
_RAW_TEXTS = []


def _grounding_system(context: str) -> str:
    """System message that forces the LLM to use ONLY the provided excerpts."""
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
    """Parse JSON from an LLM response, handling code fences and raw JSON."""
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


def extract_raw_texts(uploaded_files) -> list:
    results = []
    for f in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(f.getvalue())
            path = tmp.name
        try:
            text = load_pdf_text(path)
            results.append({"filename": f.name, "text": text})
        except Exception as e:
            results.append({"filename": f.name, "text": "", "error": str(e)})
        finally:
            os.unlink(path)
    return results


def safe_int(val, default=0):
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def get_vector_store():
    """Return the module-level FAISS vector store. This is set before pipeline runs."""
    global _VECTOR_STORE
    if not _VECTOR_STORE:
        print("[get_vector_store] WARNING: _VECTOR_STORE is None!")
    return _VECTOR_STORE


def retrieve_context(query: str, k: int = TOP_K) -> str:
    """Search FAISS for chunks relevant to query. Returns formatted context string."""
    vs = get_vector_store()
    if not vs or not query:
        print(f"[retrieve_context] SKIP — vs={'exists' if vs else 'None'}, query={bool(query)}")
        return ""
    try:
        docs = vs.similarity_search(query, k=k)
        context = "\n---\n".join(
            f"[Source: {d.metadata.get('source', 'Unknown')}]\n{d.page_content}"
            for d in docs
        )
        sources = list(set(d.metadata.get('source', '?') for d in docs))
        print(f"[retrieve_context] OK — {len(docs)} chunks, {len(context)} chars, sources={sources}")
        return context
    except Exception as e:
        print(f"[retrieve_context] ERROR: {e}")
        return ""


# ─── FAISS Vector Store for RAG Chat ─────────────────────────────────────────
def build_vector_store(raw_texts: list):
    """Chunk all paper texts and build a FAISS index for the chat feature."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    all_chunks = []
    for item in raw_texts:
        if not item.get("text"):
            continue
        docs = splitter.create_documents(
            [item["text"]],
            metadatas=[{"source": item["filename"]}],
        )
        all_chunks.extend(docs)
    if not all_chunks:
        return None
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.from_documents(all_chunks, embeddings)


def rag_answer(question: str, vector_store, rq: str) -> tuple[str, list[str]]:
    """Answer a question using RAG against the uploaded papers."""
    if not vector_store:
        response = LLM_MINI.invoke(question)
        return response.content, ["General Knowledge"]

    docs = vector_store.similarity_search(question, k=TOP_K)
    context = "\n---\n".join(
        f"[{d.metadata.get('source', '?')}]\n{d.page_content}" for d in docs
    )
    prompt = (
        f"You are a research assistant. The user's overall research question is: {rq}\n"
        f"Answer the following question based on the provided paper excerpts. "
        f"Cite sources when possible.\n\n"
        f"Question: {question}\n\nContext:\n{context}\n\nAnswer:"
    )
    response = LLM_MINI.invoke(prompt)
    sources = list(set(d.metadata.get("source", "?") for d in docs))
    return response.content, sources


# ─── Agent 1: Paper Ingestion (LLM_MINI) ─────────────────────────────────────
def paper_ingestion_agent(state: AgentState) -> dict:
    rq = state.get("research_question", "")
    global _RAW_TEXTS
    raw_texts = _RAW_TEXTS
    print(f"[Agent1-Ingestion] research_question={rq!r}")
    print(f"[Agent1-Ingestion] raw_texts_count={len(raw_texts)}, "
          f"filenames={[r.get('filename','?') for r in raw_texts]}")
    papers, failed = [], []

    for item in raw_texts:
        if not item.get("text"):
            failed.append(item["filename"])
            continue

        text = item["text"][:15000]
        print(f"[Agent1-Ingestion] Processing {item['filename']} — {len(text)} chars of text")
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
                print(f"[Agent1-Ingestion] Extracted: title={paper.get('title','?')!r}")
                papers.append(paper)
            else:
                failed.append(item["filename"])
        except Exception:
            failed.append(item["filename"])

    msg = f"Paper Ingestion: Extracted metadata from {len(papers)}/{len(raw_texts)} papers"
    if failed:
        msg += f" | Failed: {', '.join(failed)}"
    return {"papers": papers, "current_agent": "Paper Ingestion", "agent_history": [msg]}


# ─── Agent 2: Claim Extraction (LLM_MINI) ────────────────────────────────────
def claim_extraction_agent(state: AgentState) -> dict:
    rq = state.get("research_question", "")
    papers = state.get("papers", [])
    print(f"[Agent2-Claims] research_question={rq!r}")
    print(f"[Agent2-Claims] papers_count={len(papers)}, "
          f"titles={[p.get('title','?')[:40] for p in papers[:5]]}")
    if not papers:
        return {
            "claim_map": [],
            "current_agent": "Claim Extraction",
            "agent_history": ["Claim Extraction: No papers to process"],
        }

    vs = get_vector_store()
    all_claims = []
    for i in range(0, len(papers), BATCH_SIZE):
        batch = papers[i : i + BATCH_SIZE]

        context = ""
        if vs:
            docs = vs.similarity_search(rq, k=8)
            context = "\n---\n".join(d.page_content for d in docs)
            print(f"[Agent2-Claims] FAISS returned {len(docs)} chunks, {len(context)} chars")
        else:
            print("[Agent2-Claims] WARNING: no vector_store, no grounding")

        sys_msg = _grounding_system(context)
        usr_msg = (
            _GROUNDING_PREFIX +
            "Extract and categorize all significant claims from the document excerpts. "
            "Only extract claims explicitly stated in the excerpts. "
            "Rate evidence strength based on study design and sample size.\n\n"
            "For each claim return a JSON object with:\n"
            "  claim_text, source_paper (\"Title (Year)\"), "
            "claim_type (finding|hypothesis|methodology|limitation), "
            "evidence_strength (strong|moderate|weak), "
            "theme (short topic label for grouping).\n"
            "Return a JSON array. Do NOT invent claims not in the excerpts."
        )
        try:
            response = LLM_MINI.invoke([SystemMessage(content=sys_msg), HumanMessage(content=usr_msg)])
            claims = extract_json(response.content)
            if isinstance(claims, list):
                all_claims.extend(claims)
        except Exception:
            pass

    return {
        "claim_map": all_claims,
        "current_agent": "Claim Extraction",
        "agent_history": [
            f"Claim Extraction: Identified {len(all_claims)} claims across {len(papers)} papers"
        ],
    }


# ─── Agent 3: Contradiction Detector (LLM_FULL) ──────────────────────────────
def contradiction_detector_agent(state: AgentState) -> dict:
    papers = state.get("papers", [])
    claims = state.get("claim_map", [])
    print(f"[Agent3-Contradictions] research_question={state.get('research_question', '')!r}")
    print(f"[Agent3-Contradictions] papers_count={len(papers)}, claims_count={len(claims)}")

    if len(papers) < 5:
        return {
            "contradiction_map": [],
            "current_agent": "Contradiction Detector",
            "agent_history": [
                "Contradiction Detector: Skipped — upload 5+ papers for contradiction analysis"
            ],
        }
    if not claims:
        return {
            "contradiction_map": [],
            "current_agent": "Contradiction Detector",
            "agent_history": ["Contradiction Detector: No claims to analyze"],
        }

    rq = state.get("research_question", "")
    vs = get_vector_store()
    context = ""
    if vs:
        docs = vs.similarity_search(rq, k=8)
        context = "\n---\n".join(d.page_content for d in docs)
        print(f"[Agent3-Contradictions] FAISS returned {len(docs)} chunks, {len(context)} chars")

    sys_msg = _grounding_system(context)
    usr_msg = (
        _GROUNDING_PREFIX +
        "Identify contradictions where papers make opposing claims about the same "
        "phenomenon under comparable conditions. Distinguish true contradictions from "
        "differences due to context, population, or methodology.\n\n"
        f"Extracted claims:\n{json.dumps(claims[:100], indent=2)}\n\n"
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

    return {
        "contradiction_map": contradictions,
        "current_agent": "Contradiction Detector",
        "agent_history": [
            f"Contradiction Detector: Found {len(contradictions)} contradictions/disagreements"
        ],
    }


# ─── Agent 4: Methodology Comparator (LLM_MINI) ─────────────────────────────
def methodology_comparator_agent(state: AgentState) -> dict:
    papers = state.get("papers", [])
    print(f"[Agent4-Methodology] research_question={state.get('research_question', '')!r}")
    print(f"[Agent4-Methodology] papers_count={len(papers)}")
    if not papers:
        return {
            "methodology_table": [],
            "current_agent": "Methodology Comparator",
            "agent_history": ["Methodology Comparator: No papers to analyze"],
        }

    summary = json.dumps(
        [
            {
                "title": p.get("title", "Unknown"),
                "year": p.get("year", "N/A"),
                "methodology": p.get("methodology", "N/A"),
                "sample_size": p.get("sample_size", "N/A"),
                "data_collection": p.get("data_collection", "N/A"),
                "limitations": p.get("limitations", []),
            }
            for p in papers
        ],
        indent=2,
    )
    rq = state.get("research_question", "")
    vs = get_vector_store()
    context = ""
    if vs:
        docs = vs.similarity_search(rq + " methodology study design", k=8)
        context = "\n---\n".join(d.page_content for d in docs)
        print(f"[Agent4-Methodology] FAISS returned {len(docs)} chunks, {len(context)} chars")

    sys_msg = _grounding_system(context)
    usr_msg = (
        _GROUNDING_PREFIX +
        "Compare study designs across the papers in the excerpts.\n\n"
        "For each paper extract: paper_title, year, study_design, sample_size, "
        "data_collection_method, statistical_methods, key_strength, key_limitation.\n"
        "Also list overall methodology patterns.\n\n"
        'Return JSON: {"comparisons": [...], "patterns": ["..."]}.'
    )
    try:
        response = LLM_MINI.invoke([SystemMessage(content=sys_msg), HumanMessage(content=usr_msg)])
        result = extract_json(response.content)
        if isinstance(result, dict):
            comparisons = result.get("comparisons", [])
            patterns = result.get("patterns", [])
        elif isinstance(result, list):
            comparisons, patterns = result, []
        else:
            comparisons, patterns = [], []
    except Exception:
        comparisons, patterns = [], []

    st.session_state.methodology_patterns = patterns
    return {
        "methodology_table": comparisons,
        "current_agent": "Methodology Comparator",
        "agent_history": [
            f"Methodology Comparator: Compared {len(comparisons)} studies, "
            f"found {len(patterns)} patterns"
        ],
    }


# ─── Agent 5: Evidence Scorer (LLM_MINI) ─────────────────────────────────────
def evidence_scorer_agent(state: AgentState) -> dict:
    claims = state.get("claim_map", [])
    contradictions = state.get("contradiction_map", [])
    print(f"[Agent5-Evidence] research_question={state.get('research_question', '')!r}")
    print(f"[Agent5-Evidence] claims_count={len(claims)}, contradictions_count={len(contradictions)}")
    if not claims:
        return {
            "evidence_scores": [],
            "current_agent": "Evidence Scorer",
            "agent_history": ["Evidence Scorer: No claims to score"],
        }

    rq = state.get("research_question", "")
    vs = get_vector_store()
    context = ""
    if vs:
        docs = vs.similarity_search(rq, k=8)
        context = "\n---\n".join(d.page_content for d in docs)
        print(f"[Agent5-Evidence] FAISS returned {len(docs)} chunks, {len(context)} chars")

    sys_msg = _grounding_system(context)
    usr_msg = (
        _GROUNDING_PREFIX +
        "Score each claim based on evidence quality in the excerpts. "
        "Multiple large rigorous studies = high score. Single small study = low.\n\n"
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

    return {
        "evidence_scores": scores,
        "current_agent": "Evidence Scorer",
        "agent_history": [f"Evidence Scorer: Scored {len(scores)} claims"],
    }


# ─── Agent 6: Gap Analyzer (LLM_FULL) ────────────────────────────────────────
def gap_analyzer_agent(state: AgentState) -> dict:
    rq = state.get("research_question", "")
    rd = state.get("research_domain", "")
    print(f"[Agent6-Gaps] research_question={rq!r}, research_domain={rd!r}")
    print(f"[Agent6-Gaps] papers_count={len(state.get('papers', []))}, "
          f"claims_count={len(state.get('claim_map', []))}")
    papers = state.get("papers", [])
    claims = state.get("claim_map", [])
    contradictions = state.get("contradiction_map", [])

    papers_brief = "\n".join(
        f"- {p.get('title', '?')} ({p.get('year', '?')}): {p.get('methodology', '?')}"
        for p in papers
    )
    claims_brief = "\n".join(
        f"- [{c.get('evidence_strength', '?')}] {c.get('claim_text', '?')} "
        f"— {c.get('source_paper', '?')}"
        for c in claims[:50]
    )
    contra_brief = (
        "\n".join(f"- {c.get('topic', '?')}: {c.get('severity', '?')}" for c in contradictions)
        or "None identified"
    )

    vs = get_vector_store()
    context = ""
    if vs:
        docs = vs.similarity_search(rq, k=10)
        context = "\n---\n".join(d.page_content for d in docs)
        print(f"[Agent6-Gaps] FAISS returned {len(docs)} chunks, {len(context)} chars")

    sys_msg = _grounding_system(context)
    usr_msg = (
        _GROUNDING_PREFIX +
        "Identify research gaps. Think about what has NOT been studied. "
        "Consider population, methodological, temporal, and theoretical gaps.\n\n"
        f"Research Question: {rq}\nDomain: {rd}\n\n"
        f"Papers analyzed:\n{papers_brief}\n\n"
        f"Key claims:\n{claims_brief}\n\n"
        f"Contradictions:\n{contra_brief}\n\n"
        "For each gap: 1) Describe it, 2) Why it matters for the research question, "
        "3) Suggest a study design.\n\n"
        f'Start with: "Given your research question \'{rq}\', the following gaps '
        f'emerge from the literature:"'
    )
    try:
        response = LLM_FULL.invoke([SystemMessage(content=sys_msg), HumanMessage(content=usr_msg)])
        analysis = response.content
    except Exception as e:
        analysis = f"Gap analysis could not be completed: {e}"

    return {
        "gap_analysis": analysis,
        "current_agent": "Gap Analyzer",
        "agent_history": ["Gap Analyzer: Identified research gaps and opportunities"],
    }


# ─── Agent 7: Literature Review Writer (LLM_FULL) ────────────────────────────
def literature_review_writer_agent(state: AgentState) -> dict:
    rq = state.get("research_question", "")
    rd = state.get("research_domain", "")
    print(f"[Agent7-LitReview] research_question={rq!r}, research_domain={rd!r}")
    print(f"[Agent7-LitReview] papers_count={len(state.get('papers', []))}, "
          f"claims_count={len(state.get('claim_map', []))}")
    papers = state.get("papers", [])
    claims = state.get("claim_map", [])
    contradictions = state.get("contradiction_map", [])
    methodology = state.get("methodology_table", [])
    gaps = state.get("gap_analysis", "")

    papers_summary = json.dumps(
        [
            {
                "title": p.get("title"),
                "authors": p.get("authors"),
                "year": p.get("year"),
                "key_findings": p.get("key_findings"),
            }
            for p in papers
        ],
        indent=2,
    )
    claims_summary = json.dumps(claims[:60], indent=2)
    contra_summary = json.dumps(contradictions, indent=2) if contradictions else "None identified"
    methods_summary = json.dumps(methodology, indent=2)

    vs = get_vector_store()
    context = ""
    if vs:
        docs = vs.similarity_search(rq, k=10)
        context = "\n---\n".join(d.page_content for d in docs)
        print(f"[Agent7-LitReview] FAISS returned {len(docs)} chunks, {len(context)} chars")

    sys_msg = _grounding_system(context)
    usr_msg = (
        _GROUNDING_PREFIX +
        "Write a thematic literature review synthesizing the document excerpts. "
        "Never summarize papers one by one. Use formal academic language. "
        "Every empirical claim must be cited (Author, Year). "
        "Connect the conclusion directly to the research question: "
        f"'{rq}'\n\n"
        f"Domain: {rd}\n"
        f"Papers ({len(papers)} total):\n{papers_summary}\n\n"
        f"Key Claims:\n{claims_summary}\n\n"
        f"Contradictions:\n{contra_summary}\n\n"
        f"Methodology Overview:\n{methods_summary}\n\n"
        f"Research Gaps:\n{gaps[:3000]}\n\n"
        "Sections: 1. Introduction, 2. Thematic Analysis (3-5 subsections), "
        "3. Methodological Considerations, 4. Contradictions and Debates, "
        "5. Research Gaps and Future Directions, 6. Conclusion.\n\n"
        "800-1200 words. Cite as (Author, Year). "
        "Only cite papers that appear in the document excerpts."
    )
    try:
        response = LLM_FULL.invoke([SystemMessage(content=sys_msg), HumanMessage(content=usr_msg)])
        review = response.content
    except Exception as e:
        review = f"Literature review generation failed: {e}"

    return {
        "literature_review_draft": review,
        "current_agent": "Literature Review Writer",
        "agent_history": ["Literature Review Writer: Drafted thematic literature review"],
    }


# ─── Agent 8: Report Generator (LLM_MINI) ────────────────────────────────────
def report_generator_agent(state: AgentState) -> dict:
    """Assembles metadata summary stats for the Overview tab."""
    rq = state.get("research_question", "")
    rd = state.get("research_domain", "")
    print(f"[Agent8-Report] research_question={rq!r}, research_domain={rd!r}")
    print(f"[Agent8-Report] papers={len(state.get('papers', []))}, "
          f"claims={len(state.get('claim_map', []))}, "
          f"contradictions={len(state.get('contradiction_map', []))}")
    papers = state.get("papers", [])
    claims = state.get("claim_map", [])
    contradictions = state.get("contradiction_map", [])
    scores = state.get("evidence_scores", [])
    gaps = state.get("gap_analysis", "")

    n_papers = len(papers)
    n_claims = len(claims)
    n_contradictions = len(contradictions)

    gap_lines = [l.strip() for l in gaps.split("\n") if l.strip() and not l.strip().startswith("#")]
    n_gaps = sum(1 for l in gap_lines if l.startswith("-") or l.startswith("•") or (l and l[0].isdigit()))

    strong = sum(1 for s in scores if safe_int(s.get("evidence_score", 0)) >= 70)
    moderate = sum(1 for s in scores if 40 <= safe_int(s.get("evidence_score", 0)) < 70)
    weak = sum(1 for s in scores if safe_int(s.get("evidence_score", 0)) < 40)

    sys_msg = (
        "You are a research report assistant. You MUST use ONLY the statistics provided. "
        "Do not invent any data or reference any topics not mentioned below."
    )
    usr_msg = (
        _GROUNDING_PREFIX +
        f"Research Question: {rq}\nDomain: {rd}\n"
        f"Papers: {n_papers}, Claims: {n_claims}, Contradictions: {n_contradictions}, "
        f"Gaps identified: ~{n_gaps}\n"
        f"Evidence: {strong} strong, {moderate} moderate, {weak} weak\n\n"
        "Write ONLY a 3-4 sentence executive summary paragraph from these statistics."
    )
    try:
        response = LLM_MINI.invoke([SystemMessage(content=sys_msg), HumanMessage(content=usr_msg)])
        exec_summary = response.content.strip()
    except Exception:
        exec_summary = (
            f"Analysis of {n_papers} papers yielded {n_claims} claims, "
            f"{n_contradictions} contradictions, and ~{n_gaps} research gaps."
        )

    report = json.dumps({
        "research_question": rq,
        "domain": rd,
        "n_papers": n_papers,
        "n_claims": n_claims,
        "n_contradictions": n_contradictions,
        "n_gaps": n_gaps,
        "strong": strong,
        "moderate": moderate,
        "weak": weak,
        "executive_summary": exec_summary,
    })

    return {
        "final_report": report,
        "current_agent": "Report Generator",
        "agent_history": [
            f"Report Generator: Assembled overview — {n_papers} papers, "
            f"{n_claims} claims, {n_contradictions} contradictions, ~{n_gaps} gaps"
        ],
    }


# ─── LangGraph Pipeline ──────────────────────────────────────────────────────
AGENT_LABELS = [
    ("paper_ingestion", "Ingesting papers"),
    ("claim_extraction", "Extracting claims"),
    ("contradiction_detector", "Detecting contradictions"),
    ("methodology_comparator", "Comparing methodologies"),
    ("evidence_scorer", "Scoring evidence"),
    ("gap_analyzer", "Analyzing research gaps"),
    ("literature_review_writer", "Writing literature review"),
    ("report_generator", "Generating report"),
]


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("paper_ingestion", paper_ingestion_agent)
    g.add_node("claim_extraction", claim_extraction_agent)
    g.add_node("contradiction_detector", contradiction_detector_agent)
    g.add_node("methodology_comparator", methodology_comparator_agent)
    g.add_node("evidence_scorer", evidence_scorer_agent)
    g.add_node("gap_analyzer", gap_analyzer_agent)
    g.add_node("literature_review_writer", literature_review_writer_agent)
    g.add_node("report_generator", report_generator_agent)

    g.set_entry_point("paper_ingestion")
    g.add_edge("paper_ingestion", "claim_extraction")
    g.add_edge("claim_extraction", "contradiction_detector")
    g.add_edge("contradiction_detector", "methodology_comparator")
    g.add_edge("methodology_comparator", "evidence_scorer")
    g.add_edge("evidence_scorer", "gap_analyzer")
    g.add_edge("gap_analyzer", "literature_review_writer")
    g.add_edge("literature_review_writer", "report_generator")
    g.add_edge("report_generator", END)

    return g.compile()


# ─── Streamlit UI ─────────────────────────────────────────────────────────────
def apply_css():
    st.markdown(
        """<style>
        .block-container { padding-top: 1.5rem; }
        div[data-testid="stMetric"] {
            background: #f0f4f8;
            border: 1px solid #d0dbe8;
            border-radius: 10px;
            padding: 14px 18px;
        }
        .stTabs [data-baseweb="tab-list"] { gap: 6px; }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px 8px 0 0;
            padding: 10px 18px;
            font-weight: 500;
        }
        h1 { color: #1e3a5f; }
        </style>""",
        unsafe_allow_html=True,
    )


def init_session():
    defaults = {
        "raw_texts": [],
        "results": None,
        "methodology_patterns": [],
        "vector_store": None,
        "chat_history": [],
        "total_cost": 0.0,
        "total_tokens_mini": 0,
        "total_tokens_full": 0,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def run_pipeline(rq: str, rd: str) -> dict:
    """Execute the full 8-agent pipeline with progress bar, status, and cost tracking."""
    import logging
    logger = logging.getLogger("litlens")

    if not rq:
        st.error("Research question is empty — cannot start analysis.")
        return {}

    print(f"[LitLens DEBUG] Pipeline start — research_question={rq!r}, research_domain={rd!r}")
    print(f"[LitLens DEBUG] raw_texts count={len(st.session_state.get('raw_texts', []))}, "
          f"vector_store={'LOADED' if st.session_state.get('vector_store') else 'MISSING'}")
    logger.info("[LitLens] Pipeline start — research_question=%r, research_domain=%r", rq, rd)
    st.toast(f"Starting analysis: \"{rq[:80]}{'…' if len(rq) > 80 else ''}\"")

    graph = build_graph()
    initial: AgentState = {
        "research_question": rq,
        "research_domain": rd,
        "papers": [],
        "claim_map": [],
        "contradiction_map": [],
        "methodology_table": [],
        "evidence_scores": [],
        "gap_analysis": "",
        "literature_review_draft": "",
        "final_report": "",
        "current_agent": "",
        "agent_history": [],
    }

    n_agents = len(AGENT_LABELS)
    final = dict(initial)  # seed with initial so research_question/domain survive

    progress_bar = st.progress(0, text="Starting analysis...")

    with get_openai_callback() as cb:
        with st.status("🔬 Analyzing literature...", expanded=True) as status:
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
                    pct = step / n_agents
                    progress_bar.progress(pct, text=f"Agent {step} of {n_agents}: {agent_label}...")
                    if history:
                        st.write(f"✓ {history[-1]}")
            status.update(label="✅ Analysis complete!", state="complete")

    progress_bar.progress(1.0, text="Analysis complete!")

    st.session_state.total_tokens_mini = cb.prompt_tokens + cb.completion_tokens
    st.session_state.total_cost = cb.total_cost

    final["research_question"] = rq
    final["research_domain"] = rd
    logger.info(
        "[LitLens] Pipeline done — papers=%d, claims=%d, contradictions=%d, "
        "research_question=%r, research_domain=%r",
        len(final.get("papers", [])),
        len(final.get("claim_map", [])),
        len(final.get("contradiction_map", [])),
        final["research_question"],
        final["research_domain"],
    )

    return final


# ─── Tab Renderers ────────────────────────────────────────────────────────────
def render_overview_tab(results: dict):
    report_raw = results.get("final_report", "{}")
    try:
        rpt = json.loads(report_raw)
    except (json.JSONDecodeError, TypeError):
        rpt = {}

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Papers Analyzed", rpt.get("n_papers", 0))
    c2.metric("Claims Extracted", rpt.get("n_claims", 0))
    c3.metric("Contradictions Found", rpt.get("n_contradictions", 0))
    c4.metric("Gaps Identified", rpt.get("n_gaps", 0))

    st.markdown("---")
    st.markdown(rpt.get("executive_summary", ""))

    with st.expander("🔗 Agent Pipeline Trace"):
        for entry in results.get("agent_history", []):
            st.write(f"→ {entry}")


def render_contradictions_tab(results: dict):
    papers = results.get("papers", [])
    contradictions = results.get("contradiction_map", [])

    if len(papers) < 5:
        st.info(
            "**Contradiction analysis requires 5 or more papers.** "
            f"You uploaded {len(papers)} paper(s). Upload additional papers "
            "and re-run the analysis to enable cross-paper contradiction detection."
        )
        return

    if not contradictions:
        st.success("No contradictions were detected across the analyzed papers.")
        return

    for i, ct in enumerate(contradictions, 1):
        sev = ct.get("severity", "")
        icon = {
            "direct contradiction": "🔴",
            "partial disagreement": "🟡",
            "different context": "🔵",
        }.get(sev, "⚪")
        with st.expander(f"{icon} {ct.get('topic', f'Contradiction {i}')}"):
            col_a, col_b = st.columns(2)
            pa = ct.get("paper_a", {})
            pb = ct.get("paper_b", {})
            with col_a:
                st.markdown(f"**{pa.get('title', '?')} ({pa.get('year', '?')})**")
                st.write(pa.get("position", ""))
            with col_b:
                st.markdown(f"**{pb.get('title', '?')} ({pb.get('year', '?')})**")
                st.write(pb.get("position", ""))
            badge = f"{icon} **{sev.title()}**" if sev else ""
            st.markdown(badge)
            st.markdown(f"*{ct.get('possible_explanation', '—')}*")


def render_methodology_tab(results: dict):
    import pandas as pd

    methodology = results.get("methodology_table", [])
    if not methodology:
        st.info("No methodology data available.")
        return

    display_cols = [
        "paper_title", "year", "study_design", "sample_size",
        "data_collection_method", "statistical_methods", "key_strength", "key_limitation",
    ]
    rows = [{col: m.get(col, "—") for col in display_cols} for m in methodology]
    df = pd.DataFrame(rows)

    st.dataframe(df, use_container_width=True, hide_index=True)

    csv = df.to_csv(index=False)
    st.download_button("⬇ Download as CSV", csv, "methodology_comparison.csv", "text/csv")

    patterns = st.session_state.get("methodology_patterns", [])
    if patterns:
        st.markdown("---")
        st.subheader("Key Methodological Patterns")
        for p in patterns:
            st.markdown(f"- {p}")


def render_evidence_tab(results: dict):
    scores = results.get("evidence_scores", [])
    if not scores:
        st.info("No evidence scores available.")
        return

    sorted_scores = sorted(scores, key=lambda x: safe_int(x.get("evidence_score", 0)), reverse=True)

    for s in sorted_scores:
        score_val = safe_int(s.get("evidence_score", 0))
        flag = s.get("flag", "") or ""
        is_flagged = "poorly" in flag.lower()
        prefix = "⚠️ " if is_flagged else ""

        st.markdown(f"**{prefix}{s.get('claim_text', '?')}**")
        st.progress(max(score_val, 1) / 100, text=f"{score_val}/100")

        detail_parts = []
        sp = s.get("supporting_papers")
        if sp:
            detail_parts.append(f"Supporting papers: {sp}")
        eq = s.get("evidence_quality")
        if eq:
            detail_parts.append(f"Quality: {eq}")
        if flag and flag != "null":
            detail_parts.append(f"Flag: {flag}")
        if detail_parts:
            st.caption(" · ".join(detail_parts))

        st.markdown("")


def render_gaps_tab(results: dict):
    rq = results.get("research_question", "")
    gaps = results.get("gap_analysis", "")

    if rq:
        st.info(f"**Your Research Question:** {rq}")

    if gaps:
        st.markdown(gaps)
    else:
        st.info("No gap analysis available.")


def render_lit_review_tab(results: dict):
    review = results.get("literature_review_draft", "")
    if not review:
        st.info("No literature review generated.")
        return

    st.warning(
        "⚠️ **AI-generated draft.** Verify all citations and claims before academic submission."
    )

    edited = st.text_area(
        "Literature Review Draft (editable)",
        value=review,
        height=600,
        label_visibility="collapsed",
    )

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "⬇ Download as .txt",
            edited,
            "litlens_literature_review.txt",
            "text/plain",
        )
    with col2:
        st.code(edited, language="markdown")


def render_chat(results: dict):
    """Bottom-of-tab RAG chat against uploaded papers."""
    st.divider()
    rq = results.get("research_question", "")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("sources"):
                st.caption("Sources: " + ", ".join(msg["sources"]))

    if question := st.chat_input("Ask anything about your papers..."):
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching papers..."):
                try:
                    answer, sources = rag_answer(
                        question, st.session_state.get("vector_store"), rq
                    )
                except Exception as e:
                    answer, sources = f"Error: {e}", []
                st.write(answer)
                if sources and sources != ["General Knowledge"]:
                    st.caption("Sources: " + ", ".join(sources))

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })


# ─── Display ──────────────────────────────────────────────────────────────────
def display_results(results: dict):
    tabs = st.tabs([
        "📊 Overview",
        "⚔️ Contradictions",
        "🔬 Methodology Comparison",
        "💪 Evidence Strength",
        "🔍 Research Gaps",
        "📝 Literature Review Draft",
    ])

    with tabs[0]:
        render_overview_tab(results)
    with tabs[1]:
        render_contradictions_tab(results)
    with tabs[2]:
        render_methodology_tab(results)
    with tabs[3]:
        render_evidence_tab(results)
    with tabs[4]:
        render_gaps_tab(results)
    with tabs[5]:
        render_lit_review_tab(results)

    render_chat(results)

    st.divider()
    st.caption(
        "LitLens is a research aid. Always verify claims, citations, and "
        "conclusions independently before academic use."
    )


def show_landing():
    st.markdown(
        """
Welcome to **LitLens** — your AI-powered literature analysis assistant.

Upload your research papers, define your research question, and let the platform
read, extract, compare, and synthesize the literature for you.

---

**What you get:**

| Deliverable | Description |
|---|---|
| **Claim Map** | Every significant claim extracted and categorized across all papers |
| **Contradiction Analysis** | Where papers disagree — and why |
| **Methodology Comparison** | Side-by-side comparison of study designs |
| **Evidence Scoring** | How well-supported each claim really is (0–100) |
| **Gap Analysis** | What's missing, tailored to your research question |
| **Literature Review Draft** | A thematic, citation-ready draft you can edit |

---

**How to use:**

1. Upload **2 or more research papers** (PDF) in the sidebar
2. Enter your **research question** and **domain**
3. Click **Analyze Literature**
4. Explore the results across six interactive tabs
5. Use the **chat** at the bottom to ask follow-up questions about your papers
"""
    )


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    apply_css()
    init_session()

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("# 🔬 LitLens")
        st.caption("Know your literature, faster")
        st.divider()

        st.markdown("**Step 1**")
        files = st.file_uploader(
            "Upload Research Papers (PDF)",
            type=["pdf"],
            accept_multiple_files=True,
        )
        if files:
            st.success(f"**{len(files)} papers uploaded**")

        st.markdown("**Step 2**")
        research_question = st.text_input(
            "Your Research Question",
            placeholder="e.g., What factors influence patient adherence to medication?",
        )

        st.markdown("**Step 3**")
        research_domain = st.text_input(
            "Research Domain",
            placeholder="e.g., Public Health, Machine Learning, Organizational Behavior",
        )

        can_run = bool(files and len(files) >= 2 and research_question)
        analyze = st.button(
            "🔍 Analyze Literature",
            type="primary",
            disabled=not can_run,
            use_container_width=True,
        )
        if files and len(files) < 2:
            st.warning("Upload at least 2 papers to begin analysis.")
        elif files and not research_question:
            st.warning("Enter a research question to begin.")

        if st.session_state.total_cost > 0:
            st.divider()
            st.markdown(f"**Estimated cost:** ${st.session_state.total_cost:.4f}")

    # ── Main area ──
    st.title("🔬 LitLens")
    st.caption("Know your literature, faster")

    if analyze and can_run:
        run_id = str(uuid.uuid4())[:8]
        print(f"\n{'='*60}")
        print(f"[LitLens] NEW ANALYSIS RUN — run_id={run_id}")
        print(f"[LitLens] research_question={research_question!r}")
        print(f"[LitLens] research_domain={research_domain!r}")
        print(f"[LitLens] files={[f.name for f in files]}")
        print(f"{'='*60}")

        st.session_state.results = None
        st.session_state.raw_texts = []
        st.session_state.vector_store = None
        st.session_state.chat_history = []
        st.session_state.methodology_patterns = []
        st.session_state.total_cost = 0.0
        st.session_state.total_tokens_mini = 0
        st.session_state.total_tokens_full = 0

        global _VECTOR_STORE, _RAW_TEXTS

        with st.sidebar:
            with st.spinner("Reading PDFs..."):
                raw_texts = extract_raw_texts(files)
                _RAW_TEXTS = raw_texts
                st.session_state.raw_texts = raw_texts
                print(f"[LitLens] Extracted text from {len(raw_texts)} files: "
                      f"{[(r['filename'], len(r.get('text',''))) for r in raw_texts]}")
            with st.spinner("Building search index..."):
                vs = build_vector_store(raw_texts)
                _VECTOR_STORE = vs
                st.session_state.vector_store = vs
                print(f"[LitLens] FAISS _VECTOR_STORE={'SET OK' if _VECTOR_STORE else 'FAILED — None'}")

        results = run_pipeline(research_question, research_domain or "General")
        st.session_state.results = results
        print(f"[LitLens] Pipeline returned — papers={len(results.get('papers', []))}, "
              f"rq_in_results={results.get('research_question', '')!r}")
        display_results(results)
    elif st.session_state.get("results"):
        display_results(st.session_state.results)
    else:
        show_landing()


main()
