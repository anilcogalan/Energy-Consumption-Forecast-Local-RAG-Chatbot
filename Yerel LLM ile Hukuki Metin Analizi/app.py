"""
🏛️ LexGLUE Legal RAG System — Streamlit Demo
Hukuki belgelerin yerel LLM mimarisi ile işlenmesi ve sorgulanması.

Dataset: https://www.kaggle.com/datasets/thedevastator/lexglue-legal-nlp-benchmark-dataset
"""

import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
import requests
import time
import re
import json
import numpy as np
import pandas as pd
import os
import io
import html as html_module
import uuid
from datetime import datetime

from doc_upload import extract_text, check_if_legal, compute_doc_hash, is_duplicate, index_document

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="LexGLUE Legal RAG",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS
# ============================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Crimson+Pro:wght@400;600;700&family=Source+Sans+3:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global */
    .stApp {
        font-family: 'Source Sans 3', sans-serif;
    }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        color: white;
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border-left: 5px solid #e94560;
    }
    .main-header h1 {
        font-family: 'Crimson Pro', serif;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0 0 0.3rem 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        font-size: 1rem;
        opacity: 0.85;
        margin: 0;
        font-weight: 300;
    }

    /* Source cards */
    .source-card {
        background: #f8f9fc;
        border: 1px solid #e2e6ef;
        border-left: 4px solid #0f3460;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        font-size: 0.9rem;
        transition: border-color 0.2s;
    }
    .source-card:hover {
        border-left-color: #e94560;
    }
    .source-card .source-meta {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: #6b7280;
        margin-bottom: 0.4rem;
    }
    .source-card .source-text {
        color: #374151;
        line-height: 1.5;
    }

    /* Domain badges */
    .domain-badge {
        display: inline-block;
        padding: 0.15rem 0.6rem;
        border-radius: 12px;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-case_law { background: #dbeafe; color: #1e40af; }
    .badge-human_rights { background: #f3e8ff; color: #7c3aed; }
    .badge-eu_legislation { background: #fee2e2; color: #dc2626; }
    .badge-contracts { background: #e0e7ff; color: #4338ca; }
    .badge-consumer_law { background: #fef3c7; color: #d97706; }
    .badge-supreme_court { background: #ccfbf1; color: #0d9488; }

    /* RAG method cards */
    .method-card {
        background: white;
        border: 2px solid #e5e7eb;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        transition: all 0.2s;
    }
    .method-card.active {
        border-color: #0f3460;
        box-shadow: 0 4px 12px rgba(15, 52, 96, 0.15);
    }

    /* Metrics row */
    .metric-box {
        background: linear-gradient(135deg, #f8f9fc, #eef1f8);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #e2e6ef;
    }
    .metric-box .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.5rem;
        font-weight: 600;
        color: #0f3460;
    }
    .metric-box .metric-label {
        font-size: 0.78rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Answer box */
    .answer-box {
        background: white;
        border: 1px solid #e2e6ef;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        line-height: 1.7;
        font-size: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }

    /* WhatsApp-style chat bubbles */
    .chat-container { margin-bottom: 1.5rem; }
    .chat-row { display: flex; margin-bottom: 0.75rem; }
    .chat-row.user { justify-content: flex-end; }
    .chat-row.assistant { justify-content: flex-start; }
    .chat-bubble {
        max-width: 75%;
        padding: 0.75rem 1rem;
        border-radius: 18px;
        line-height: 1.5;
        font-size: 0.95rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.06);
    }
    .chat-bubble.user {
        background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
        color: #fff;
        border-bottom-right-radius: 4px;
    }
    .chat-bubble.assistant {
        background: #f0f2f5;
        color: #1f2937;
        border: 1px solid #e5e7eb;
        border-bottom-left-radius: 4px;
    }
    .chat-bubble .bubble-meta { font-size: 0.7rem; opacity: 0.85; margin-top: 0.4rem; }
    .chat-bubble.user .bubble-meta { color: #e0e7ff; }
    .chat-bubble.assistant .bubble-meta { color: #6b7280; }

    /* Sidebar — okunabilir metin */
    section[data-testid="stSidebar"] {
        background: #fafbfd;
    }
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div {
        color: #1f2937 !important;
    }
    section[data-testid="stSidebar"] .stMarkdown strong {
        color: #111827 !important;
    }
    section[data-testid="stSidebar"] input {
        color: #1f2937 !important;
        background-color: #fff !important;
        border-color: #d1d5db !important;
    }
    /* Selectbox: seçili değer (Basic RAG, Tüm Alanlar vb.) okunaklı */
    section[data-testid="stSidebar"] [data-baseweb="select"],
    section[data-testid="stSidebar"] [data-baseweb="select"] div,
    section[data-testid="stSidebar"] [data-baseweb="select"] span {
        color: #1f2937 !important;
        background-color: #ffffff !important;
    }
    section[data-testid="stSidebar"] [data-baseweb="select"] > div {
        background-color: #ffffff !important;
        border: 1px solid #d1d5db !important;
        color: #1f2937 !important;
    }
    section[data-testid="stSidebar"] [data-baseweb="select"] input,
    section[data-testid="stSidebar"] [data-baseweb="select"] [role="combobox"] {
        color: #1f2937 !important;
        background-color: #ffffff !important;
    }
    section[data-testid="stSidebar"] .stSlider label {
        color: #1f2937 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    section[data-testid="stSidebar"] .stSuccess,
    section[data-testid="stSidebar"] .stError,
    section[data-testid="stSidebar"] .stInfo {
        color: #1f2937 !important;
    }
    section[data-testid="stSidebar"] a {
        color: #0f3460 !important;
    }
    /* Sidebar: secondary ve download butonları — açık arka plan, okunaklı metin */
    section[data-testid="stSidebar"] button:not([kind="primary"]) {
        color: #1f2937 !important;
        background-color: #e5e7eb !important;
    }
    section[data-testid="stSidebar"] button:not([kind="primary"]):hover {
        background-color: #d1d5db !important;
    }
    section[data-testid="stSidebar"] button span,
    section[data-testid="stSidebar"] a[download] button,
    section[data-testid="stSidebar"] a[download] button span {
        color: #1f2937 !important;
    }
    section[data-testid="stSidebar"] a[download] button {
        background-color: #e5e7eb !important;
    }
    section[data-testid="stSidebar"] a[download]:hover button {
        background-color: #d1d5db !important;
    }
    /* Primary (kırmızı) butonlarda beyaz yazı kalsın */
    section[data-testid="stSidebar"] button[kind="primary"],
    section[data-testid="stSidebar"] button[kind="primary"] span {
        color: #ffffff !important;
    }
    /* Sidebar: dosya yükleme alanı — açık arka plan */
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] {
        background-color: #e5e7eb !important;
        border-radius: 8px;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] > div,
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {
        background-color: #e5e7eb !important;
        border-color: #d1d5db !important;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ============================================
# CONSTANTS & CONFIG
# ============================================
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "lexglue_legal_corpus"

DOMAIN_LABELS = {
    "case_law": "Case Law",
    "human_rights": "Human Rights",
    "eu_legislation": "EU Legislation",
    "contracts": "Contracts",
    "consumer_law": "Consumer Law",
    "supreme_court": "Supreme Court"
}

DOMAIN_ICONS = {
    "case_law": "📋",
    "human_rights": "⚖️",
    "eu_legislation": "🇪🇺",
    "contracts": "📝",
    "consumer_law": "🛡️",
    "supreme_court": "🏛️"
}

# ============================================
# KONUŞMA GEÇMİŞİ (ChatGPT tarzı)
# ============================================
CONVERSATIONS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".lexglue_conversations.json")


def load_conversations():
    """Kayıtlı konuşmaları dosyadan yükle."""
    if not os.path.exists(CONVERSATIONS_FILE):
        return {}
    try:
        with open(CONVERSATIONS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_conversations(conversations):
    """Konuşmaları dosyaya kaydet."""
    try:
        with open(CONVERSATIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(conversations, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def init_conversation_state():
    """Konuşma state'ini başlat veya güncelle."""
    if "conversations" not in st.session_state:
        st.session_state.conversations = load_conversations()
    if "current_conv_id" not in st.session_state:
        st.session_state.current_conv_id = None
    # Mevcut konuşmaya göre chat_messages
    if st.session_state.current_conv_id and st.session_state.current_conv_id in st.session_state.conversations:
        st.session_state.chat_messages = st.session_state.conversations[st.session_state.current_conv_id].get("messages", [])
    elif "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []


init_conversation_state()


def build_conversation_pdf(conv_id, conversations, chat_messages):
    """Konuşmayı kaynaklar ve metadata ile PDF olarak döndürür."""
    try:
        from fpdf import FPDF
    except ImportError:
        return None

    def safe(s):
        """Helvetica için Unicode/ASCII-safe metin (… → ..., Türkçe vb. replace)."""
        if not s:
            return ""
        s = str(s).replace("\r\n", "\n").replace("\u2026", "...")  # Unicode ellipsis
        return s.encode("latin-1", "replace").decode("latin-1")

    pdf = FPDF()
    pdf.set_auto_page_break(True, margin=15)
    pdf.add_page()
    # Başlık (safe ile: başlıkta … vb. olabilir)
    title = "LexGLUE Legal RAG - Konuşma"
    if conv_id and conv_id in conversations:
        title = (conversations[conv_id].get("title") or title)[:80]
    pdf.set_font("Helvetica", "B", 14)
    pdf.multi_cell(0, 8, safe(title), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 6, f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M')}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    for msg in chat_messages:
        role = msg.get("role", "")
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        label = "User" if role == "user" else "Assistant"
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(0, 7, f" {label} ", fill=True, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 9)
        pdf.multi_cell(0, 6, safe(content), new_x="LMARGIN", new_y="NEXT")
        if role == "assistant" and msg.get("sources"):
            pdf.set_font("Helvetica", "I", 8)
            pdf.cell(0, 6, "  Kaynaklar:", new_x="LMARGIN", new_y="NEXT")
            for idx, s in enumerate(msg["sources"]):
                ref = (s.get("text") or "")[:500]
                meta = s.get("metadata") or {}
                meta_str = json.dumps(meta, ensure_ascii=False)[:300] if meta else ""
                pdf.multi_cell(0, 5, safe(f"  [{idx+1}] {s.get('dataset','')} | {s.get('domain','')} | similarity: {s.get('similarity',0):.4f}"), new_x="LMARGIN", new_y="NEXT")
                if ref:
                    pdf.set_font("Helvetica", "", 7)
                    pdf.multi_cell(0, 4, safe("     " + ref[:400]), new_x="LMARGIN", new_y="NEXT")
                if meta_str:
                    pdf.multi_cell(0, 4, safe("     Metadata: " + meta_str[:250]), new_x="LMARGIN", new_y="NEXT")
                pdf.set_font("Helvetica", "I", 8)
            pdf.ln(2)
    try:
        out = pdf.output(dest="S")
        return bytes(out) if isinstance(out, bytearray) else out
    except Exception:
        return None


# ============================================
# LLM INTERFACE (notebook ile uyumlu: _extract_text_from_json + /v1/chat/completions)
# ============================================
def _extract_text_from_json(j):
    """Farklı LLM server'larının döndürebileceği JSON şemalarından metni çıkarır."""
    if not isinstance(j, dict):
        return None
    # Ollama /api/generate
    if "response" in j and isinstance(j["response"], str):
        return j["response"]
    # Diğer olası anahtarlar
    for key in ("output", "completion", "text", "content", "generated_text"):
        if key in j and isinstance(j[key], str):
            return j[key]
    # OpenAI-compatible chat completions
    try:
        return j["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        pass
    try:
        return j["choices"][0]["text"]
    except (KeyError, IndexError, TypeError):
        pass
    return None


def call_llm(prompt: str, base_url: str, model: str, temperature: float = 0.1) -> str:
    """Notebook ile aynı mantık: önce /v1/chat/completions, güvenli JSON parse."""
    headers = {"Content-Type": "application/json"}
    # 1) OpenAI-uyumlu endpoint (notebook'taki gibi)
    try:
        r = requests.post(
            f"{base_url.rstrip('/')}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
            },
            headers=headers,
            timeout=120,
        )
        if r.ok:
            text = _extract_text_from_json(r.json())
            if text is not None:
                return text
    except requests.exceptions.ConnectionError:
        pass
    except Exception:
        pass

    # 2) Ollama /api/generate fallback
    try:
        r = requests.post(
            f"{base_url.rstrip('/')}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature},
            },
            headers=headers,
            timeout=120,
        )
        if r.ok:
            text = _extract_text_from_json(r.json())
            if text is not None:
                return text
    except Exception:
        pass

    return "⚠️ LLM bağlantısı kurulamadı veya beklenmeyen cevap şeması. Sidebar'dan endpoint ve modeli kontrol edin."


# ============================================
# CHAT MEMORY (son 5 mesaj)
# ============================================
MAX_CHAT_MEMORY = 5


def clean_html_junk(content: str) -> str:
    """Kullanıcı/LLM metnindeki </div>, </p> vb. HTML etiketlerini ve etiket satırlarını kaldırır."""
    if not (content or content.strip()):
        return content or ""
    try:
        content = html_module.unescape(content)
    except Exception:
        pass
    # Önce tam olarak görünen artık bloğu kaldır (girintili satırlar dahil)
    content = re.sub(r"\s*<\s*/\s*div\s*>\s*(\s*<\s*/\s*div\s*>\s*)*", "\n", content, flags=re.IGNORECASE)
    content = re.sub(r"(\s*<\s*/\s*div\s*>\s*)+", "", content, flags=re.IGNORECASE)
    for tag in ("div", "p", "span", "br", "b", "i", "u", "script", "style", "section", "article", "main"):
        content = re.sub(rf"</\s*{tag}\s*>", "", content, flags=re.IGNORECASE)
        content = re.sub(rf"<\s*{tag}\s*>", "", content, flags=re.IGNORECASE)
        content = re.sub(rf"<\s*{tag}\s+[^>]*>", "", content, flags=re.IGNORECASE)
    content = re.sub(r"<\s*/\s*[a-zA-Z][^>]*\s*>", "", content, flags=re.IGNORECASE)
    content = re.sub(r"<[a-zA-Z][^>]*>", "", content, flags=re.IGNORECASE)
    content = re.sub(r"^\s*<\s*/\s*[a-zA-Z][^>]*\s*>\s*$", "", content, flags=re.IGNORECASE | re.MULTILINE)
    content = re.sub(r"<\s*/\s*[a-zA-Z][^>]*\s*>", "", content, flags=re.IGNORECASE)
    # Sadece boşluk/etiket kalan satırları atla
    lines = [
        ln for ln in content.split("\n")
        if ln.strip() and not re.match(r"^\s*<\s*/\s*\w+\s*>\s*$", ln, re.IGNORECASE)
    ]
    content = "\n".join(lines)
    content = re.sub(r"\n\s*\n\s*\n+", "\n\n", content)
    return re.sub(r"\n{3,}", "\n\n", content.strip())


def format_chat_context(messages):
    """Son N mesajı LLM prompt'u için metne çevirir."""
    if not messages:
        return "(önceki mesaj yok)"
    lines = []
    for m in messages:
        role = "Kullanıcı" if m["role"] == "user" else "Asistan"
        lines.append(f"{role}: {m['content'][:800]}")
    return "\n".join(lines)


# ============================================
# RAG PIPELINES
# ============================================
RAG_PROMPT = """You are a legal assistant specialized in analyzing legal documents.
Answer the question based ONLY on the provided context. If the context doesn't contain
enough information, say so clearly. Always cite which source documents support your answer.

RECENT CONVERSATION (for context; you may refer to previous Q&A):
{conversation_context}

CONTEXT:
{context}

CURRENT QUESTION: {question}

ANSWER (cite sources):"""


CLASSIFY_PROMPT = """Classify the following legal question complexity. 
Respond with ONLY one word: SIMPLE, MODERATE, or COMPLEX.

SIMPLE: Basic legal definitions, straightforward factual questions
MODERATE: Questions requiring analysis of specific legal provisions or case law
COMPLEX: Multi-faceted questions requiring cross-referencing multiple legal domains

Question: {question}

Complexity:"""


DECOMPOSE_PROMPT = """Break down this complex legal question into 2-3 simpler sub-questions.
Return ONLY the sub-questions, one per line, numbered:

Question: {question}

1.
2.
3."""


SYNTHESIZE_PROMPT = """Synthesize the following sub-answers into a comprehensive response.
Cite sources where applicable.

ORIGINAL QUESTION: {question}

SUB-ANSWERS:
{sub_answers}

SYNTHESIZED ANSWER:"""


QUERY_GEN_PROMPT = """Generate {n} different search queries to help answer this legal question.
Each query should approach from a different angle. Return ONLY the queries, one per line.

Original question: {question}

Alternative queries:"""


def basic_rag(question, collection, n_results, base_url, model, domain_filter=None, chat_history=None):
    """Basic (Naive) RAG Pipeline."""
    start = time.time()

    where = {"domain": domain_filter} if domain_filter else None
    results = collection.query(
        query_texts=[question], n_results=n_results, where=where
    )
    retrieval_time = time.time() - start

    context_parts, sources = [], []
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    )):
        sim = 1 - dist
        meta_dict = dict(meta) if meta else {}
        context_parts.append(f"[Source {i+1}: {meta_dict.get('source_dataset','')} | {meta_dict.get('domain','')}]\n{doc}")
        sources.append({
            "rank": i + 1, "dataset": meta_dict.get("source_dataset", ""),
            "domain": meta_dict.get("domain", ""), "similarity": sim,
            "text": doc[:500] if isinstance(doc, str) else str(doc)[:500],
            "metadata": meta_dict,
        })

    conversation_context = format_chat_context(chat_history or [])
    prompt = RAG_PROMPT.format(
        conversation_context=conversation_context,
        context="\n\n---\n\n".join(context_parts), question=question
    )

    llm_start = time.time()
    answer = call_llm(prompt, base_url, model)
    llm_time = time.time() - llm_start

    return {
        "answer": answer, "sources": sources, "method": "Basic RAG",
        "metrics": {
            "retrieval_time": retrieval_time, "llm_time": llm_time,
            "total_time": time.time() - start,
            "llm_calls": 1, "n_sources": len(sources),
            "avg_similarity": np.mean([s["similarity"] for s in sources]) if sources else 0
        }
    }


def adaptive_rag(question, collection, n_results, base_url, model, domain_filter=None, chat_history=None):
    """Adaptive RAG — complexity-based routing."""
    start = time.time()

    complexity = call_llm(CLASSIFY_PROMPT.format(question=question), base_url, model)
    complexity = complexity.strip().upper()
    for level in ["COMPLEX", "MODERATE", "SIMPLE"]:
        if level in complexity:
            complexity = level
            break
    else:
        complexity = "MODERATE"

    llm_calls = 1
    all_sources = []

    if complexity == "SIMPLE":
        result = basic_rag(question, collection, min(n_results, 3), base_url, model, domain_filter, chat_history)
        answer = result["answer"]
        all_sources = result["sources"]
        llm_calls += 1

    elif complexity == "MODERATE":
        result = basic_rag(question, collection, n_results, base_url, model, domain_filter, chat_history)
        answer = result["answer"]
        all_sources = result["sources"]
        llm_calls += 1

    else:  # COMPLEX
        sub_qs_raw = call_llm(DECOMPOSE_PROMPT.format(question=question), base_url, model)
        llm_calls += 1

        sub_questions = []
        for line in sub_qs_raw.strip().split("\n"):
            cleaned = re.sub(r'^[\d\.\-\)]+\s*', '', line.strip())
            if cleaned and len(cleaned) > 10:
                sub_questions.append(cleaned)
        if not sub_questions:
            sub_questions = [question]

        sub_answers = []
        for sq in sub_questions[:3]:
            sub_result = basic_rag(sq, collection, 3, base_url, model, domain_filter, chat_history=None)
            sub_answers.append(f"Q: {sq}\nA: {sub_result['answer']}")
            all_sources.extend(sub_result["sources"])
            llm_calls += 1

        answer = call_llm(SYNTHESIZE_PROMPT.format(
            question=question, sub_answers="\n\n".join(sub_answers)
        ), base_url, model)
        llm_calls += 1

        # Dedup sources
        seen = set()
        unique = []
        for s in all_sources:
            key = s["text"][:100]
            if key not in seen:
                seen.add(key)
                unique.append(s)
        all_sources = unique

    return {
        "answer": answer, "sources": all_sources, "method": "Adaptive RAG",
        "complexity": complexity,
        "metrics": {
            "total_time": time.time() - start,
            "llm_calls": llm_calls, "n_sources": len(all_sources),
            "avg_similarity": np.mean([s["similarity"] for s in all_sources]) if all_sources else 0
        }
    }


def reciprocal_rank_fusion(results_list, k=60):
    """RRF scoring."""
    fused = {}
    for results in results_list:
        for rank, r in enumerate(results):
            key = r["text"][:200]
            if key not in fused:
                fused[key] = {**r, "rrf_score": 0, "appearances": 0}
            fused[key]["rrf_score"] += 1.0 / (k + rank + 1)
            fused[key]["appearances"] += 1
    return sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)


def rag_fusion(question, collection, n_results, base_url, model, domain_filter=None, chat_history=None):
    """RAG Fusion — multi-query + RRF."""
    start = time.time()

    variants_raw = call_llm(
        QUERY_GEN_PROMPT.format(question=question, n=4), base_url, model
    )
    variants = [question]
    for line in variants_raw.strip().split("\n"):
        cleaned = re.sub(r'^[\d\.\-\)]+\s*', '', line.strip())
        if cleaned and len(cleaned) > 10:
            variants.append(cleaned)
    variants = variants[:5]

    where = {"domain": domain_filter} if domain_filter else None
    all_results = []
    for v in variants:
        res = collection.query(query_texts=[v], n_results=n_results, where=where)
        variant_results = []
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            meta_dict = dict(meta) if meta else {}
            variant_results.append({
                "rank": 0, "dataset": meta_dict.get("source_dataset", ""),
                "domain": meta_dict.get("domain", ""), "similarity": 1 - dist,
                "text": doc[:500] if isinstance(doc, str) else str(doc)[:500],
                "metadata": meta_dict,
            })
        all_results.append(variant_results)

    fused = reciprocal_rank_fusion(all_results)
    top = fused[:n_results]

    context_parts = []
    sources = []
    for i, r in enumerate(top):
        context_parts.append(f"[Source {i+1}: {r['dataset']} | RRF={r['rrf_score']:.4f}]\n{r['text']}")
        sources.append({
            "rank": i + 1, "dataset": r["dataset"], "domain": r["domain"],
            "similarity": r["rrf_score"], "appearances": r.get("appearances", 0),
            "text": r["text"], "metadata": r.get("metadata", {}),
        })

    conversation_context = format_chat_context(chat_history or [])
    prompt = RAG_PROMPT.format(
        conversation_context=conversation_context,
        context="\n\n---\n\n".join(context_parts), question=question
    )
    answer = call_llm(prompt, base_url, model)

    return {
        "answer": answer, "sources": sources, "method": "RAG Fusion",
        "query_variants": variants,
        "metrics": {
            "total_time": time.time() - start,
            "llm_calls": 2, "n_sources": len(sources),
            "n_variants": len(variants),
            "total_retrieved": sum(len(r) for r in all_results),
            "avg_similarity": np.mean([s["similarity"] for s in sources]) if sources else 0
        }
    }


# ============================================
# INIT ChromaDB
# ============================================
@st.cache_resource
def load_collection():
    """ChromaDB collection yükle."""
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        collection = client.get_collection(
            name=COLLECTION_NAME, embedding_function=embedding_fn
        )
        return collection, collection.count()
    except Exception as e:
        return None, 0


# ============================================
# SIDEBAR
# ============================================
# ChromaDB collection (sidebar belge yükleme ve ana alan için gerekli)
collection, doc_count = load_collection()

with st.sidebar:
    st.markdown("### 💬 Konuşma geçmişi")
    if st.button("➕ Yeni konuşma", use_container_width=True, type="primary"):
        st.session_state.current_conv_id = None
        st.session_state.chat_messages = []
        st.rerun()
    convs = st.session_state.get("conversations", {})
    if convs:
        # En yeni üstte
        conv_ids = sorted(convs.keys(), key=lambda cid: convs[cid].get("created_at", 0), reverse=True)
        for cid in conv_ids[:20]:
            c = convs[cid]
            title = (c.get("title") or "Konuşma")[:40]
            if len((c.get("title") or "")) > 40:
                title += "..."
            is_current = st.session_state.get("current_conv_id") == cid
            if st.button(title, key=f"conv_{cid}", use_container_width=True, type="primary" if is_current else "secondary"):
                st.session_state.current_conv_id = cid
                st.session_state.chat_messages = c.get("messages", [])
                st.rerun()
    # Seçili konuşmayı sil (sadece bir konuşma seçiliyken göster)
    if st.session_state.get("current_conv_id") and convs:
        if st.button("🗑️ Bu konuşmayı sil", use_container_width=True, type="secondary"):
            cid = st.session_state.current_conv_id
            if cid in st.session_state.conversations:
                del st.session_state.conversations[cid]
                save_conversations(st.session_state.conversations)
            st.session_state.current_conv_id = None
            st.session_state.chat_messages = []
            st.rerun()
    # PDF export — sidebar'da her zaman görünsün (konuşma varken)
    if st.session_state.get("chat_messages"):
        st.markdown("**📥 PDF olarak kaydet**")
        _pdf_bytes = build_conversation_pdf(
            st.session_state.get("current_conv_id"),
            st.session_state.get("conversations", {}),
            st.session_state.chat_messages,
        )
        if _pdf_bytes:
            _cid = st.session_state.get("current_conv_id") or "yeni"
            _fname = f"lexglue_konusma_{_cid}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
            st.download_button("Konuşmayı PDF indir", data=_pdf_bytes, file_name=_fname, mime="application/pdf", use_container_width=True, key="pdf_sidebar")
        else:
            st.caption("PDF Export devre dışı")
    st.divider()
    st.markdown("### ⚙️ Ayarlar")

    st.markdown("**LLM Bağlantısı**")
    llm_base_url = st.text_input(
        "Endpoint URL",
        value="http://localhost:1234",
        help="LM Studio veya OpenAI-uyumlu API endpoint (LM Studio varsayılan: 1234)"
    )
    llm_model = st.text_input(
        "Model",
        value="meta-llama-3-8b-instruct",
        help="LM Studio model adı: meta-llama-3-8b-instruct, mistral-7b-instruct-v0.3, llama3-8b-lawyer-v2"
    )

    st.divider()

    st.markdown("**RAG Ayarları**")
    rag_method = st.selectbox(
        "RAG Yöntemi",
        ["Basic RAG", "Adaptive RAG", "RAG Fusion"],
        index=0,
        help="Basic: tek sorgu | Adaptive: akıllı yönlendirme | Fusion: çoklu sorgu + RRF"
    )
    n_results = st.slider("Top-K Sonuç", 3, 10, 5)
    st.caption("💬 Hafıza: son 5 mesaj (takip soruları için)")

    st.divider()

    st.markdown("**Domain Filtresi**")
    domain_filter = st.selectbox(
        "Hukuki Alan",
        ["Tümü"] + list(DOMAIN_LABELS.keys()),
        format_func=lambda x: "🔍 Tüm Alanlar" if x == "Tümü" else f"{DOMAIN_ICONS.get(x, '')} {DOMAIN_LABELS.get(x, x)}"
    )
    if domain_filter == "Tümü":
        domain_filter = None

    st.divider()
    st.markdown("### 📄 Belge Yükle")
    st.caption("PDF, DOCX veya TXT — sadece hukuki belgeler kabul edilir.")
    uploaded_file = st.file_uploader(
        "Belge seçin",
        type=["pdf", "docx", "txt"],
        key="doc_uploader",
        label_visibility="collapsed",
    )
    if uploaded_file is not None:
        if st.button("📤 Belgeyi İşle ve İndexle", use_container_width=True, type="primary", key="process_upload"):
            if collection is None:
                st.error("ChromaDB bağlantısı yok — önce notebook ile DB oluşturun.")
            else:
                with st.spinner("Belge işleniyor..."):
                    try:
                        file_bytes = uploaded_file.read()
                        fname = uploaded_file.name
                        text = extract_text(fname, file_bytes)
                        if not text or len(text.strip()) < 50:
                            st.error("❌ Belgeden yeterli metin çıkarılamadı.")
                        else:
                            is_legal, legal_msg = check_if_legal(text, call_llm, llm_base_url, llm_model)
                            if not is_legal:
                                st.error(f"⛔ {legal_msg}")
                            else:
                                st.info(f"✅ {legal_msg}")
                                doc_hash = compute_doc_hash(text)
                                if is_duplicate(doc_hash, collection):
                                    st.warning("⚠️ Bu belge daha önce yüklenmiş. Tekrar indexlenmedi.")
                                else:
                                    result = index_document(text, fname, doc_hash, collection)
                                    if result.get("error"):
                                        st.error(f"❌ {result['error']}")
                                    else:
                                        st.success(
                                            f"✅ Belge indexlendi!\n\n"
                                            f"📄 {fname}\n"
                                            f"📊 {result['n_chunks']} chunk\n"
                                            f"📝 {result['word_count']:,} kelime\n"
                                            f"⏱️ {result['time']:.1f}s"
                                        )
                                        if "uploaded_docs" not in st.session_state:
                                            st.session_state.uploaded_docs = []
                                        st.session_state.uploaded_docs.append({
                                            "filename": fname,
                                            "n_chunks": result["n_chunks"],
                                            "word_count": result["word_count"],
                                            "doc_hash": doc_hash,
                                        })
                    except Exception as e:
                        st.error(f"❌ Hata: {str(e)}")
    if st.session_state.get("uploaded_docs"):
        st.markdown("**Yüklenen Belgeler:**")
        for doc in st.session_state.uploaded_docs:
            st.caption(f"📄 {doc['filename']} — {doc['n_chunks']} chunk")

    st.divider()

    # Connection status
    collection, doc_count = load_collection()
    if collection and doc_count > 0:
        st.success(f"✅ ChromaDB: {doc_count:,} chunk")
    else:
        st.error("❌ ChromaDB bağlantısı yok")
        st.info("Önce notebook'u çalıştırarak ChromaDB'yi oluşturun.")

    st.divider()
    st.markdown(
        "**Kaggle Dataset:**\n\n"
        "[LexGLUE Legal NLP Benchmark](https://www.kaggle.com/datasets/thedevastator/lexglue-legal-nlp-benchmark-dataset)"
    )
    st.caption("7 dataset • 6 hukuki alan • 236K+ belge")


# ============================================
# MAIN CONTENT
# ============================================

# Header
st.markdown("""
<div class="main-header">
    <h1>⚖️ LexGLUE Legal RAG System</h1>
    <p>Hukuki belgelerin yerel LLM mimarisi ile işlenmesi ve sorgulanması</p>
</div>
""", unsafe_allow_html=True)

# Method info tabs
method_descriptions = {
    "Basic RAG": {
        "icon": "🎯",
        "desc": "Tek sorgu → Retrieval → LLM cevabı",
        "detail": "Doğrudan semantic search ile en ilgili dokümanları bulur ve LLM'e gönderir."
    },
    "Adaptive RAG": {
        "icon": "🧠",
        "desc": "Soru analizi → Akıllı yönlendirme → Cevap",
        "detail": "Sorunun karmaşıklığını analiz eder (Simple/Moderate/Complex) ve buna göre strateji seçer."
    },
    "RAG Fusion": {
        "icon": "🔀",
        "desc": "Çoklu sorgu → RRF birleştirme → Cevap",
        "detail": "Sorudan 4-5 farklı perspektif üretir, her biriyle ayrı arama yapar, Reciprocal Rank Fusion ile birleştirir."
    }
}

info = method_descriptions[rag_method]
st.markdown(f"**{info['icon']} {rag_method}:** {info['detail']}")

st.divider()

# ============================================
# SOHBET BALONCUKLARI (WhatsApp tarzı)
# ============================================
if st.session_state.chat_messages:
    st.markdown("### 💬 Sohbet")
    # PDF indir butonunu en üstte göster (kaydırmadan görünsün)
    pdf_bytes = build_conversation_pdf(
        st.session_state.get("current_conv_id"),
        st.session_state.get("conversations", {}),
        st.session_state.chat_messages,
    )
    if pdf_bytes:
        conv_id = st.session_state.get("current_conv_id") or "yeni"
        filename = f"lexglue_konusma_{conv_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
        st.download_button(
            "📥 Konuşmayı PDF olarak kaydet (kaynaklar + metadata dahil)",
            data=pdf_bytes,
            file_name=filename,
            mime="application/pdf",
            use_container_width=True,
            key="pdf_download_top",
        )
    st.markdown("---")
    for i, msg in enumerate(st.session_state.chat_messages):
        role = msg["role"]
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        content = clean_html_junk(content)
        # HTML ile balon; geri kalanı escape et
        content_escaped = content.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
        bubble_class = "user" if role == "user" else "assistant"
        row_class = "user" if role == "user" else "assistant"
        meta = ""
        if role == "assistant" and msg.get("metrics"):
            m = msg["metrics"]
            meta = f'{m.get("total_time", 0):.1f}s • {m.get("n_sources", 0)} kaynak'
        if meta:
            meta = f'<div class="bubble-meta">{meta}</div>'
        if bubble_class == "user":
            _bg = "linear-gradient(135deg, #0f3460 0%, #16213e 100%)"
            _tc = "#fff"; _al = "right"; _br = "18px 18px 4px 18px"; _mc = "#e0e7ff"
        else:
            _bg = "#f0f2f5"
            _tc = "#1f2937"; _al = "left"; _br = "18px 18px 18px 4px"; _mc = "#6b7280"
        _meta_span = ""
        if meta and meta.strip():
            _clean_meta = meta.replace("<div", "<span").replace("</div>", "</span>")
            _meta_span = f'<br><span style="font-size:0.7rem;color:{_mc};">{_clean_meta}</span>'
        st.markdown(
            f'<div style="text-align:{_al};margin-bottom:0.75rem;">'
            f'<span style="display:inline-block;max-width:75%;padding:0.75rem 1rem;'
            f'border-radius:{_br};background:{_bg};color:{_tc};'
            f'line-height:1.5;font-size:0.95rem;text-align:left;'
            f'box-shadow:0 1px 2px rgba(0,0,0,0.06);">'
            f'{content_escaped}{_meta_span}'
            f'</span></div>',
            unsafe_allow_html=True
        )
        # Her cevap için kaynak seçeneği: tıklanınca referans + metadata
        if role == "assistant" and msg.get("sources"):
            n_src = len(msg["sources"])
            with st.expander(f"📎 Kaynaklar ({n_src}) — tıklayın, referans ve metadata görünsün", expanded=False):
                for idx, s in enumerate(msg["sources"]):
                    domain = s.get("domain", "")
                    badge_class = f"badge-{domain}"
                    icon = DOMAIN_ICONS.get(domain, "📄")
                    label = DOMAIN_LABELS.get(domain, domain)
                    score_val = s.get("similarity", 0)
                    score_label = "RRF" if s.get("appearances") else "Similarity"
                    st.markdown(f"**Kaynak {idx + 1}** — {icon} {label} · {s.get('dataset', '')} · {score_label}: {score_val:.4f}")
                    # Referans metni
                    ref_text = (s.get("text") or "").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
                    st.markdown("**Referans metni:**")
                    st.markdown(f'<div class="source-card"><div class="source-text">{ref_text}</div></div>', unsafe_allow_html=True)
                    # Metadata (tam)
                    meta = s.get("metadata") or {}
                    if meta:
                        st.markdown("**Metadata:**")
                        st.json(meta)
                    st.markdown("---")

# Örnek soru tıklanınca: widget oluşturulmadan ÖNCE key'e yazıyoruz (Streamlit buna izin veriyor)
if "example_prefill" in st.session_state and st.session_state.example_prefill:
    st.session_state.question_input_widget = st.session_state.example_prefill
    del st.session_state.example_prefill

col_input, col_examples = st.columns([3, 1])

with col_input:
    question = st.text_area(
        "💬 Hukuki sorunuzu yazın:",
        height=100,
        placeholder="Örn: What constitutes a violation of the right to fair trial under Article 6?",
        key="question_input_widget",
    )

with col_examples:
    st.markdown("**Örnek Sorular:**")
    example_questions = [
        "What constitutes a violation of the right to fair trial?",
        "Can an employer terminate a contract without notice?",
        "What are unfair terms in consumer service agreements?",
        "How do courts balance free speech and privacy?"
    ]
    for idx, eq in enumerate(example_questions):
        if st.button(eq[:50] + "..." if len(eq) > 50 else eq, key=f"example_q_{idx}", use_container_width=True):
            st.session_state.example_prefill = eq
            st.rerun()

# Search button
search_clicked = st.button(
    "🔍 Sorgula", type="primary", use_container_width=True,
    disabled=(not question or not collection)
)


# ============================================
# RESULTS
# ============================================
if search_clicked and question and collection:
    with st.spinner(f"{info['icon']} {rag_method} çalışıyor..."):
        try:
            # Son 5 mesajı RAG'e ver (hafıza katmanı)
            chat_history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.chat_messages[-MAX_CHAT_MEMORY:]
            ]
            if rag_method == "Basic RAG":
                result = basic_rag(
                    question, collection, n_results,
                    llm_base_url, llm_model, domain_filter,
                    chat_history=chat_history
                )
            elif rag_method == "Adaptive RAG":
                result = adaptive_rag(
                    question, collection, n_results,
                    llm_base_url, llm_model, domain_filter,
                    chat_history=chat_history
                )
            else:
                result = rag_fusion(
                    question, collection, n_results,
                    llm_base_url, llm_model, domain_filter,
                    chat_history=chat_history
                )

            # Sohbet baloncuklarına ekle (kullanıcı + asistan + kaynaklar); her iki mesajda da HTML artıklarını temizle
            st.session_state.chat_messages.append({"role": "user", "content": clean_html_junk(question)})
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": clean_html_junk(result["answer"]),
                "sources": result.get("sources", []),
                "metrics": result.get("metrics", {}),
            })
            st.session_state.chat_messages = st.session_state.chat_messages[-50:]

            # Konuşma geçmişine kaydet
            conv_id = st.session_state.get("current_conv_id")
            if conv_id is None:
                conv_id = str(uuid.uuid4())[:8]
                st.session_state.current_conv_id = conv_id
                st.session_state.conversations[conv_id] = {
                    "title": (question[:50] + "...") if len(question) > 50 else question,
                    "messages": list(st.session_state.chat_messages),
                    "created_at": time.time(),
                }
            else:
                st.session_state.conversations[conv_id]["messages"] = list(st.session_state.chat_messages)
                if not st.session_state.conversations[conv_id].get("title"):
                    st.session_state.conversations[conv_id]["title"] = (question[:50] + "...") if len(question) > 50 else question
            save_conversations(st.session_state.conversations)

            st.rerun()

        except Exception as e:
            st.error(f"Hata: {str(e)}")
            st.info("LLM endpoint'inin çalıştığından emin olun.")


# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#9ca3af; font-size:0.82rem; padding:0.5rem;">
    <strong>LexGLUE Legal RAG System</strong> &nbsp;|&nbsp;
    Dataset: <a href="https://www.kaggle.com/datasets/thedevastator/lexglue-legal-nlp-benchmark-dataset" target="_blank">Kaggle</a> &nbsp;|&nbsp;
    7 Dataset • 6 Hukuki Alan • 3 RAG Yöntemi &nbsp;|&nbsp;
    Embedding: all-MiniLM-L6-v2 • VectorDB: ChromaDB
</div>
""", unsafe_allow_html=True)