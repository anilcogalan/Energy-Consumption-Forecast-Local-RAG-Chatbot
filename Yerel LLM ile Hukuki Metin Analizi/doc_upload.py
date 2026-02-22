"""
📄 Belge Yükleme Modülü — LexGLUE Legal RAG
Hukuki belge doğrulama, metin çıkarma, chunking ve ChromaDB indexleme.
"""

import hashlib
import re
import io
import time

# ============================================
# TEXT EXTRACTION
# ============================================

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """PDF'den metin çıkar (pypdf)."""
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n\n".join(pages)


def extract_text_from_docx(file_bytes: bytes) -> str:
    """DOCX'ten metin çıkar (python-docx)."""
    import docx
    doc = docx.Document(io.BytesIO(file_bytes))
    return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())


def extract_text_from_txt(file_bytes: bytes) -> str:
    """TXT'den metin çıkar."""
    for enc in ("utf-8", "latin-1", "cp1254"):
        try:
            return file_bytes.decode(enc)
        except (UnicodeDecodeError, AttributeError):
            continue
    return file_bytes.decode("utf-8", errors="replace")


def extract_text(filename: str, file_bytes: bytes) -> str:
    """Dosya uzantısına göre metin çıkar."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext == "pdf":
        return extract_text_from_pdf(file_bytes)
    elif ext in ("docx", "doc"):
        return extract_text_from_docx(file_bytes)
    elif ext in ("txt", "text", "md"):
        return extract_text_from_txt(file_bytes)
    else:
        raise ValueError(f"Desteklenmeyen dosya formatı: .{ext}  (PDF, DOCX veya TXT yükleyin)")


# ============================================
# LEGAL VALIDATION (LLM-based)
# ============================================

LEGAL_CHECK_PROMPT = """You are a document classifier. Determine if the following text excerpt is a LEGAL document.

Legal documents include: contracts, court decisions, legislation, regulations, legal opinions, 
terms of service, privacy policies, patents, legal briefs, compliance documents, law articles,
human rights rulings, constitutional texts, treaties, and similar.

Non-legal documents include: news articles (unless about a specific case ruling), 
blog posts, marketing materials, technical manuals, recipes, fiction, etc.

TEXT EXCERPT (first 1500 chars):
---
{text_excerpt}
---

Respond with ONLY one word: LEGAL or NOT_LEGAL"""


def check_if_legal(text: str, call_llm_fn, base_url: str, model: str) -> tuple:
    """LLM ile belgenin hukuki olup olmadığını kontrol et.
    Returns: (is_legal: bool, explanation: str)
    """
    excerpt = text[:1500].strip()
    if not excerpt:
        return False, "Belge boş veya metin çıkarılamadı."
    
    response = call_llm_fn(
        LEGAL_CHECK_PROMPT.format(text_excerpt=excerpt),
        base_url, model, temperature=0.0
    )
    response_clean = response.strip().upper()
    
    if "NOT_LEGAL" in response_clean:
        return False, "Bu belge hukuki bir doküman olarak tanımlanamadı. Lütfen sözleşme, mahkeme kararı, mevzuat veya benzeri bir hukuki belge yükleyin."
    elif "LEGAL" in response_clean:
        return True, "Belge hukuki doküman olarak doğrulandı."
    else:
        # Ambiguous — varsayılan olarak kabul et ama uyar
        return True, f"Belge sınıflandırması belirsiz (LLM yanıtı: {response[:100]}). Yine de işleme alınıyor."


# ============================================
# DUPLICATE CHECK (hash-based)
# ============================================

def compute_doc_hash(text: str) -> str:
    """Belge metninden SHA-256 hash üret."""
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def is_duplicate(doc_hash: str, collection) -> bool:
    """ChromaDB'de aynı hash'e sahip belge var mı kontrol et."""
    try:
        results = collection.get(
            where={"doc_hash": doc_hash},
            limit=1
        )
        return len(results["ids"]) > 0
    except Exception:
        return False


# ============================================
# CHUNKING
# ============================================

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list:
    """Metni karakter bazında chunk'la (overlap ile).
    Kısa metinler (< chunk_size) tek chunk olarak döner.
    """
    text = re.sub(r"\s+", " ", text).strip()
    
    if len(text) <= chunk_size:
        return [text] if text else []
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        # Kelime sınırında kes
        if end < len(text):
            space_idx = text.rfind(" ", start + chunk_size - 100, end + 50)
            if space_idx > start:
                end = space_idx
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
        if start >= len(text):
            break
    
    return chunks


# ============================================
# INDEX TO CHROMADB
# ============================================

def index_document(
    text: str,
    filename: str,
    doc_hash: str,
    collection,
    chunk_size: int = 1000,
    overlap: int = 150,
) -> dict:
    """Belgeyi chunk'la ve ChromaDB'ye indexle.
    Returns: {"n_chunks": int, "doc_hash": str, "time": float}
    """
    start = time.time()
    chunks = chunk_text(text, chunk_size, overlap)
    
    if not chunks:
        return {"n_chunks": 0, "doc_hash": doc_hash, "time": 0, "error": "Metin boş, chunk oluşturulamadı."}
    
    ids = []
    documents = []
    metadatas = []
    
    for i, chunk in enumerate(chunks):
        chunk_id = f"upload_{doc_hash}_{i}"
        ids.append(chunk_id)
        documents.append(chunk)
        metadatas.append({
            "source_dataset": "user_upload",
            "split": "upload",
            "domain": "user_upload",
            "label": filename,
            "chunk_id": i,
            "total_chunks": len(chunks),
            "doc_hash": doc_hash,
            "filename": filename,
        })
    
    # Batch indexle
    batch_size = 100
    for batch_start in range(0, len(ids), batch_size):
        batch_end = batch_start + batch_size
        collection.add(
            ids=ids[batch_start:batch_end],
            documents=documents[batch_start:batch_end],
            metadatas=metadatas[batch_start:batch_end],
        )
    
    elapsed = time.time() - start
    return {
        "n_chunks": len(chunks),
        "doc_hash": doc_hash,
        "time": elapsed,
        "word_count": len(text.split()),
        "char_count": len(text),
    }