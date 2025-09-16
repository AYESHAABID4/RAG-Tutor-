"""
RAG Tutor - Streamlit app (Gemini version, multi-PDF support)
- Upload one or more PDFs
- Ingest into ChromaDB (chunks + embeddings)
- Ask questions (answers only from uploaded books, with page citations)
"""

import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import numpy as np
import os
import io
import hashlib
from typing import List, Dict
import google.generativeai as genai
from dotenv import load_dotenv

# --------------------------
# Load environment and configure Gemini
# --------------------------
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

st.set_page_config(page_title="RAG Tutor", layout="wide")

# --------------------------
# Embedding model
# --------------------------
@st.cache_resource
def load_embed_model(model_name: str = "all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)

# --------------------------
# PDF Extraction Function
# --------------------------
def extract_text_by_page(pdf_bytes: bytes) -> List[Dict]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages.append({"page": i + 1, "text": text})
    return pages
# --------------------------
# PDF  Chuncking
# --------------------------
def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        chunks.append(" ".join(chunk_words))
        i += chunk_size - overlap
    return chunks
# --------------------------
# Embedding
# --------------------------
def compute_embeddings(model, texts: List[str]) -> np.ndarray:
    return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
# --------------------------
# Cosine Similarity
# --------------------------
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# --------------------------
# Chroma DB Function
# --------------------------
def get_chroma_client(persist_directory: str):
    os.makedirs(persist_directory, exist_ok=True)
    try:
        settings = Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_directory)
        client = chromadb.Client(settings)
    except Exception:
        client = chromadb.Client()
    return client

def create_or_get_collection(client, name: str):
    try:
        coll = client.get_collection(name)
    except Exception:
        coll = client.create_collection(name=name)
    return coll

# --------------------------
# Ingest PDF Function
# --------------------------
def ingest_pdf_bytes(
    pdf_bytes: bytes,
    filename: str,
    model,
    persist_root: str = "chroma_persist",
    chunk_size: int = 300,
    overlap: int = 50
):
    pages = extract_text_by_page(pdf_bytes)
    all_texts, all_ids, all_metadatas = [], [], []

    for p in pages:
        page_num = p["page"]
        page_text = p["text"] or ""
        page_chunks = chunk_text(page_text, chunk_size=chunk_size, overlap=overlap)
        for ci, chunk in enumerate(page_chunks):
            uid = f"{hashlib.md5((filename + str(page_num) + str(ci)).encode()).hexdigest()}"
            all_ids.append(uid)
            all_texts.append(chunk)
            all_metadatas.append({"page": page_num, "source": filename, "chunk_id": ci})

    if not all_texts:
        return {"error": "No text extracted from PDF."}

    embeddings = compute_embeddings(model, all_texts).tolist()

    file_hash = hashlib.md5(filename.encode()).hexdigest()
    persist_dir = os.path.join(persist_root, file_hash)
    client = get_chroma_client(persist_dir)
    collection_name = "book"
    coll = create_or_get_collection(client, collection_name)

    try:
        coll.add(
            ids=all_ids,
            documents=all_texts,
            embeddings=embeddings,
            metadatas=all_metadatas
        )
        try:
            client.persist()
        except Exception:
            pass
    except Exception:
        client.delete_collection(collection_name)
        coll = create_or_get_collection(client, collection_name)
        coll.add(
            ids=all_ids,
            documents=all_texts,
            embeddings=embeddings,
            metadatas=all_metadatas
        )
        try:
            client.persist()
        except Exception:
            pass

    return {
        "client": client,
        "collection": coll,
        "persist_dir": persist_dir,
        "n_chunks": len(all_texts),
        "filename": filename
    }

# --------------------------
# Retrieval
# --------------------------
def retrieve_top_k(question: str, model, collection, top_k: int = 3):
    q_emb = compute_embeddings(model, [question])[0].tolist()
    res = collection.query(query_embeddings=[q_emb], n_results=top_k, include=["documents", "metadatas", "distances"])
    docs = res.get("documents", [[]])[0]
    metadatas = res.get("metadatas", [[]])[0]
    distances = res.get("distances", [[]])[0] if "distances" in res else [None] * len(docs)

    doc_embs = compute_embeddings(model, docs)
    q_vec = np.array(q_emb)
    out = []
    for i, d in enumerate(docs):
        meta = metadatas[i] if i < len(metadatas) else {}
        dist = distances[i] if i < len(distances) else None
        sim = cosine_sim(q_vec, doc_embs[i])
        out.append({"doc": d, "metadata": meta, "distance": dist, "similarity": sim})
    out = sorted(out, key=lambda x: x["similarity"], reverse=True)
    return out

# --------------------------
# Gemini Answering
# --------------------------
SYSTEM_PROMPT = """You are RAG Tutor. Answer the user's question *only* using the provided book excerpts.
- DO NOT use any outside knowledge.
- If the answer cannot be found in the excerpts, respond exactly: ❌ Insufficient evidence in the uploaded book to answer this question.
- Always include citations like [Page 12].
- Be concise and factual.
"""

def generate_answer_with_gemini(question: str, top_chunks: List[Dict], model_name="gemini-1.5-flash"):
    model = genai.GenerativeModel(model_name)
    context_parts = []
    for c in top_chunks:
        page = c["metadata"].get("page", "?")
        context_parts.append(f"[Page {page}]\n{c['doc']}")
    context_text = "\n\n-----\n\n".join(context_parts)

    prompt = f"""{SYSTEM_PROMPT}

Book excerpts:
{context_text}

Question: {question}
"""

    response = model.generate_content(prompt)
    return response.text.strip() if response and response.text else None

# --------------------------
# Streamlit UI
# --------------------------

st.set_page_config(page_title="RAG Tutor", layout="wide")
st.title(" **RAG Tutor** - Book-based AI Learning Assistant")
st.write("Upload one or more PDFs and ask questions. Answers are returned **only** from the uploaded books with page citations.")

# --------------------------
# Sidebar settings (collapsible)
# --------------------------
with st.sidebar.expander("**Settings**"):
    chunk_size = st.slider("Chunk size (words)", 100, 800, 300, step=50)
    overlap = st.slider("Chunk overlap (words)", 0, 200, 50, step=10)
    top_k = st.slider("Top chunks retrieved", 1, 8, 3)
    similarity_threshold = st.slider("Similarity threshold (0-1)", 0.0, 1.0, 0.45, step=0.01)
    gemini_model = st.selectbox("Gemini model", ["gemini-1.5-flash", "gemini-1.5-pro"], index=0)

# --------------------------
# Tabs for multi-PDF workflow
# --------------------------
tab1, tab2 = st.tabs(["📁 Manage PDFs", "❓ Ask Questions"])

# --------------------------
# Tab 1: Upload & Manage PDFs
# --------------------------
with tab1:
    uploaded_files = st.file_uploader("Upload PDF books", type=["pdf"], accept_multiple_files=True)

    if "collections" not in st.session_state:
        st.session_state.collections = {}

    for uploaded_file in uploaded_files:
        filename = uploaded_file.name
        if filename not in st.session_state.collections:
            bytes_data = uploaded_file.getvalue()
            st.info(f"Ingesting '{filename}' — extracting text, chunking, creating embeddings...")
            model = load_embed_model()
            with st.spinner(f"Processing {filename}..."):
                result = ingest_pdf_bytes(
                    pdf_bytes=bytes_data,
                    filename=filename,
                    model=model,
                    persist_root="chroma_persist",
                    chunk_size=chunk_size,
                    overlap=overlap
                )
            if result.get("error"):
                st.error(f"Ingest failed for {filename}: {result['error']}")
            else:
                st.success(f"Ingested {result['n_chunks']} chunks from '{filename}'.")
                st.session_state.collections[filename] = {
                    "persist_dir": result["persist_dir"],
                    "n_chunks": result["n_chunks"],
                    "collection_name": "book"
                }

    # Display all uploaded PDFs
    if st.session_state.collections:
        st.subheader("Uploaded PDFs")
        for fname, info in st.session_state.collections.items():
            cols = st.columns([3, 1])
            cols[0].markdown(f"**{fname}** — {info['n_chunks']} chunks")
            if cols[1].button(f"❌ Remove", key=f"remove_{fname}"):
                del st.session_state.collections[fname]
                st.experimental_rerun()

# --------------------------
# Tab 2: Ask Questions
# --------------------------
with tab2:
    if not st.session_state.collections:
        st.info("Upload at least one PDF to begin asking questions.")
    else:
        selected_pdf = st.selectbox("Select PDF to query", list(st.session_state.collections.keys()))
        question = st.text_input("Ask a question about the book:")

        if question:
            model = load_embed_model()
            client = get_chroma_client(st.session_state.collections[selected_pdf]["persist_dir"])
            coll = create_or_get_collection(client, st.session_state.collections[selected_pdf]["collection_name"])
            with st.spinner("Retrieving relevant excerpts..."):
                retrieved = retrieve_top_k(question, model, coll, top_k=top_k)

            st.markdown("### 🔎 Retrieved excerpts")
            for r in retrieved:
                page = r["metadata"].get("page", "?")
                sim = r["similarity"]
                with st.expander(f"Page {page} — similarity {sim:.3f}"):
                    st.write(r["doc"][:1000] + ("..." if len(r["doc"]) > 1000 else ""))
                    st.json(r["metadata"])

            best_sim = retrieved[0]["similarity"] if retrieved else 0.0
            if best_sim < similarity_threshold:
                st.warning("❌ Similarity too low — insufficient evidence.")
                st.markdown("**Answer:** ❌ Insufficient evidence in the uploaded book to answer this question.")
            else:
                top_chunks = retrieved[:top_k]
                if os.getenv("GOOGLE_API_KEY"):
                    try:
                        with st.spinner("Asking Gemini..."):
                            answer = generate_answer_with_gemini(question, top_chunks, gemini_model)
                    except Exception as e:
                        st.error(f"Gemini call failed: {e}")
                        answer = None
                else:
                    answer = None

                if answer:
                    st.markdown("### ✅ Answer")
                    st.write(answer)
                    cited_pages = sorted({c["metadata"].get("page") for c in top_chunks})
                    st.markdown(f"**Citations:** {', '.join(f'[Page {p}]' for p in cited_pages)}")
                else:
                    top = top_chunks[0]
                    page = top["metadata"].get("page", "?")
                    excerpt = top["doc"]
                    words = excerpt.split()
                    take = 250
                    short = " ".join(words[:take]) + ("..." if len(words) > take else "")
                    st.markdown("### ✂️ Fallback answer (extractive)")
                    st.write(short)
                    st.markdown(f"**Cited:** [Page {page}]")

