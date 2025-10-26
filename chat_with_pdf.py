# Task 1: Provided Codespace Setup (runs in this environment; read secrets from env)
# Task 2: .txt upload (UI + backend handling)
# Task 3.1: Ingestion & Chunking
# Task 3.2: RAG Pipeline (Embeddings -> Chroma -> Retrieval -> LLM grounded answer)
# Task 3.3: Conversational Interface (multi-turn chat, clear feedback)
# Task 4: Support .txt and .pdf
# Task 5: Multiple documents

import os
import hashlib
import tempfile
import time
from typing import List, Tuple

import streamlit as st

# Task 3.1: text splitter (compatible imports for different LC versions)
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except Exception:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

# Task 3.x: Document object (compatible imports for different LC versions)
try:
    from langchain.schema import Document
except Exception:
    from langchain_core.documents import Document

# Task 3.2: Vector store + OpenAI adapters
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Task 1: App boot in Codespace (basic Streamlit setup + env-config)
st.set_page_config(page_title="RAG · INFO5940", page_icon="📚", layout="wide")
st.title("Retrieval-Augmented Q&A")

# Read secrets from environment; do NOT hardcode keys (security requirement of Task 1)
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL") or os.getenv("BASE_URL") or "https://api.ai.it.cornell.edu"
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
EMB_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

# Local persistent vector DB path + collection name (used by Tasks 3.1–3.2)
PERSIST_DIR = ".chroma"
COLLECTION = "assignment1"

# Session state for Task 3.3 (multi-turn chat) and Task 5 (track which docs are indexed)
if "ready" not in st.session_state:
    st.session_state.ready = False
if "indexed_docs" not in st.session_state:
    st.session_state.indexed_docs = set()
if "history" not in st.session_state:
    st.session_state.history = []
if "persist_dir" not in st.session_state:
    st.session_state.persist_dir = os.path.join(PERSIST_DIR, f"session_{int(time.time())}")
    os.makedirs(st.session_state.persist_dir, exist_ok=True)
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = f"uploader_{int(time.time())}"
if "tmp_dirs" not in st.session_state:
    st.session_state.tmp_dirs = []

# Task 4 + Task 2: Loaders for .pdf and .txt/.md (backend handling)
def save_upload_to_tmp(uploaded_file) -> str:
    """Persist an uploaded file to a temp path (PDF loaders need a file path)."""
    tmpdir = tempfile.mkdtemp(prefix="uploads_")
    st.session_state.tmp_dirs.append(tmpdir)
    path = os.path.join(tmpdir, uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.read())
    return path

def load_documents(files) -> List[Document]:
    docs: List[Document] = []
    for f in files:
        name = f.name
        ext = name.split(".")[-1].lower()
        if ext in ("txt", "md"):
            text = f.read().decode("utf-8", errors="ignore")
            f.seek(0)
            docs.append(Document(page_content=text, metadata={"source": name}))
        elif ext == "pdf":
            from langchain_community.document_loaders import PyPDFLoader
            path = save_upload_to_tmp(f)
            pages = PyPDFLoader(path).load()
            for p in pages:
                meta = dict(p.metadata or {})
                meta["source"] = name
                page = meta.get("page", meta.get("page_number"))
                if page is not None:
                    try:
                        meta["page"] = int(page)
                    except Exception:
                        meta["page"] = str(page)
                else:
                    meta.pop("page", None)
                docs.append(Document(page_content=p.page_content, metadata=meta))
        else:
            st.warning(f"Unsupported file type: {name}")
    return docs

# Task 3.1: Chunking strategy (configurable size/overlap; add chunk_idx for traceability)
def chunk_documents(docs: List[Document], chunk_size: int = 1200, overlap: int = 150, auto: bool = True) -> List[Document]:
    out: List[Document] = []

    def choose_params(n_chars: int) -> tuple[int, int]:
        if n_chars <= 2000:
            return n_chars, 0
        if n_chars <= 15000:
            return max(chunk_size, 1000), max(overlap, 120)
        if n_chars <= 60000:
            return max(chunk_size, 1600), max(overlap, 200)
        return max(chunk_size, 2200), max(overlap, 260)

    for d in docs:
        if not auto:
            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
            pieces = splitter.split_documents([d])
        else:
            cs, ov = choose_params(len(d.page_content or ""))
            splitter = RecursiveCharacterTextSplitter(chunk_size=cs, chunk_overlap=ov)
            pieces = splitter.split_documents([d])
        out.extend(pieces)

    for i, d in enumerate(out):
        d.metadata["chunk_idx"] = i
    return out

# Task 3.2: Vector store wiring (create/get Chroma; index chunks with stable IDs)
def get_vectorstore(emb) -> Chroma:
    try:
        vs = Chroma(collection_name=COLLECTION, embedding_function=emb, persist_directory=st.session_state.persist_dir)
    except TypeError:
        vs = Chroma(collection_name=COLLECTION, persist_directory=st.session_state.persist_dir, embedding_function=emb)
    return vs

def sanitize_metadata(meta: dict) -> dict:
    if not meta:
        return {}
    cleaned = {}
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            cleaned[k] = v
        else:
            cleaned[k] = str(v)
    return cleaned

def index_chunks(chunks: List[Document], emb) -> Tuple[Chroma, int]:
    vs = get_vectorstore(emb)
    ids = []
    cleaned = []
    for d in chunks:
        d.metadata = sanitize_metadata(d.metadata)
        base = d.page_content + str(d.metadata.get("source")) + str(d.metadata.get("page")) + str(d.metadata.get("chunk_idx"))
        ids.append(hashlib.sha1(base.encode("utf-8")).hexdigest())
        cleaned.append(d)
    vs.add_documents(cleaned, ids=ids)
    vs.persist()
    return vs, len(ids)

# Task 3.2: Retrieval + generation (RAG core)
def retrieve(vs: Chroma, query: str, k: int = 5) -> List[Document]:
    retriever = vs.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(query)

def call_llm(context: str, question: str) -> str:
    llm = ChatOpenAI(api_key=API_KEY, base_url=BASE_URL, model=CHAT_MODEL, temperature=0.2)
    system = "Answer strictly based on the provided context. If the context is insufficient, say you don't know and suggest uploading more relevant files."
    prompt = f"{system}\n\n# Context\n{context}\n\n# Question\n{question}\n\n# Answer"
    resp = llm.invoke(prompt)
    return getattr(resp, "content", str(resp))

def format_citations(docs: List[Document]) -> str:
    cites = []
    for d in docs:
        src = d.metadata.get("source")
        page = d.metadata.get("page")
        idx = d.metadata.get("chunk_idx")
        tag = f"{src}" + (f" p.{page}" if page is not None else "") + (f" #{idx}" if idx is not None else "")
        cites.append(tag)
    seen, uniq = set(), []
    for c in cites:
        if c not in seen:
            uniq.append(c); seen.add(c)
    return ", ".join(uniq)

# Task 2 + Task 4 + Task 5: UI to upload multiple .txt/.md/.pdf and index
with st.sidebar:
    st.header("Ingestion & Index")
    uploaded_files = st.file_uploader(
        "Upload .txt/.md/.pdf",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True,
        key=st.session_state.uploader_key,
    )
    chunk_size = st.number_input("Chunk size", 200, 4000, 1200, 100)
    overlap = st.number_input("Chunk overlap", 0, 800, 150, 10)
    auto_chunk = st.checkbox("Auto chunking", value=True)
    do_index = st.button("Index to Chroma")
    do_clear = st.button("Clear local Chroma")
    if do_clear:
        import shutil
        old_dir = st.session_state.get("persist_dir", PERSIST_DIR)
        if os.path.isdir(old_dir):
            try:
                shutil.rmtree(old_dir, ignore_errors=True)
            except Exception as e:
                st.warning(f"Partial cleanup: {e}")
        st.session_state.persist_dir = os.path.join(PERSIST_DIR, f"session_{int(time.time())}")
        os.makedirs(st.session_state.persist_dir, exist_ok=True)
        st.session_state.indexed_docs = set()
        st.session_state.ready = False
        st.success("Cleared .chroma/ for this session")

    st.divider()
    if st.button("New chat (clear all)"):
        import shutil
        try:
            if os.path.isdir(PERSIST_DIR):
                shutil.rmtree(PERSIST_DIR, ignore_errors=True)
        except Exception as e:
            st.warning(f"Partial cleanup: {e}")
        for d in st.session_state.get("tmp_dirs", []):
            try:
                shutil.rmtree(d, ignore_errors=True)
            except Exception:
                pass
        st.session_state.tmp_dirs = []
        st.session_state.history = []
        st.session_state.indexed_docs = set()
        st.session_state.ready = False
        st.session_state.persist_dir = os.path.join(PERSIST_DIR, f"session_{int(time.time())}")
        os.makedirs(st.session_state.persist_dir, exist_ok=True)
        st.session_state.uploader_key = f"uploader_{int(time.time())}"
        st.rerun()

# Execute indexing when user clicks the button (Tasks 3.1 + 3.2 end-to-end)
if do_index:
    if not uploaded_files:
        st.warning("Please upload files first.")
    else:
        try:
            emb = OpenAIEmbeddings(api_key=API_KEY, base_url=BASE_URL, model=EMB_MODEL)
        except Exception as e:
            st.error(f"Failed to init embeddings: {e}")
            st.stop()
        os.makedirs(st.session_state.persist_dir, exist_ok=True)
        docs = load_documents(uploaded_files)
        chunks = chunk_documents(docs, chunk_size=chunk_size, overlap=overlap, auto=auto_chunk)
        vs, n = index_chunks(chunks, emb)
        for f in uploaded_files:
            st.session_state.indexed_docs.add(f.name)
        st.session_state.ready = True
        st.success(f"Indexed {n} chunks")
        st.caption(f"Indexed files: {', '.join(sorted(st.session_state.indexed_docs)) or '(none)'}")

# Task 3.3: Conversational interface (multi-turn chat + grounded citations)
st.subheader("Chat")

for role, content, cites in st.session_state.history:
    with st.chat_message(role):
        st.markdown(content)
        if cites:
            with st.expander("Sources"):
                st.markdown(cites)

query = st.chat_input("Ask something about your uploaded documents…")

if query:
    if not st.session_state.ready:
        st.warning("No documents indexed yet. Upload and index first.")
        st.stop()
    try:
        emb = OpenAIEmbeddings(api_key=API_KEY, base_url=BASE_URL, model=EMB_MODEL)
        vs = get_vectorstore(emb)
        retrieved = retrieve(vs, query, k=5)
        st.session_state.history.append(("user", query, None))
        with st.chat_message("user"):
            st.markdown(query)
        if not retrieved:
            answer = "No sufficient information found in the indexed documents."
            st.session_state.history.append(("assistant", answer, None))
            with st.chat_message("assistant"):
                st.markdown(answer)
        else:
            context = "\n\n---\n\n".join([f"[{i+1}] {d.page_content[:1200]}" for i, d in enumerate(retrieved)])
            answer = call_llm(context, query)
            cites = format_citations(retrieved)
            st.session_state.history.append(("assistant", answer, cites))
            with st.chat_message("assistant"):
                st.markdown(answer)
                with st.expander("Sources"):
                    st.markdown(cites)
    except Exception as e:
        st.error(f"Query failed: {e}")
