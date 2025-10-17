# INFO5940 Assignment 1 — RAG App (Streamlit + LangChain + Chroma)

## Overview
A Retrieval-Augmented Generation (RAG) chat app with:
- Streamlit for the web UI
- LangChain for ingestion, chunking, retrieval, and LLM calls
- ChromaDB as the local persistent vector store

Users can upload multiple `.txt` and `.pdf` files, index them into Chroma, and chat with the documents. Answers are grounded in retrieved chunks with clear citations.

## Features
- Multiple document upload: `.txt`, `.md`, `.pdf`
- Efficient chunking with configurable `chunk_size` and `chunk_overlap`
- Persistent Chroma index saved under `.chroma/`
- Top-k similarity retrieval → LLM answer strictly based on context
- Citations of sources (filename + page + chunk index)
- Chat-style interface with multi-turn history

## How to Run (GitHub Codespaces)
1) Open your fork on branch `assignment1`, then create a Codespace.
2) Install dependencies:
   $ pip install -r requirements.txt
3) Set environment variables (do not commit keys):
   $ export API_KEY="<your key>"
   $ export OPENAI_BASE_URL="https://api.ai.it.cornell.edu"
   $ export OPENAI_CHAT_MODEL="openai.gpt-4o-mini"
   $ export OPENAI_EMBED_MODEL="openai.text-embedding-3-large"

   Cornell’s gateway uses prefixed model IDs (e.g., openai.text-embedding-3-large). To list models your key can use:
   $ python - <<'PY'
import os
from openai import OpenAI
client = OpenAI(api_key=os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL") or "https://api.ai.it.cornell.edu")
print("\n".join(sorted(m.id for m in client.models.list().data)))
PY

4) Launch the app:
   $ streamlit run chat_with_pdf.py
5) In the sidebar:
   - Upload one or more `.txt/.pdf`
   - Optionally adjust Chunk size and Chunk overlap
   - Click “Index to Chroma”
   - Ask questions in the chat input; expand “Sources” to view citations
   - Use “Clear local Chroma” to wipe the index when changing embedding models

## App Behavior
- The app answers only from retrieved document chunks. If the documents do not contain the information, it will say “I don’t know. Please upload more relevant files.”
- When you change the embedding model, you must Clear local Chroma and re-index.
- Re-index anytime after uploading (e.g., after changing chunk parameters).

## Design Notes
### Ingestion & Chunking (Task 3.1)
- RecursiveCharacterTextSplitter with defaults `chunk_size=1200`, `chunk_overlap=150`.
- Overlap preserves sentence continuity; chunk size balances recall with token cost.

### Vector Store (Task 3.2)
- Chroma persistent store at `.chroma/`, collection `assignment1`.
- Each chunk metadata: `source` (filename), `page` (for PDFs), `chunk_idx`.
- Stable SHA1 IDs used to avoid duplicate inserts of the same chunk content.

### Retrieval & Generation (Task 3.2)
- Similarity retrieval with `k=5` by default.
- LLM prompt instructs: answer strictly based on provided context; otherwise say “I don’t know”.

### Formats (Task 4)
- `.txt/.md`: UTF-8 decoding.
- `.pdf`: parsed via PyPDFLoader, recording `page` for citations.

### Conversational UI (Task 3.3)
- Built with `st.chat_input` and `st.chat_message`, history stored in `st.session_state`.

## Configuration (environment variables)
- API_KEY
- OPENAI_BASE_URL (e.g., https://api.ai.it.cornell.edu)
- OPENAI_CHAT_MODEL (e.g., openai.gpt-4o-mini)
- OPENAI_EMBED_MODEL (e.g., openai.text-embedding-3-large)

## Project Structure
.
├── chat_with_pdf.py          (Streamlit app: RAG pipeline + UI)
├── requirements.txt          (Dependencies)
├── ref-log.md                (References & GenAI usage log)
├── .gitignore                (Ignore .chroma/, temp uploads, caches)
├── data/                     (Sample documents)
└── .chroma/                  (Chroma persistence, generated, ignored by git)

## Changes to Template (Task 1)
- Kept the provided Codespaces setup.
- Updated requirements.txt by adding: chromadb, pypdf, langchain-text-splitters (kept original packages as requested).
- Implemented the RAG app in chat_with_pdf.py.
- Added .gitignore entries for .chroma/, temp uploads, caches.

## Security
- Never commit keys. Use environment variables or Codespaces Secrets.
- `.chroma/` is in `.gitignore` to avoid committing local embeddings/indexes.

## Troubleshooting
- Invalid model name (400): your gateway likely uses prefixed IDs. List models and set OPENAI_EMBED_MODEL / OPENAI_CHAT_MODEL to one of the returned IDs.
- “No documents indexed yet”: click “Index to Chroma” after uploading files.
- Missing packages: run `pip install -r requirements.txt` again.
- Changed embedding model: “Clear local Chroma”, then re-index.

## Grading Checklist Mapping
- Task 1: Runs in Codespaces; README explains setup; dependencies documented.
- Task 2: `.txt` upload UI + backend handling implemented.
- Task 3.1: Ingestion & chunking with adjustable parameters.
- Task 3.2: Retrieval (Chroma) + LLM answers grounded in context with citations.
- Task 3.3: Multi-turn chat interface with feedback.
- Task 4: `.txt` and `.pdf` supported; PDFs parsed with pages.
- Task 5: Multiple document uploads and retrieval across them.
