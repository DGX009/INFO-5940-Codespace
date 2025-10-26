INFO5940 Assignment 1 — RAG App 

Overview
A Retrieval-Augmented Generation (RAG) chat app built with:
- Streamlit for the web UI
- LangChain for ingestion, chunking, retrieval, and LLM calls
- ChromaDB as the local persistent vector store

Users can upload multiple .txt/.md/.pdf files, index them into Chroma, and chat with the documents. Answers are grounded in retrieved chunks with clear citations.

Key Features
- Multiple document upload: .txt, .md, .pdf
- Efficient chunking with configurable chunk_size and chunk_overlap
- Optional Auto chunking (size/overlap adapt to document length)
- Persistent Chroma index in a per-session folder under .chroma/
- Top-k similarity retrieval → LLM answer strictly based on retrieved context
- Citations (filename + page + chunk index) in an expandable “Sources”
- Chat-style interface with multi-turn history
- “Clear local Chroma” (clear current session’s index) and “New chat (clear all)” (reset everything including uploads & index)

-------------------------------------------------------------------------------------------------------------------------------------------------------------

How to Run (GitHub Codespaces)
1) Open your fork on branch assignment1 (or your working branch), then create a Codespace.

2) Install dependencies:
   pip install -r requirements.txt

3) Set environment variables (do NOT commit keys):
   export API_KEY="YOUR_KEY"
   export OPENAI_BASE_URL="https://api.ai.it.cornell.edu"
   export OPENAI_CHAT_MODEL="openai.gpt-4o-mini"
   export OPENAI_EMBED_MODEL="openai.text-embedding-3-large"

   Notes:
   - The Cornell gateway uses prefixed model IDs (e.g., openai.gpt-4o-mini, openai.text-embedding-3-large).
   - You can list models available to your key:
     python - <<'PY'
     import os
     from openai import OpenAI
     client = OpenAI(
         api_key=os.getenv("API_KEY") or os.getenv("OPENAI_OPENAI_API_KEY"),
         base_url=os.getenv("OPENAI_BASE_URL") or "https://api.ai.it.cornell.edu",
     )
     print("\n".join(sorted(m.id for m in client.models.list().data)))
     PY

4) Launch the app:
   streamlit run chat_with_pdf.py

-------------------------------------------------------------------------------------------------------------------------------------------------------------

Usage Guide
Sidebar (left):
  Upload .txt/.md/.pdf (multiple allowed).  
  Configure Chunk size and Chunk overlap (or keep defaults).  
  Optional: enable Auto chunking (default ON) to adapt chunk size/overlap by doc length.(this way is recommended)  
  Click “Index to Chroma” to build the vector index from current uploads.  
  Click “Clear local Chroma” to delete the CURRENT SESSION index only (for example, when changing embedding model or re-building with different chunk params).  
  Click “New chat (clear all)” to reset EVERYTHING: chat history, Chroma folders, temporary upload folders, and the file uploader (like a full refresh).  

-------------------------------------------------------------------------------------------------------------------------------------------------------------

Main panel:
- After indexing, type your question in the chat input.
- Expand “Sources” below an answer to see citations.

-------------------------------------------------------------------------------------------------------------------------------------------------------------

Special Cases
- Grounded answers only: The assistant answers strictly from retrieved chunks. If the info is not in the indexed documents, it will say it doesn’t know and suggest uploading more relevant files.
- Re-index semantics:
  - “Index to Chroma” appends chunks into the same session collection. It does NOT automatically remove chunks from previously indexed files that you no longer upload.
  - To make the index match exactly the currently uploaded files, use “Clear local Chroma” before re-indexing, or click “New chat (clear all)” to start fresh.
- Model changes:
  - If you change OPENAI_EMBED_MODEL or OPENAI_CHAT_MODEL, you should “Clear local Chroma” (or “New chat”) and re-index to avoid mixing embeddings from different models.
- Session isolation:
  - Each app run uses a unique per-session directory under .chroma/. “Clear local Chroma” removes only the current session directory. “New chat (clear all)” removes the top-level .chroma/ (all sessions) and resets the uploader & temp folders.
- Image-based PDFs:
  - The app uses PyPDFLoader (text extraction). If your PDF is scanned/image-based (no selectable text), there will be little/no text to index. Use OCR first and upload the resulting text/PDF.
- Retrieval returns nothing:
  - Try larger chunk_size and/or lower overlap, upload more relevant documents, or re-index with Auto chunking ON.
- Multiple uploads and large files:
  - Auto chunking increases chunk_size for larger documents to reduce fragmentation while keeping enough overlap for context continuity.

-------------------------------------------------------------------------------------------------------------------------------------------------------------

Design Notes (Mapping to Assignment Tasks)
- Task 1: Runs in GitHub Codespaces; env variables used (no hardcoded keys); dependencies recorded.
- Task 2: .txt/.md upload; UTF-8 decoding with errors ignored.
- Task 3.1: Ingestion & Chunking
  - RecursiveCharacterTextSplitter with adjustable chunk_size/overlap, plus Auto chunking heuristic by document length.
  - chunk_idx saved in metadata for traceability.
- Task 3.2: Retrieval-Augmented Generation
  - Chroma persistent vector store per session under .chroma/.
  - Stable SHA1 IDs to avoid duplicate inserts for identical chunks.
  - Similarity retrieval (k=5) → LLM answers. System prompt enforces “answer strictly from context; otherwise say you don’t know”.
- Task 3.3: Conversational Interface
  - st.chat_input & st.chat_message with multi-turn history in st.session_state.
  - Clear feedback when no index found, when retrieval finds nothing, or when errors occur.
- Task 4: File Formats
  - .txt/.md text parsing; .pdf via PyPDFLoader (records page numbers for citations).
- Task 5: Multiple Documents
  - Accepts multiple files in one go; retrieval searches across all indexed chunks.

-------------------------------------------------------------------------------------------------------------------------------------------------------------

Security
- Never commit keys. Use environment variables or Codespaces Secrets.
- .chroma/ and temporary upload folders are ignored by git via .gitignore.

-------------------------------------------------------------------------------------------------------------------------------------------------------------

Troubleshooting
- Invalid model name (400):
  - Your gateway likely requires prefixed IDs (e.g., openai.text-embedding-3-large). List models (see command above) and set OPENAI_EMBED_MODEL / OPENAI_CHAT_MODEL accordingly.
- “No documents indexed yet”:
  - You must click “Index to Chroma” after uploading files; otherwise there’s nothing to search.
- “No sufficient information found…”:
  - Increase chunk_size, reduce overlap, enable Auto chunking, and/or upload more relevant documents; then re-index.
- “attempt to write a readonly database” or “Chroma already exists with different settings”:
  - Use “Clear local Chroma” or “New chat (clear all)” to reset the session directories, then re-index.
- Metadata type errors:
  - Already handled by sanitizing metadata before upsert; if you still hit errors, clear and re-index.

-------------------------------------------------------------------------------------------------------------------------------------------------------------

Have fun with the features