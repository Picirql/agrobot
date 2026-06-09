# AI Agriculture Bot – Level 1 Implementation Tasks (ChromaDB RAG)

**This file provides a clean, stand‑alone checklist for building the agriculture chatbot with Streamlit and ChromaDB.**

---

## 1️⃣ Project Initialization
1. **Create project folder** `c:/Users/rishi/Downloads/new ai stuff/ai agriculture bot` (already exists).
2. **Virtual environment**:
   ```bash
   cd "c:/Users/rishi/Downloads/new ai stuff/ai agriculture bot"
   python -m venv .venv
   .venv\Scripts\activate   # PowerShell; use activate.bat for CMD
   ```
3. **Dependencies** – create `requirements.txt` with:
   ```text
   streamlit>=1.35.0
   google-genai>=0.1.1
   pypdf>=4.0.0
   chromadb>=0.4.0
   python-dotenv>=1.0.1
   numpy>=1.20.0
   ```
   Then run `pip install -r requirements.txt`.
4. **Environment variables** – add a `.env` file (git‑ignore it) containing:
   ```env
   GEMINI_API_KEY=YOUR_GEMINI_API_KEY
   ```

---

## 2️⃣ Streamlit UI Skeleton (`app.py`)
Create `app.py` with the following sections (inline comments explain each part):
```python
import streamlit as st
import os
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader
import chromadb
from chromadb.utils import embedding_functions
import numpy as np

# ---------------------------------------------------
# Load environment & initialise Gemini client
# ---------------------------------------------------
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ---------------------------------------------------
# Initialise ChromaDB (persistent on disk)
# ---------------------------------------------------
persist_dir = "./chroma_db"
embed_fn = embedding_functions.GoogleGenerativeAIEmbeddingFunction(
    model_name="gemini-embedding-001",
    api_key=os.getenv("GEMINI_API_KEY"),
)
chroma_client = chromadb.PersistentClient(path=persist_dir)
collection = chroma_client.get_or_create_collection(
    name="agri_docs",
    embedding_function=embed_fn,
)

# ---------------------------------------------------
# Helper: PDF ingestion → chunk → add to collection
# ---------------------------------------------------
def ingest_pdf(file):
    reader = PdfReader(file)
    text = "\n".join(p.extract_text() or "" for p in reader.pages)
    # Chunk: 800 chars, 100 char overlap
    size, overlap = 800, 100
    chunks = [text[i:i+size] for i in range(0, len(text), size-overlap)]
    ids = [f"{file.name.split('.')[0]}_{i}" for i in range(len(chunks))]
    collection.add(ids=ids, documents=chunks)
    st.success(f"Added {len(chunks)} chunks from `{file.name}` to vector store.")

# ---------------------------------------------------
# Sidebar – file uploader & clear‑chat button
# ---------------------------------------------------
st.sidebar.title("📚 Document Upload & Controls")
uploaded = st.sidebar.file_uploader("Upload agricultural PDF", type="pdf")
if uploaded:
    ingest_pdf(uploaded)
if st.sidebar.button("🗑️ Clear Conversation"):
    st.session_state.messages = []
    st.rerun()

# ---------------------------------------------------
# Main chat UI
# ---------------------------------------------------
st.set_page_config(page_title="AgroBot – Farming Assistant", page_icon="🌱", layout="wide")
st.title("🌾 AgroBot – Your Farming Assistant")

# Quick‑start prompts (feel free to extend)
quick_prompts = [
    "What crops thrive in monsoon climates?",
    "How to prevent fungal infections in wheat?",
    "Optimal irrigation schedule for rice paddies?",
]
cols = st.columns(len(quick_prompts))
for col, qp in zip(cols, quick_prompts):
    if col.button(qp, key=qp):
        st.session_state.user_input = qp

# Capture user message
user_msg = st.chat_input("Ask a farming question…", key="chat_inp")
if user_msg:
    st.session_state.user_input = user_msg

if "user_input" in st.session_state:
    query = st.session_state.pop("user_input")
    # ---------------------------------------------------
    # RAG: retrieve relevant chunks from ChromaDB
    # ---------------------------------------------------
    query_emb = client.models.embed_content(
        model="gemini-embedding-001",
        contents=query,
    ).embeddings[0].values
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=3,
    )
    context = "\n\n".join(results["documents"][0])

    # ---------------------------------------------------
    # System prompt & final Gemini request
    # ---------------------------------------------------
    system_prompt = (
        "You are AgroBot, an expert agricultural assistant. "
        "Answer the user's question using the supplied context. "
        "If the context does not contain the answer, respond from your own knowledge but note that the answer is not sourced from uploaded documents."
    )
    full_prompt = f"{system_prompt}\n\n---\nContext:\n{context}\n\n---\nUser question: {query}"

    # ---------------------------------------------------
    # Stream response back to UI
    # ---------------------------------------------------
    with st.chat_message("assistant"):
        placeholder = st.empty()
        reply = ""
        stream = client.models.generate_content_stream(
            model="gemini-1.5-flash",
            contents=full_prompt,
        )
        for chunk in stream:
            if chunk.text:
                reply += chunk.text
                placeholder.markdown(reply + "▌")
        placeholder.markdown(reply)
        # store in session for history if you want to display later
```

---

## 3️⃣ Styling – Premium Dark Theme
Create `.streamlit/config.toml` with:
```toml
[theme]
primaryColor = "#2a9d8f"   # Forest‑green accent
backgroundColor = "#0f172a" # Deep slate background
secondaryBackgroundColor = "#1e293b"
textColor = "#e2e8f0"
font = "sans serif"
```
Add a small CSS block (already shown in `app.py`) to load the **Outfit** Google Font and give the quick‑start buttons a glass‑morphism hover effect.

---

## 4️⃣ Verification Checklist
1. **UI/Theme** – Verify the dark theme, custom font, and green accents appear.
2. **File Upload & Indexing** – Upload a 2‑page agriculture PDF; the sidebar should display a success message with the number of chunks added.
3. **RAG Retrieval** – Ask a question whose answer exists verbatim in the PDF. The response should include the exact phrasing from the document and optionally a note like `(Source: Uploaded PDF)`.
4. **General Knowledge** – Ask a non‑document question (e.g., *"What is photosynthesis?"*). The bot should answer and add `(Note: Answer based on general agricultural knowledge, not found in uploaded files)`.
5. **Guardrails** – Ask an off‑topic query such as *"How do I fix a leaky faucet?"*. The bot must politely decline, stating it only handles agriculture‑related topics.
6. **Persistence** – Stop the Streamlit server (`Ctrl+C`) and restart (`streamlit run app.py`). Previously uploaded PDFs should remain searchable (ChromaDB persists on disk).

---

## 5️⃣ Optional Future Extensions (Level 2+)
- **Hybrid Retrieval** – Combine keyword search with vector similarity for higher recall.
- **Metadata Enrichment** – Store page numbers, section titles, and source URLs in ChromaDB to enable citations.
- **Tool Integration** – Add a weather‑fetch function (OpenWeatherMap) via Gemini function calling.
- **Agentic Planner** – Use LangGraph or CrewAI to orchestrate multi‑step reasoning (e.g., diagnose disease → suggest treatment → fetch market price).

---

*End of task file.*
