import os
import re
import time

import chromadb
import streamlit as st
from chromadb import Documents, EmbeddingFunction, Embeddings
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError
from pypdf import PdfReader

from utils import get_weather

load_dotenv(override=True)

MODEL_NAME = "gemini-2.5-flash"
EMBEDDING_MODEL_NAME = "gemini-embedding-001"
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "agri_docs"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
EMBED_BATCH_SIZE = 100  # Gemini's embed_content allows at most 100 texts per request
EMBED_BATCH_PAUSE_SECONDS = 2  # spacing between batches to stay under per-minute rate limits

SYSTEM_INSTRUCTION = """
You are AgroBot, a highly knowledgeable and supportive AI agricultural consultant. Your purpose is to assist farmers, gardeners, and agricultural students with:
1. Crop management (planting, watering, crop rotation, soil health).
2. Pest and disease identification and eco-friendly treatment.
3. Sustainable farming practices and soil enrichment tips.
4. Weather-related farming decisions.

Instructions:
- Be encouraging, professional, and practical.
- Structure your responses clearly using markdown formatting (bullet points, bold text, or tables if comparing things).
- If a user asks a question completely unrelated to agriculture, farming, plants, weather, soil, or gardening, politely decline to answer, explaining that you are specialized in agriculture.
- If the user provides symptoms of crop/plant issues, ask clarifying questions (e.g., region, soil type, watering habits) if necessary, or provide organic and chemical control options.
- When a user asks about current weather conditions for a location, use the get_weather tool to fetch live data and incorporate it into farming-relevant advice. If the tool returns an error, let the user know the weather lookup failed and answer from general knowledge instead.
"""

QUICK_PROMPTS = [
    {
        "icon": "🌱",
        "title": "Crop Rotation & Soil",
        "prompt": "Suggest crop rotations to restore nitrogen in clay soil.",
    },
    {
        "icon": "🐛",
        "title": "Pest Control",
        "prompt": "Organic treatments for tomato hornworms.",
    },
    {
        "icon": "☀️",
        "title": "Watering Guide",
        "prompt": "Optimal watering schedule for drip-irrigating potatoes.",
    },
    {
        "icon": "🌾",
        "title": "Disease Diagnosis",
        "prompt": "Leaf symptoms: yellowing edges and brown spots on cucumber leaves.",
    },
    {
        "icon": "🌦️",
        "title": "Weather Check",
        "prompt": "What's the current weather in Mumbai, and how should it affect my farming plans today?",
    },
]

st.set_page_config(
    page_title="AgroBot · AI Agriculture Consultant",
    page_icon="🌿",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS — premium "dark forest" look & feel
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

    html, body {
        font-family: 'Outfit', 'Inter', sans-serif;
    }

    /* ---- Page header ---- */
    .agrobot-hero {
        padding: 0.25rem 0 1rem 0;
    }
    .agrobot-hero h1 {
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.1rem;
    }
    .agrobot-hero p {
        color: #9FB3A0;
        font-size: 1.02rem;
        margin-top: 0;
    }

    /* ---- Quick-action cards (rendered as Streamlit buttons in columns) ---- */
    div[data-testid="column"] .stButton > button {
        height: 100%;
        min-height: 132px;
        width: 100%;
        border-radius: 18px;
        border: 1px solid rgba(46, 125, 50, 0.30);
        background: linear-gradient(150deg, #131B13 0%, #0E140E 100%);
        color: #ECF0EC;
        padding: 1.1rem 1.2rem;
        font-weight: 500;
        font-size: 0.95rem;
        text-align: left;
        white-space: pre-wrap;
        line-height: 1.45;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.28);
        transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
    }
    div[data-testid="column"] .stButton > button:hover {
        transform: translateY(-6px);
        border-color: #2E7D32;
        box-shadow: 0 14px 30px rgba(46, 125, 50, 0.32);
        color: #ECF0EC;
    }
    div[data-testid="column"] .stButton > button:active,
    div[data-testid="column"] .stButton > button:focus {
        color: #ECF0EC;
        border-color: #66BB6A;
    }

    /* ---- Sidebar buttons (e.g. Clear Conversation) ---- */
    section[data-testid="stSidebar"] .stButton > button {
        border-radius: 12px;
        border: 1px solid rgba(46, 125, 50, 0.4);
        background-color: rgba(46, 125, 50, 0.12);
        color: #ECF0EC;
        font-weight: 500;
        transition: background-color 0.18s ease, transform 0.18s ease;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background-color: rgba(46, 125, 50, 0.28);
        transform: translateY(-2px);
        border-color: #2E7D32;
        color: #ECF0EC;
    }

    /* ---- Chat container & bubble refinements ---- */
    [data-testid="stChatMessage"] {
        background-color: #131B13;
        border: 1px solid rgba(46, 125, 50, 0.16);
        border-radius: 16px;
        padding: 0.95rem 1.2rem;
        margin-bottom: 0.9rem;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.22);
    }
    [data-testid="stChatMessageContent"] p {
        line-height: 1.65;
    }
    [data-testid="stChatInput"] {
        border-radius: 14px;
        border: 1px solid rgba(46, 125, 50, 0.35) !important;
    }

    /* ---- Section labels ---- */
    .agrobot-section-label {
        color: #9FB3A0;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.5rem 0 0.75rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# API key setup — env var first, fall back to a sidebar prompt
# ---------------------------------------------------------------------------
api_key = os.environ.get("GEMINI_API_KEY", "").strip()
if api_key == "your_gemini_api_key_here":
    api_key = ""

if not api_key:
    with st.sidebar:
        st.markdown("### 🔑 Gemini API Key")
        st.caption("No `GEMINI_API_KEY` found in the environment. Enter it below to chat with AgroBot.")
        entered_key = st.text_input(
            "Gemini API Key",
            type="password",
            value=st.session_state.get("gemini_api_key", ""),
            placeholder="AIza...",
            label_visibility="collapsed",
        )
        if entered_key:
            st.session_state.gemini_api_key = entered_key

    api_key = st.session_state.get("gemini_api_key", "")

if not api_key:
    st.markdown(
        '<div class="agrobot-hero"><h1>🌿 AgroBot</h1>'
        "<p>Your AI agriculture consultant for crops, pests, soil, and sustainable farming.</p></div>",
        unsafe_allow_html=True,
    )
    st.info("👋 Please enter your Gemini API key in the sidebar to start chatting with AgroBot.")
    st.stop()

# ---------------------------------------------------------------------------
# Client initialization
# ---------------------------------------------------------------------------
client = genai.Client(api_key=api_key)


# ---------------------------------------------------------------------------
# ChromaDB knowledge base — persists uploaded PDFs on disk for retrieval
# ---------------------------------------------------------------------------
class GeminiEmbeddingFunction(EmbeddingFunction):
    """Embeds documents and queries with Gemini's embedding model for ChromaDB."""

    def __init__(self, genai_client: genai.Client, model_name: str = EMBEDDING_MODEL_NAME):
        self._client = genai_client
        self._model_name = model_name

    def __call__(self, input: Documents) -> Embeddings:
        texts = list(input)
        embeddings: Embeddings = []
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            if start > 0:
                time.sleep(EMBED_BATCH_PAUSE_SECONDS)
            batch = texts[start:start + EMBED_BATCH_SIZE]
            response = self._client.models.embed_content(model=self._model_name, contents=batch)
            embeddings.extend(embedding.values for embedding in response.embeddings)
        return embeddings

    @staticmethod
    def name() -> str:
        return "gemini-embedding-function"

    def get_config(self) -> dict:
        return {"model_name": self._model_name}

    @staticmethod
    def build_from_config(config: dict) -> "GeminiEmbeddingFunction":
        return GeminiEmbeddingFunction(client, model_name=config.get("model_name", EMBEDDING_MODEL_NAME))


chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=GeminiEmbeddingFunction(client),
)


def ingest_pdf(uploaded_file) -> int:
    """Chunk an uploaded PDF and add it to the knowledge base. Returns the chunk count."""
    reader = PdfReader(uploaded_file)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    step = CHUNK_SIZE - CHUNK_OVERLAP
    chunks = [
        chunk.strip()
        for chunk in (text[i:i + CHUNK_SIZE] for i in range(0, len(text), step))
        if chunk.strip()
    ]
    if not chunks:
        return 0

    base_id = re.sub(r"[^A-Za-z0-9]+", "_", uploaded_file.name.rsplit(".", 1)[0])
    offset = collection.count()
    collection.add(
        ids=[f"{base_id}_{offset + i}" for i in range(len(chunks))],
        documents=chunks,
        metadatas=[{"source": uploaded_file.name} for _ in chunks],
    )
    return len(chunks)


def retrieve_context(query: str, n_results: int = 3) -> list[str]:
    """Return the most relevant knowledge-base chunks for a query, if any have been indexed."""
    available = collection.count()
    if available == 0:
        return []
    try:
        results = collection.query(query_texts=[query], n_results=min(n_results, available))
    except APIError:
        # Embedding the query hit a transient API error (e.g. rate limit) — fall back to
        # answering from general knowledge rather than failing the whole chat turn.
        return []
    return results["documents"][0]


RAG_PROMPT_TEMPLATE = """Reference excerpts retrieved from the user's uploaded documents:

{context}

Using the excerpts above, answer the user's question below.
- If the excerpts directly answer the question, base your answer on them and end your reply with "(Source: Uploaded PDF)".
- If the excerpts are not relevant or don't contain the answer, answer from your own agricultural knowledge and end your reply with "(Note: Answer based on general agricultural knowledge, not found in uploaded files)".

User question: {query}"""


def build_model_input(query: str) -> str:
    """Augment the user's question with retrieved context when the knowledge base has matches."""
    chunks = retrieve_context(query)
    if not chunks:
        return query
    return RAG_PROMPT_TEMPLATE.format(context="\n\n---\n\n".join(chunks), query=query)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🌿 AgroBot")
    st.caption("AI consultant for farming, gardening & crop health")
    st.divider()

    st.markdown("#### 📚 Knowledge Base")
    st.caption("Upload agricultural PDFs to ground AgroBot's answers in your own documents.")
    uploaded_pdf = st.file_uploader("Upload agricultural PDF", type="pdf", label_visibility="collapsed")
    if uploaded_pdf is not None:
        upload_results = st.session_state.setdefault("upload_results", {})
        if uploaded_pdf.file_id not in upload_results:
            try:
                with st.spinner(f"Indexing `{uploaded_pdf.name}`..."):
                    chunk_count = ingest_pdf(uploaded_pdf)
            except APIError:
                # Record the attempt regardless of outcome so a rate-limit error doesn't
                # retrigger ingestion (and burn more quota) on every subsequent rerun.
                upload_results[uploaded_pdf.file_id] = ("error", uploaded_pdf.name)
            else:
                upload_results[uploaded_pdf.file_id] = ("ok", uploaded_pdf.name, chunk_count)

        outcome = upload_results[uploaded_pdf.file_id]
        if outcome[0] == "ok":
            st.success(f"Added {outcome[2]} chunks from `{outcome[1]}` to the knowledge base.")
        else:
            st.error(
                f"Couldn't index `{outcome[1]}` — Gemini's embedding rate limit was hit. "
                "Wait about a minute, then remove and re-upload the file to retry."
            )
    if collection.count() > 0:
        st.caption(f"📖 Knowledge base holds **{collection.count()}** indexed chunks.")

    st.divider()
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.caption("Powered by Gemini · `gemini-2.5-flash`")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="agrobot-hero"><h1>🌿 AgroBot</h1>'
    "<p>Your AI agriculture consultant — ask about crops, pests, soil health, irrigation, and more.</p></div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Quick-start prompt cards (only shown before the conversation starts)
# ---------------------------------------------------------------------------
if not st.session_state.messages:
    st.markdown('<p class="agrobot-section-label">Quick start</p>', unsafe_allow_html=True)
    cols = st.columns(5)
    for col, item in zip(cols, QUICK_PROMPTS):
        with col:
            card_label = f"{item['icon']}  {item['title']}\n\n{item['prompt']}"
            if st.button(card_label, key=f"quick_{item['title']}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": item["prompt"]})
                st.rerun()

# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def stream_agrobot_response(prompt: str):
    """Yield text chunks from Gemini for use with st.write_stream."""
    try:
        response_stream = client.models.generate_content_stream(
            model=MODEL_NAME,
            contents=build_model_input(prompt),
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                tools=[get_weather],
            ),
        )
        for chunk in response_stream:
            if chunk.text:
                yield chunk.text
    except APIError:
        yield (
            "⚠️ Gemini's API hit a temporary error and couldn't complete this response. "
            "Please try asking again in a moment."
        )


# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------
user_prompt = st.chat_input("Ask AgroBot about crops, pests, soil, watering, weather...")
if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

# ---------------------------------------------------------------------------
# Generate a streamed assistant reply whenever the latest message has no
# response yet (covers both manual chat input and quick-start card clicks).
# ---------------------------------------------------------------------------
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    last_prompt = st.session_state.messages[-1]["content"]
    with st.chat_message("assistant"):
        full_response = st.write_stream(stream_agrobot_response(last_prompt))
    st.session_state.messages.append({"role": "assistant", "content": full_response})
