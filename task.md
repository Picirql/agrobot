## Task 1: Initialize Project Files

Create the following files in the root of the workspace directory:

### 1. `requirements.txt`
Create a `requirements.txt` file specifying:
```text
streamlit>=1.35.0
google-genai>=0.1.1
python-dotenv>=1.0.1
pypdf>=4.0.0
numpy>=1.20.0
```

### 2. `.env`
Create a `.env` file containing a placeholder for the API key:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. `.streamlit/config.toml`
Create a directory `.streamlit` and add a `config.toml` file to define our premium dark forest theme:
```toml
[theme]
primaryColor = "#2E7D32" # Forest Green
backgroundColor = "#0B0F0B" # Deep slate-green background
secondaryBackgroundColor = "#131B13" # Lighter green-tinted card background
textColor = "#ECF0EC" # Soft off-white text
font = "sans serif"
```

---

## Task 2: Create the Main Application (`app.py`)

Create `app.py` with the complete RAG chatbot implementation below.

```python
import streamlit as st
import os
import pypdf
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load env variables
load_dotenv()

# App Page Configurations
st.set_page_config(page_title="AgroBot - AI Agricultural Advisor", page_icon="🌱", layout="wide")

# Custom CSS for Premium Organic Dark Mode Look & Card Transitions
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Styled Quick Start Cards */
    .card-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 20px;
        margin: 20px 0;
    }
    
    .card {
        background: rgba(19, 27, 19, 0.6);
        border: 1px solid rgba(46, 125, 50, 0.3);
        border-radius: 12px;
        padding: 20px;
        text-align: left;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .card:hover {
        transform: translateY(-5px);
        border-color: #2E7D32;
        box-shadow: 0 8px 15px rgba(46, 125, 50, 0.2);
        background: rgba(24, 34, 24, 0.8);
    }
    
    .card h3 {
        color: #2E7D32;
        margin-top: 0;
        font-size: 1.1rem;
    }
    
    .card p {
        color: #B0C0B0;
        font-size: 0.9rem;
        margin-bottom: 0;
        line-height: 1.4;
    }
</style>
""", unsafe_allow_html=True)

# --- SYSTEM INSTRUCTION ---
SYSTEM_INSTRUCTION = """
You are AgroBot, a highly knowledgeable and supportive AI agricultural consultant. Your purpose is to assist farmers, gardeners, and agricultural students with:
1. Crop management (planting, watering, crop rotation, soil health).
2. Pest and disease identification and eco-friendly treatment.
3. Sustainable farming practices and soil enrichment tips.
4. Weather-related farming decisions.

Instructions:
- Be encouraging, professional, and practical.
- Structure your responses clearly using markdown formatting (bullet points, bold text, or tables if comparing things).
- If a document context is provided below, prioritize answering the user's question using the provided context from the uploaded documents.
- If the answer is found in the document context, clearly mention: "(Source: Uploaded PDF)".
- If the answer is NOT found in the document context, use your general agricultural knowledge to answer, but state: "(Note: Answer based on general agricultural knowledge, not found in uploaded files)".
- If a user asks a question completely unrelated to agriculture, farming, plants, weather, soil, or gardening, politely decline to answer, explaining that you are specialized in agriculture.
"""

# --- RAG HELPER FUNCTIONS ---

def parse_pdf(file) -> str:
    """Extracts raw text from a PDF file object."""
    try:
        pdf_reader = pypdf.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        st.error(f"Error parsing PDF: {e}")
        return ""

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list:
    """Splits raw text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def get_embeddings(client, texts: list) -> list:
    """Requests embeddings for a list of text chunks using Gemini text-embedding-004."""
    embeddings = []
    # Send in batches of 50 to avoid API payloads limits
    batch_size = 50
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        try:
            response = client.models.embed_content(
                model="text-embedding-004",
                contents=batch
            )
            # Add embedding values to list
            for emb in response.embeddings:
                embeddings.append(emb.values)
        except Exception as e:
            st.error(f"Error calling embedding API: {e}")
            return None
    return embeddings

def cosine_similarity(query_vector, doc_vectors) -> np.ndarray:
    """Calculates cosine similarity scores between query vector and matrix of doc vectors."""
    dot_product = np.dot(doc_vectors, query_vector)
    norm_query = np.linalg.norm(query_vector)
    norm_docs = np.linalg.norm(doc_vectors, axis=1)
    return dot_product / (norm_query * norm_docs + 1e-9)

# --- SIDEBAR & API CONFIGURATION ---

st.sidebar.title("🌱 AgroBot Control")

# Handle API Key
api_key = os.environ.get("GEMINI_API_KEY", "")
if not api_key:
    api_key = st.sidebar.text_input("Enter Gemini API Key", type="password", help="Input your key if GEMINI_API_KEY env variable is not set.")

if not api_key:
    st.info("⚠️ Please enter a Gemini API Key in the sidebar or set GEMINI_API_KEY in a .env file to begin.")
    st.stop()

# Initialize Gemini Client
client = genai.Client(api_key=api_key)

# RAG Ingestion in Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("📚 Knowledge Base (RAG)")
uploaded_file = st.sidebar.file_uploader("Upload Agricultural Manual (PDF)", type="pdf")

# Ingest and Index PDF if uploaded
if uploaded_file:
    # Use the file name to cache the index and prevent re-indexing on every rerun
    if "indexed_filename" not in st.session_state or st.session_state.indexed_filename != uploaded_file.name:
        with st.sidebar.status("📖 Reading and Indexing PDF...", expanded=True) as status:
            raw_text = parse_pdf(uploaded_file)
            if raw_text.strip():
                chunks = chunk_text(raw_text)
                status.write(f"Created {len(chunks)} text segments.")
                
                status.write("Generating vector embeddings...")
                embeddings = get_embeddings(client, chunks)
                
                if embeddings:
                    st.session_state.rag_chunks = chunks
                    st.session_state.rag_embeddings = np.array(embeddings)
                    st.session_state.indexed_filename = uploaded_file.name
                    status.update(label="✅ Indexing Complete!", state="complete", expanded=False)
                    st.sidebar.success(f"Indexed: {uploaded_file.name}")
            else:
                status.update(label="❌ Failed to parse PDF.", state="error")
else:
    # Clear index if uploader is cleared
    if "indexed_filename" in st.session_state:
        del st.session_state.rag_chunks
        del st.session_state.rag_embeddings
        del st.session_state.indexed_filename

# Clear Chat History Button
st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Clear Conversation"):
    st.session_state.messages = []
    st.rerun()

# --- MAIN CHAT INTERFACE ---

st.title("🌱 AgroBot")
st.markdown("Your digital farming assistant. Ask questions about soil, watering, pests, and crops.")

# Initialize messages list in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# If chat history is empty, show Quick Start cards
if not st.session_state.messages:
    st.markdown("### Quick Start Prompts")
    
    # 4 columns for cards
    cols = st.columns(2)
    
    cards = [
        {"title": "🌱 Crop & Soil", "text": "Suggest crop rotations to restore nitrogen in clay soil."},
        {"title": "🐛 Pest Control", "text": "Organic treatments for tomato hornworms."},
        {"title": "💧 Watering Guide", "text": "Optimal watering schedule for drip-irrigating potatoes."},
        {"title": "🍂 Disease Diagnosis", "text": "Leaf symptoms: yellowing edges and brown spots on cucumber leaves."}
    ]
    
    # Render cards as interactive Streamlit buttons wrapped in columns
    for idx, card in enumerate(cards):
        col_idx = idx % 2
        with cols[col_idx]:
            if st.button(f"{card['title']}\n\n{card['text']}", key=f"btn_{idx}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": card['text']})
                st.rerun()

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Capture user chat input
if prompt := st.chat_input("Ask about soil health, pest control, or type a farming query..."):
    # Display user input in UI
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Save user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Initialize augmented prompt
    augmented_prompt = prompt
    
    # RAG Logic - Search local vector store if indexed
    if "rag_embeddings" in st.session_state and st.session_state.rag_embeddings.size > 0:
        with st.spinner("Searching document database..."):
            try:
                # 1. Embed query
                q_response = client.models.embed_content(
                    model="text-embedding-004",
                    contents=prompt
                )
                query_vector = np.array(q_response.embeddings[0].values)
                
                # 2. Compute Cosine Similarity
                similarities = cosine_similarity(query_vector, st.session_state.rag_embeddings)
                
                # 3. Retrieve Top 3 matches
                top_indices = np.argsort(similarities)[-3:][::-1]
                
                # Retrieve matching text chunks (minimum similarity threshold of 0.25)
                context_chunks = [
                    st.session_state.rag_chunks[i]
                    for i in top_indices
                    if similarities[i] > 0.25
                ]
                
                if context_chunks:
                    context_text = "\n\n".join(context_chunks)
                    # Create the augmented prompt
                    augmented_prompt = f"""DOCUMENT CONTEXT:
{context_text}

---
USER QUESTION:
{prompt}"""
            except Exception as e:
                st.error(f"Error searching RAG index: {e}")
    
    # Generate streamed response from Gemini
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        try:
            # We call model gemini-2.5-flash with custom instructions and prompt
            stream = client.models.generate_content_stream(
                model="gemini-2.5-flash",
                contents=augmented_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION
                )
            )
            
            # Stream the answer word by word
            for chunk in stream:
                full_response += chunk.text or ""
                response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
            
            # Save assistant response to history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"Error generating AI response: {e}")
```

---

## Task 3: Local Environment Setup & Run

Execute the following shell commands in the root workspace directory:

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```
2. Activate the virtual environment:
   * **Windows (PowerShell):** `.venv\Scripts\Activate.ps1`
   * **Windows (CMD):** `.venv\Scripts\activate.bat`
   * **Linux/macOS:** `source .venv/bin/activate`
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

---

## Task 4: Verification Checkpoints

1. **Verify Home Screen:** Make sure the dashboard loads with a forest-green theme, the sidebar has a file uploader, and 4 quick-start buttons are displayed in the main pane.
2. **Verify Basic Chat:** Click a quick-start card or enter a question. Ensure answers stream in dynamically.
3. **Verify Restrictive Guardrails:**
   * Ask: *"How do I fix a flat bicycle tire?"*
   * Expect: The bot should refuse to answer, stating that it only answers questions related to agriculture, plants, and soil.
4. **Verify RAG Functionality:**
   * Upload an agricultural manual PDF in the sidebar. Wait for the green "Index Complete" status.
   * Ask a question specific to the uploaded file.
   * Expect: The bot should answer using the context and append `(Source: Uploaded PDF)` at the end of the matching points.
   * Ask a general question (e.g. *"What is photosynthesis?"*) that is not in the document.
   * Expect: The bot should answer from general knowledge and append `(Note: Answer based on general agricultural knowledge, not found in uploaded files)`.
