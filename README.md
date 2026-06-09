# 🌿 AgroBot — AI Agriculture Consultant

A Streamlit-based chatbot specialized in farming, gardening, and crop health, powered by the `google-genai` SDK and the `gemini-2.5-flash` model.

## Setup & Run

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
4. Add your Gemini API key to `.env` (or enter it in the sidebar at runtime):
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   ```
5. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

## Verification Checklist

- **Styling:** Page background is deep slate-green, fonts use Outfit/Inter, and primary interactive components are forest green.
- **Quick-Start Cards:** Clicking any card inputs the question and immediately starts a streaming chat response.
- **Guardrails:** Asking an off-topic question (e.g. *"How do I fix a leaky kitchen sink?"*) results in AgroBot politely declining and explaining it's specialized in agriculture.
