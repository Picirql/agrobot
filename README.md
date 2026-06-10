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
5. (Optional) Add your OpenWeatherMap API key to `.env` to enable live weather lookups:
   ```env
   OPENWEATHER_API_KEY=your_openweathermap_api_key_here
   ```
6. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

## Verification Checklist

- **Styling:** Page background is deep slate-green, fonts use Outfit/Inter, and primary interactive components are forest green.
- **Quick-Start Cards:** Clicking any card inputs the question and immediately starts a streaming chat response.
- **Guardrails:** Asking an off-topic question (e.g. *"How do I fix a leaky kitchen sink?"*) results in AgroBot politely declining and explaining it's specialized in agriculture.

## Weather Tool

AgroBot can fetch live weather conditions and use them to inform its farming advice.

- **Setup:** Get a free API key from [OpenWeatherMap](https://openweathermap.org/api) and set `OPENWEATHER_API_KEY` in `.env`. Without this key, weather queries will get a polite "lookup failed" message and AgroBot falls back to general knowledge.
- **Usage:** Ask things like *"What's the weather in Hyderabad?"* or use the **Weather Check** quick-start card. Gemini automatically calls `get_weather(location)` (defined in `utils.py`) and works the live temperature, conditions, humidity, and wind speed into its reply.
- **Location format:** Accepts a city name (`"Bangalore"`) or `"lat,lon"` coordinates (`"12.97,77.59"`).
- **Caching:** Results are cached in-memory per location via `functools.lru_cache`, so repeat queries for the same location don't trigger new API calls.
