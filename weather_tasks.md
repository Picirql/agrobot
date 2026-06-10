# Weather‑Tool Tasks for AI Agriculture Bot

Use this checklist with Claude Code (or any developer) to add the `get_weather(location)` capability while preserving the existing ChromaDB RAG pipeline.

---

## 1️⃣ Update Dependencies
- [ ] Add **requests** to `requirements.txt` (keep existing lines).
  ```diff
  + requests>=2.32.0
  ```
- [ ] Run `pip install -r requirements.txt`.

## 2️⃣ Store the Weather API Key
- [ ] Append to `.env`:
  ```env
  OPENWEATHER_API_KEY=YOUR_OPENWEATHER_API_KEY_HERE
  ```
- [ ] Verify the key loads with `os.getenv("OPENWEATHER_API_KEY")`.

## 3️⃣ Create `utils.py` (Weather Helper)
- [ ] In the project root, create `utils.py` containing only the cached weather helper:
  ```python
  """Utility functions for external data sources used by the Agriculture Bot.
  Currently provides the `get_weather` function.
  """
  import os
  import requests
  from functools import lru_cache
  from typing import Dict, Any

  @lru_cache(maxsize=128)
  def get_weather(location: str) -> Dict[str, Any]:
      """Fetch current weather from OpenWeatherMap.

      Args:
          location: City name (e.g., "Delhi") or "lat,lon" string.

      Returns:
          dict with temperature (°C), description, humidity, wind speed, city, country.
      """
      api_key = os.getenv("OPENWEATHER_API_KEY")
      if not api_key:
          raise RuntimeError("OPENWEATHER_API_KEY not set in environment.")

      base_url = "https://api.openweathermap.org/data/2.5/weather"
      params = {"appid": api_key, "units": "metric"}

      if "," in location:
          lat, lon = location.split(",", 1)
          params.update({"lat": lat.strip(), "lon": lon.strip()})
      else:
          params["q"] = location

      resp = requests.get(base_url, params=params, timeout=10)
      resp.raise_for_status()
      data = resp.json()

      return {
          "temperature": data["main"]["temp"],
          "description": data["weather"][0]["description"],
          "humidity": data["main"]["humidity"],
          "wind_speed": data["wind"]["speed"],
          "city": data.get("name"),
          "country": data["sys"].get("country"),
      }
  ```

## 4️⃣ Amend `app.py`
### 4.1 Imports
- [ ] Replace the existing utility import line with:
  ```python
  from utils import get_weather
  ```
### 4.2 Define Gemini Function Schema (weather only)
- [ ] Add a single entry to the `tools` list:
  ```python
  tools = [
      {
          "name": "get_weather",
          "description": "Fetch current weather conditions for a given location.",
          "parameters": {
              "type": "object",
              "properties": {
                  "location": {
                      "type": "string",
                      "description": "City name or latitude,longitude (e.g., 'Bangalore' or '12.97,77.59')."
                  }
              },
              "required": ["location"]
          }
      }
  ]
  ```
### 4.3 Tool‑call Handling Logic
- [ ] Replace any multi‑tool branching with a weather‑only block:
  ```python
  if function_name == "get_weather":
      result = get_weather(arguments["location"])
  else:
      result = {"error": f"Unsupported tool {function_name}"}
  ```
### 4.4 (Optional) Show Raw Weather JSON
- [ ] After the function call, optionally display:
  ```python
  st.info(f"🛰️ Weather data:\n{json.dumps(result, indent=2)}")
  ```
### 4.5 Add Quick‑Start Weather Card
- [ ] Insert a new card into the quick‑start list (anywhere you define them):
  ```python
  {"title": "☀️ Weather Query", "text": "Current weather in Mumbai"}
  ```

## 5️⃣ Verification Checklist (Weather‑Focused)
- [ ] App starts without errors and loads `OPENWEATHER_API_KEY`.
- [ ] Query *"What is the weather in Hyderabad?"* triggers the `get_weather` tool; final answer includes temperature, description, etc.
- [ ] Re‑issuing the same query hits the cache (no new HTTP request).
- [ ] Invalid location yields a polite error message.
- [ ] Off‑topic queries still receive the agriculture‑only refusal.
- [ ] New quick‑start weather card appears and works.
- [ ] Existing RAG functionality (PDF upload, retrieval) remains unchanged.

## 6️⃣ Documentation Update
- [ ] Add a **Weather Tool** section to `README.md` describing the API key setup, usage examples, and caching behavior.

## 7️⃣ (Optional) Unit Test
- [ ] Add `tests/test_weather.py` that mocks `requests.get` and verifies:
  * Correct URL/parameters are built.
  * Proper parsing of the JSON response.
  * Cache hit after the first call.

---

**When ready**, hand this `weather_tasks.md` file to Claude Code and ask it to execute the checklist in order.
