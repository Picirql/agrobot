"""Utility functions for external data sources used by the Agriculture Bot.

Currently provides the `get_weather` function.
"""
import os
from functools import lru_cache
from typing import Any, Dict

import requests


@lru_cache(maxsize=128)
def _fetch_weather(location: str, api_key: str) -> Dict[str, Any]:
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


def get_weather(location: str) -> Dict[str, Any]:
    """Get the current weather conditions for a location.

    Args:
        location: City name (e.g. "Bangalore") or "lat,lon" coordinates (e.g. "12.97,77.59").

    Returns:
        A dict with temperature (Celsius), description, humidity, wind speed, city, and
        country, or a dict with an "error" key if the lookup failed.
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return {"error": "OPENWEATHER_API_KEY is not configured."}

    try:
        return _fetch_weather(location, api_key)
    except requests.RequestException as exc:
        return {"error": f"Could not fetch weather for '{location}': {exc}"}
