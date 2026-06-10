from unittest.mock import patch

import requests

from utils import _fetch_weather, get_weather

SAMPLE_RESPONSE = {
    "main": {"temp": 28.5, "humidity": 70},
    "weather": [{"description": "scattered clouds"}],
    "wind": {"speed": 3.6},
    "name": "Mumbai",
    "sys": {"country": "IN"},
}


@patch("utils.requests.get")
def test_get_weather_builds_correct_request_and_parses_response(mock_get, monkeypatch):
    monkeypatch.setenv("OPENWEATHER_API_KEY", "test-key")
    _fetch_weather.cache_clear()

    mock_get.return_value.raise_for_status.return_value = None
    mock_get.return_value.json.return_value = SAMPLE_RESPONSE

    result = get_weather("Mumbai")

    args, kwargs = mock_get.call_args
    assert args[0] == "https://api.openweathermap.org/data/2.5/weather"
    assert kwargs["params"]["q"] == "Mumbai"
    assert kwargs["params"]["appid"] == "test-key"
    assert kwargs["params"]["units"] == "metric"

    assert result == {
        "temperature": 28.5,
        "description": "scattered clouds",
        "humidity": 70,
        "wind_speed": 3.6,
        "city": "Mumbai",
        "country": "IN",
    }


@patch("utils.requests.get")
def test_get_weather_uses_lat_lon_params(mock_get, monkeypatch):
    monkeypatch.setenv("OPENWEATHER_API_KEY", "test-key")
    _fetch_weather.cache_clear()

    mock_get.return_value.raise_for_status.return_value = None
    mock_get.return_value.json.return_value = SAMPLE_RESPONSE

    get_weather("12.97,77.59")

    _, kwargs = mock_get.call_args
    assert kwargs["params"]["lat"] == "12.97"
    assert kwargs["params"]["lon"] == "77.59"
    assert "q" not in kwargs["params"]


@patch("utils.requests.get")
def test_get_weather_caches_repeat_calls(mock_get, monkeypatch):
    monkeypatch.setenv("OPENWEATHER_API_KEY", "test-key")
    _fetch_weather.cache_clear()

    mock_get.return_value.raise_for_status.return_value = None
    mock_get.return_value.json.return_value = SAMPLE_RESPONSE

    get_weather("Pune")
    get_weather("Pune")

    assert mock_get.call_count == 1


def test_get_weather_missing_api_key_returns_error(monkeypatch):
    monkeypatch.delenv("OPENWEATHER_API_KEY", raising=False)

    result = get_weather("Delhi")

    assert "error" in result


@patch("utils.requests.get")
def test_get_weather_request_error_is_not_cached(mock_get, monkeypatch):
    monkeypatch.setenv("OPENWEATHER_API_KEY", "test-key")
    _fetch_weather.cache_clear()

    mock_get.return_value.raise_for_status.side_effect = [
        requests.HTTPError("404"),
        None,
    ]
    mock_get.return_value.json.return_value = SAMPLE_RESPONSE

    first = get_weather("Atlantis")
    assert "error" in first

    second = get_weather("Atlantis")
    assert "error" not in second
    assert mock_get.call_count == 2
