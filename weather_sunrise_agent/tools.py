"""
Tools for the Weather & Sunrise/Sunset Agent.

Uses:
  - Open-Meteo API   (free, no key needed) for weather
  - Sunrise-Sunset.org API (free, no key needed) for sun times
"""

import requests
from datetime import datetime, timezone


# ─────────────────────────────────────────────
# Weather code → human-readable description map
# ─────────────────────────────────────────────
WMO_CODES = {
    0: "Clear sky",
    1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Icy fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
    77: "Snow grains",
    80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
    85: "Slight snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail",
}


def get_weather(latitude: float, longitude: float) -> dict:
    """
    Fetch current weather conditions for a given latitude and longitude.

    Args:
        latitude:  Geographic latitude  (e.g. 40.7128 for New York)
        longitude: Geographic longitude (e.g. -74.0060 for New York)

    Returns:
        A dictionary with temperature, feels-like, humidity, wind speed,
        precipitation, weather description, and timezone.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": [
            "temperature_2m",
            "apparent_temperature",
            "relative_humidity_2m",
            "wind_speed_10m",
            "wind_direction_10m",
            "weather_code",
            "precipitation",
            "cloud_cover",
            "surface_pressure",
        ],
        "wind_speed_unit": "kmh",
        "timezone": "auto",
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return {"error": f"Weather API request failed: {exc}"}

    cur = data.get("current", {})
    code = cur.get("weather_code", -1)

    return {
        "latitude": latitude,
        "longitude": longitude,
        "timezone": data.get("timezone", "Unknown"),
        "local_time": cur.get("time", "N/A"),
        "temperature_c": cur.get("temperature_2m"),
        "feels_like_c": cur.get("apparent_temperature"),
        "humidity_percent": cur.get("relative_humidity_2m"),
        "wind_speed_kmh": cur.get("wind_speed_10m"),
        "wind_direction_deg": cur.get("wind_direction_10m"),
        "precipitation_mm": cur.get("precipitation"),
        "cloud_cover_percent": cur.get("cloud_cover"),
        "pressure_hpa": cur.get("surface_pressure"),
        "weather_code": code,
        "weather_description": WMO_CODES.get(code, "Unknown condition"),
    }


def get_sunrise_sunset(latitude: float, longitude: float) -> dict:
    """
    Fetch today's sunrise and sunset times for a given latitude and longitude.

    Args:
        latitude:  Geographic latitude  (e.g. 40.7128 for New York)
        longitude: Geographic longitude (e.g. -74.0060 for New York)

    Returns:
        A dictionary with sunrise, sunset, solar noon, day length,
        civil/nautical/astronomical twilight times in UTC.
    """
    url = "https://api.sunrise-sunset.org/json"
    params = {
        "lat": latitude,
        "lng": longitude,
        "formatted": 0,   # ISO 8601 UTC times
        "date": "today",
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return {"error": f"Sunrise/Sunset API request failed: {exc}"}

    if data.get("status") != "OK":
        return {"error": f"Sunrise/Sunset API returned status: {data.get('status')}"}

    r = data["results"]

    def fmt(iso: str) -> str:
        """Convert ISO 8601 UTC string to a readable HH:MM UTC label."""
        try:
            dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
            return dt.strftime("%H:%M UTC")
        except Exception:
            return iso

    # day_length comes back as total seconds
    day_sec = r.get("day_length", 0)
    hours, remainder = divmod(int(day_sec), 3600)
    minutes = remainder // 60

    return {
        "latitude": latitude,
        "longitude": longitude,
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "sunrise_utc": fmt(r.get("sunrise", "")),
        "sunset_utc": fmt(r.get("sunset", "")),
        "solar_noon_utc": fmt(r.get("solar_noon", "")),
        "day_length": f"{hours}h {minutes}m",
        "civil_twilight_begin_utc": fmt(r.get("civil_twilight_begin", "")),
        "civil_twilight_end_utc": fmt(r.get("civil_twilight_end", "")),
        "nautical_twilight_begin_utc": fmt(r.get("nautical_twilight_begin", "")),
        "nautical_twilight_end_utc": fmt(r.get("nautical_twilight_end", "")),
        "astronomical_twilight_begin_utc": fmt(r.get("astronomical_twilight_begin", "")),
        "astronomical_twilight_end_utc": fmt(r.get("astronomical_twilight_end", "")),
    }


def get_full_report(latitude: float, longitude: float) -> dict:
    """
    Fetch a combined weather AND sunrise/sunset report for a location.

    Args:
        latitude:  Geographic latitude  (e.g. 40.7128 for New York)
        longitude: Geographic longitude (e.g. -74.0060 for New York)

    Returns:
        A dictionary containing both weather and sun time data.
    """
    weather = get_weather(latitude, longitude)
    sun = get_sunrise_sunset(latitude, longitude)
    return {
        "weather": weather,
        "sun_times": sun,
    }
