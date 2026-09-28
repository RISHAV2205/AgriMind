"""Weather-agent adapter backed by the Open-Meteo geocoding and forecast APIs."""

from datetime import datetime, timezone

import httpx


WMO_WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    80: "Rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    95: "Thunderstorm",
}


class WeatherAgent:
    """Retrieves current weather facts for a farmer-supplied place name."""

    name = "weather_agent"
    source = "Open-Meteo"
    geocoding_url = "https://geocoding-api.open-meteo.com/v1/search"
    forecast_url = "https://api.open-meteo.com/v1/forecast"

    def __init__(self, client: httpx.Client | None = None) -> None:
        self._client = client or httpx.Client(timeout=httpx.Timeout(10.0))

    def _coordinates_for(self, location: str) -> dict:
        try:
            response = self._client.get(self.geocoding_url, params={"name": location, "count": 1})
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise RuntimeError("Weather provider could not resolve the location.") from error

        results = response.json().get("results", [])
        if not results:
            raise ValueError(f"Location '{location}' was not found.")
        return results[0]

    def get_forecast(self, location: str) -> dict:
        """Resolve a location then normalize the provider's current-weather response."""
        cleaned_location = location.strip()
        if not cleaned_location:
            raise ValueError("A non-empty location is required for weather.")

        place = self._coordinates_for(cleaned_location)
        requested_variables = ",".join(
            [
                "temperature_2m",
                "relative_humidity_2m",
                "precipitation",
                "precipitation_probability",
                "weather_code",
                "wind_speed_10m",
            ]
        )
        try:
            response = self._client.get(
                self.forecast_url,
                params={
                    "latitude": place["latitude"],
                    "longitude": place["longitude"],
                    "current": requested_variables,
                    "timezone": "auto",
                },
            )
            response.raise_for_status()
            current = response.json()["current"]
        except (httpx.HTTPError, KeyError, TypeError) as error:
            raise RuntimeError("Weather provider could not return a current forecast.") from error

        weather_code = current.get("weather_code")
        return {
            "agent": self.name,
            "status": "success",
            "data": {
                "location": ", ".join(
                    part for part in [place.get("name"), place.get("admin1"), place.get("country")]
                    if part
                ),
                "temperature_c": current.get("temperature_2m"),
                "humidity_percent": current.get("relative_humidity_2m"),
                "rain_probability_percent": current.get("precipitation_probability"),
                "precipitation_mm": current.get("precipitation"),
                "wind_speed_kmh": current.get("wind_speed_10m"),
                "condition": WMO_WEATHER_CODES.get(weather_code, "Unknown condition"),
                "observed_at": current.get("time"),
                "source": self.source,
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
            },
            "warnings": [],
        }
