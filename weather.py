import math

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
           "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]


def deg_to_compass(deg: float) -> str:
    return COMPASS[round(deg / 22.5) % 16]


def get_wind(
    lat: float,
    lng: float,
    date_utc: str,       # "YYYY-MM-DD"
    start_hour_utc: int, # 0-23
    duration_h: float,   # activity duration in hours
) -> tuple[float | None, float | None]:
    """
    Return (wind_speed_ms, wind_from_deg) averaged over the activity window,
    using Open-Meteo ERA5 reanalysis (free, no key, ~1 km resolution).
    Returns (None, None) if data is unavailable (e.g. too recent or network error).
    """
    try:
        r = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude":          round(lat, 4),
                "longitude":         round(lng, 4),
                "start_date":        date_utc,
                "end_date":          date_utc,
                "hourly":            "wind_speed_10m,wind_direction_10m",
                "wind_speed_unit":   "ms",
                "timezone":          "UTC",
            },
            timeout=10,
            verify=False,
        )
        r.raise_for_status()
        data = r.json()

        times  = data["hourly"]["time"]
        speeds = data["hourly"]["wind_speed_10m"]
        dirs   = data["hourly"]["wind_direction_10m"]

        end_hour = min(start_hour_utc + math.ceil(duration_h), 23)
        sel_s, sel_d = [], []
        for t, s, d in zip(times, speeds, dirs):
            h = int(t[11:13])
            if start_hour_utc <= h <= end_hour and s is not None and d is not None:
                sel_s.append(s)
                sel_d.append(d)

        if not sel_s:
            return None, None

        avg_speed = sum(sel_s) / len(sel_s)
        # Circular mean for direction
        sin_s = sum(math.sin(math.radians(d)) for d in sel_d)
        cos_s = sum(math.cos(math.radians(d)) for d in sel_d)
        avg_dir = (math.degrees(math.atan2(sin_s, cos_s)) + 360) % 360

        # ERA5 wind is at 10 m; scale to cyclist height (~1.5 m) via atmospheric power law
        # v(z) = v_10m × (z / 10)^0.14  →  (1.5/10)^0.14 ≈ 0.767
        avg_speed *= (1.5 / 10) ** 0.14

        return avg_speed, avg_dir

    except Exception:
        return None, None
