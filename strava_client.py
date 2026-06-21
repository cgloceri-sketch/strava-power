import warnings

import requests
import urllib3
from typing import Optional

# Windows + Python 3.14 SSL CA bundle can fail against Strava's cert chain.
# For a local dev tool talking to a fixed, trusted endpoint this is acceptable.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
_SSL = False

CYCLING_SPORT_TYPES = {
    "Ride", "VirtualRide", "EBikeRide", "GravelRide",
    "MountainBikeRide", "Handcycle", "Velomobile",
}


class StravaClient:
    BASE = "https://www.strava.com/api/v3"
    TOKEN_URL = "https://www.strava.com/oauth/token"

    def __init__(self, client_id: str, client_secret: str, access_token: str = ""):
        self.client_id = str(client_id)
        self.client_secret = client_secret
        self.access_token = access_token

    def exchange_code(self, code: str, redirect_uri: str) -> Optional[dict]:
        r = requests.post(self.TOKEN_URL, data={
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": redirect_uri,
        }, timeout=10, verify=_SSL)
        try:
            return r.json()  # return body regardless of status so callers can inspect errors
        except Exception:
            return {"http_status": r.status_code, "text": r.text}

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.access_token}"}

    def get_activities(self, n: int = 40) -> list:
        r = requests.get(
            f"{self.BASE}/athlete/activities",
            headers=self._headers(),
            params={"per_page": n},
            timeout=10,
            verify=_SSL,
        )
        r.raise_for_status()
        return r.json()

    def get_streams(self, activity_id: int) -> dict:
        """Return streams keyed by type. Required keys: time, velocity_smooth,
        grade_smooth, altitude. Optional: distance."""
        r = requests.get(
            f"{self.BASE}/activities/{activity_id}/streams",
            headers=self._headers(),
            params={
                "keys": "time,velocity_smooth,grade_smooth,altitude,distance,latlng,moving",
                "key_by_type": "true",
            },
            timeout=15,
            verify=_SSL,
        )
        r.raise_for_status()
        return r.json()
