# Strava Power Estimator

A physics-based cycling power and calorie estimator that connects to your Strava account. No power meter needed — wattage is derived from speed, gradient, rider/bike mass, aerodynamics, rolling resistance, and optional wind data.

**[Live demo →](https://strava-power.streamlit.app)**

---

## Features

- **Strava OAuth** — connect with one click, pick any recent cycling activity
- **Physics model** — estimates power sample-by-sample from aero drag, rolling resistance, gravity, and acceleration forces
- **Wind correction** — fetches historical wind from [Open-Meteo ERA5](https://open-meteo.com/) (free, no API key) and subtracts the tailwind component from aerodynamic drag
- **Bike + tire selector** — choose frame type (road/gravel/MTB/…) and tire compound separately; CdA and Crr are auto-filled from a database of measured values
- **Normalized Power** — computed on moving samples only (uses Strava's `moving` stream flag)
- **History** — save results across rides and compare with trend charts, scatter plots, box plots, and a W/kg ranking table
- **CSV export** — download your full history as a spreadsheet
- **Dual mode** — runs locally with SQLite persistence (`LOCAL_MODE=true`) or on Streamlit Cloud with session-state history

---

## Physics model

```
P = (F_aero + F_roll + F_grav + F_acc) × v_ground / η_drivetrain
```

| Force | Formula | Notes |
|---|---|---|
| Aerodynamic | `½ ρ CdA v_air |v_air|` | `v_air = v_ground − v_wind_tail`; signed so tailwind reduces drag |
| Rolling | `Crr × m × g × cos(θ)` | Crr from tyre database |
| Gravity | `m × g × sin(θ)` | per-sample grade from Strava stream |
| Acceleration | `m × a` | 10-sample smoothed velocity gradient, clamped ±3 m/s² |

Air density `ρ` is computed per sample via the ISA barometric formula. Heading is derived from GPS with a `cos(lat)` correction on longitude deltas. Calories = mechanical work / metabolic efficiency (default 23 %).

---

## Quick start — local

```bash
git clone https://github.com/cgloceri-sketch/strava-power
cd strava-power
pip install -r requirements.txt
```

Create a `.env` file:

```env
STRAVA_CLIENT_ID=your_client_id
STRAVA_CLIENT_SECRET=your_client_secret
LOCAL_MODE=true
```

```bash
streamlit run app.py
```

History is saved to `results.db` (SQLite) in the project folder.

---

## Deploy on Streamlit Cloud

1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → pick your fork, branch `main`, file `app.py`
3. Under **Settings → Secrets**, add:

```toml
STRAVA_CLIENT_ID = "your_client_id"
STRAVA_CLIENT_SECRET = "your_client_secret"
```

`REDIRECT_URI` is auto-detected from the app's hostname — no manual setting needed.

---

## Strava API setup

1. Go to [strava.com/settings/api](https://www.strava.com/settings/api) and create an app
2. Set **Authorization Callback Domain**:
   - Local: `localhost`
   - Streamlit Cloud: `your-app-name.streamlit.app`
3. Copy **Client ID** and **Client Secret** into your `.env` or Streamlit secrets

---

## Configuration

| Variable | Where | Description |
|---|---|---|
| `STRAVA_CLIENT_ID` | `.env` / Streamlit secrets | From strava.com/settings/api |
| `STRAVA_CLIENT_SECRET` | `.env` / Streamlit secrets | From strava.com/settings/api |
| `LOCAL_MODE` | `.env` only | Set to `true` to enable SQLite history |

---

## Project structure

```
app.py            — Streamlit UI: OAuth, activity selector, physics compute, charts, history
physics.py        — Power model: forces, air density, heading, NP, kcal
strava_client.py  — Strava API wrapper (activities, streams, token exchange)
weather.py        — Open-Meteo ERA5 wind fetch
db.py             — SQLite persistence (local mode only)
requirements.txt
.streamlit/
  secrets.toml.example
```

---

## Tech stack

- [Streamlit](https://streamlit.io) — UI and cloud hosting
- [Plotly](https://plotly.com/python/) — interactive charts
- [Open-Meteo](https://open-meteo.com/) — free historical weather (ERA5 reanalysis)
- [Strava API v3](https://developers.strava.com/) — activity data and OAuth
- SQLite (local mode) / session state (cloud mode) — history persistence

---

## Limitations & future ideas

- Power is estimated, not measured — accuracy depends on correct CdA, Crr, and wind data
- Wind is a single vector for the whole ride (ERA5 hourly); gusty or variable wind conditions will affect accuracy
- Cloud mode history resets when the browser tab closes (no server-side storage)
- Potential additions: `.fit`/`.gpx` file upload (no Strava required), comparison of estimated vs actual power for users with a power meter
