import os
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Allow running from any working directory
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from physics import apparent_air_speed, energy_to_kcal, estimate_power, normalized_power
from strava_client import CYCLING_SPORT_TYPES, StravaClient
from weather import deg_to_compass, get_wind

load_dotenv(override=True)

# LOCAL_MODE=true in .env → SQLite persistence; unset on Streamlit Cloud → session-only
_LOCAL = os.getenv("LOCAL_MODE", "").lower() in ("1", "true", "yes")
if _LOCAL:
    from db import delete_result, init_db, load_history, save_result
    init_db()


def _secret(key: str) -> str:
    """Read from st.secrets (Streamlit Cloud) then fall back to env / .env."""
    try:
        return st.secrets.get(key, "") or os.getenv(key, "")
    except Exception:
        return os.getenv(key, "")

# Bike type → frame weight + base CdA (riding position implied by bike + hand pos)
BIKE_TYPES: dict[str, dict] = {
    "Road – hoods":          dict(bike_kg=8.0,  CdA=0.32),
    "Road – drops":          dict(bike_kg=8.0,  CdA=0.27),
    "Road – TT / aero bars": dict(bike_kg=8.5,  CdA=0.22),
    "Gravel – hoods":        dict(bike_kg=11.5, CdA=0.33),
    "Gravel – drops":        dict(bike_kg=11.5, CdA=0.28),
    "MTB / VTT":             dict(bike_kg=13.0, CdA=0.45),
    "Hybrid / Trekking":     dict(bike_kg=14.0, CdA=0.50),
    "City / Dutch":          dict(bike_kg=16.0, CdA=0.55),
    "Custom":                dict(bike_kg=12.0, CdA=0.32),
}

# Tire type → Crr + small CdA penalty for width (vs 25 mm baseline).
# Crr values from bicyclerollingresistance.com at ~30 km/h on smooth tarmac;
# real off-road Crr is higher — these are conservative averages.
TIRE_TYPES: dict[str, dict] = {
    "Road slick 25 mm – tubeless":      dict(Crr=0.0040, cda_delta=0.000),
    "Road slick 25 mm – tube":          dict(Crr=0.0050, cda_delta=0.000),
    "Road slick 28 mm – tubeless":      dict(Crr=0.0045, cda_delta=0.005),
    "Road slick 28 mm – tube":          dict(Crr=0.0055, cda_delta=0.005),
    "Gravel slick 35 mm – tubeless":    dict(Crr=0.0055, cda_delta=0.010),
    "Gravel slick 38 mm – tubeless":    dict(Crr=0.0060, cda_delta=0.012),
    "Gravel slick 38 mm – tube":        dict(Crr=0.0080, cda_delta=0.012),
    "Gravel knobby 38 mm – tube":       dict(Crr=0.0100, cda_delta=0.012),
    "Gravel semi-slick 40 mm – tubeless": dict(Crr=0.0075, cda_delta=0.014),
    "Gravel semi-slick 40 mm – tube":   dict(Crr=0.0095, cda_delta=0.014),
    "Gravel knobby 40 mm – tube":       dict(Crr=0.0110, cda_delta=0.014),
    "Gravel knobby 45 mm – tube":       dict(Crr=0.0130, cda_delta=0.018),
    "MTB XC 2.2\" – tubeless":          dict(Crr=0.0140, cda_delta=0.025),
    "MTB trail 2.4\" – tubeless":       dict(Crr=0.0180, cda_delta=0.030),
    "MTB enduro 2.5\"+ – tubeless":     dict(Crr=0.0220, cda_delta=0.035),
    "Trekking / city 35–40 mm":         dict(Crr=0.0080, cda_delta=0.014),
    "Custom":                           dict(Crr=0.0070, cda_delta=0.012),
}

def _detect_redirect_uri() -> str:
    """Auto-detect the correct redirect URI from the current request host.
    Explicit secret always wins; otherwise derive from Host header."""
    explicit = _secret("REDIRECT_URI")
    if explicit:
        return explicit
    try:
        host = st.context.headers.get("host", "")
        if host and not host.startswith("localhost") and not host.startswith("127."):
            return f"https://{host}/"
    except Exception:
        pass
    return "http://localhost:8501/"

REDIRECT_URI = _detect_redirect_uri()

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Strava Power Estimator",
    layout="wide",
    page_icon="🚴",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Configuration")

    _default_id     = _secret("STRAVA_CLIENT_ID")
    _default_secret = _secret("STRAVA_CLIENT_SECRET")
    have_env_creds  = bool(_default_id)
    with st.expander("Strava API credentials", expanded=not have_env_creds):
        client_id = st.text_input(
            "Client ID",
            value=_default_id,
            help="From https://www.strava.com/settings/api",
        )
        client_secret = st.text_input(
            "Client Secret",
            value=_default_secret,
            type="password",
        )

    st.subheader("Rider & Bike")
    rider_weight = st.number_input("Rider weight (kg)", value=72.0, step=0.5,
                                    min_value=30.0, max_value=200.0)

    bike_type_name = st.selectbox(
        "Bike type",
        list(BIKE_TYPES.keys()),
        index=list(BIKE_TYPES.keys()).index("Gravel – hoods"),
        help="Frame, geometry & riding position → bike weight + base CdA",
    )
    bike = BIKE_TYPES[bike_type_name]

    tire_type_name = st.selectbox(
        "Tires",
        list(TIRE_TYPES.keys()),
        index=list(TIRE_TYPES.keys()).index("Gravel knobby 38 mm – tube"),
        help="Tire width & casing → rolling resistance (Crr) + small aero penalty for width",
    )
    tire = TIRE_TYPES[tire_type_name]

    bike_profile_name = f"{bike_type_name} + {tire_type_name}"

    bike_weight = st.number_input(
        "Bike weight (kg)", value=float(bike["bike_kg"]), step=0.1,
        min_value=3.0, max_value=40.0,
    )
    bags_weight = st.number_input(
        "Bags & luggage (kg)", value=0.0, step=0.1, min_value=0.0, max_value=40.0,
        help="Sum of all bags, panniers, and cargo — set this per ride",
    )
    total_mass = rider_weight + bike_weight + bags_weight
    st.caption(f"Total system mass: **{total_mass:.1f} kg**")

    _cda_default = float(bike["CdA"]) + float(tire["cda_delta"])
    with st.expander("Advanced physics"):
        st.caption(
            f"Base CdA from bike: **{bike['CdA']:.3f}** + tire width penalty "
            f"**{tire['cda_delta']:+.3f}** = **{_cda_default:.3f}**"
        )
        CdA = st.number_input(
            "CdA (m²)", value=_cda_default, step=0.01, format="%.3f",
            help=(
                "Aerodynamic drag coefficient × frontal area.\n"
                "Hoods ≈ 0.32 | Drops ≈ 0.27 | TT ≈ 0.22 | Upright MTB ≈ 0.45\n"
                "Wider tires add ~0.01–0.03. Large bar bag adds ~0.01–0.03 more."
            ),
        )
        Crr = st.number_input(
            "Crr (rolling resistance)", value=float(tire["Crr"]), step=0.0005, format="%.4f",
            help=(
                "From bike tires database, smooth tarmac at ~30 km/h.\n"
                "Real Crr on rough/gravel can be 1.5–2× higher.\n"
                "Road tubeless 25 mm ≈ 0.004 | Gravel 40 mm knobby ≈ 0.011 | MTB XC ≈ 0.014"
            ),
        )
        mech_eff = st.number_input(
            "Metabolic efficiency", value=0.23, step=0.01, format="%.2f",
            help="Fraction of food energy converted to mechanical work (~0.23 for cycling)",
        )
        drive_eff = st.number_input(
            "Drivetrain efficiency", value=0.97, step=0.01, format="%.2f",
            help="Clean chain ≈ 0.97",
        )

    if "token" in st.session_state:
        st.divider()
        if st.button("Disconnect Strava"):
            for k in ["token", "athlete", "streams", "cached_id"]:
                st.session_state.pop(k, None)
            st.rerun()

    st.divider()
    if _LOCAL:
        st.caption("🖥️ **Local mode** — history saved to SQLite database")
    else:
        st.caption("☁️ **Cloud mode** — history resets when tab closes")

# ── OAuth code exchange ───────────────────────────────────────────────────────

params = st.query_params
if "code" in params and "token" not in st.session_state:
    code = params["code"]
    st.query_params.clear()  # clear URL immediately — Strava codes are single-use

    if not client_id or not client_secret:
        st.warning("Enter your Strava API credentials in the sidebar, then reconnect.")
        st.stop()

    try:
        data = StravaClient(client_id, client_secret).exchange_code(code, REDIRECT_URI)
    except Exception as exc:
        st.error(f"Network error during OAuth: {exc}")
        st.stop()

    if data and "access_token" in data:
        st.session_state["token"] = data["access_token"]
        st.session_state["athlete"] = data.get("athlete", {})
        st.rerun()
    else:
        st.error(f"Strava token exchange failed. Response: `{data}`")
        st.stop()

# ── Main ──────────────────────────────────────────────────────────────────────

st.title("🚴 Strava Power Estimator")
st.caption("Physics-based power & calorie estimation — no power meter needed")

# ── Not authenticated: show connect screen ────────────────────────────────────

if "token" not in st.session_state:
    st.markdown("---")
    st.subheader("Getting started")
    from urllib.parse import urlparse
    _callback_host = urlparse(REDIRECT_URI).hostname or "localhost"
    st.markdown(
        f"""
1. Go to **[strava.com/settings/api](https://www.strava.com/settings/api)** and create (or edit) an app
   — set *Authorization Callback Domain* to **`{_callback_host}`** (just the hostname, no `https://`, no trailing `/`)
2. Paste your **Client ID** and **Client Secret** in the sidebar (or set them in Streamlit secrets)
3. Click Connect below
        """
    )
    st.caption(f":grey[Redirect URI being sent: `{REDIRECT_URI}` — callback domain on Strava must be `{_callback_host}`]")

    if client_id and client_secret:
        auth_url = "https://www.strava.com/oauth/authorize?" + urlencode(
            {
                "client_id": client_id,
                "redirect_uri": REDIRECT_URI,
                "response_type": "code",
                "approval_prompt": "auto",
                "scope": "activity:read_all",
            }
        )
        st.markdown(
            f'<a href="{auth_url}" target="_top">'
            '<button style="background:#FC4C02;color:#fff;border:none;padding:12px 28px;'
            "border-radius:6px;font-size:16px;cursor:pointer;font-weight:700;"
            'margin-top:8px">Connect with Strava</button></a>',
            unsafe_allow_html=True,
        )
    else:
        st.warning("Enter your Strava API credentials in the sidebar to continue.")
    st.stop()

# ── Authenticated: activity list ──────────────────────────────────────────────

athlete = st.session_state.get("athlete", {})
if athlete:
    name = f"{athlete.get('firstname', '')} {athlete.get('lastname', '')}".strip()
    st.caption(f"Logged in as **{name}**")


@st.cache_data(ttl=300, show_spinner="Loading activities…")
def fetch_activities(token: str) -> list:
    return StravaClient("", "", token).get_activities(40)


all_activities = fetch_activities(st.session_state["token"])
cycling = [
    a for a in all_activities
    if a.get("sport_type") in CYCLING_SPORT_TYPES
    or a.get("type") in CYCLING_SPORT_TYPES
]

if not cycling:
    st.warning("No cycling activities found in your last 40 activities.")
    st.stop()


def _fmt_duration(seconds: int) -> str:
    h, m = divmod(seconds // 60, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m"


activity_options = {
    (
        f"{a['name']}  ·  {a['start_date_local'][:10]}"
        f"  ·  {a['distance'] / 1000:.1f} km"
        f"  ·  {_fmt_duration(a['moving_time'])}"
    ): a["id"]
    for a in cycling
}

col_sel, col_btn = st.columns([5, 1])
with col_sel:
    chosen_label = st.selectbox("Activity", list(activity_options.keys()),
                                 label_visibility="collapsed")
with col_btn:
    analyse_clicked = st.button("Analyse", type="primary", use_container_width=True)

chosen_id = activity_options[chosen_label]
chosen_activity = next(a for a in cycling if a["id"] == chosen_id)

# ── Fetch streams (cached per activity) ───────────────────────────────────────

need_fetch = analyse_clicked or (
    "streams" in st.session_state and st.session_state.get("cached_id") != chosen_id
)
have_data = "streams" in st.session_state and st.session_state.get("cached_id") == chosen_id

if analyse_clicked and not have_data:
    with st.spinner("Fetching activity data from Strava…"):
        try:
            raw = StravaClient("", "", st.session_state["token"]).get_streams(chosen_id)
            st.session_state["streams"] = raw
            st.session_state["cached_id"] = chosen_id
            have_data = True
        except Exception as exc:
            st.error(f"Stream fetch failed: {exc}")
            st.stop()

if not have_data:
    st.stop()

# ── Compute power ─────────────────────────────────────────────────────────────

streams = st.session_state["streams"]
REQUIRED = {"time", "velocity_smooth", "grade_smooth", "altitude"}
missing = REQUIRED - set(streams)
if missing:
    st.error(
        f"Missing data streams: {missing}. "
        "Manual activities or activities without GPS won't have this data."
    )
    st.stop()

t      = np.array(streams["time"]["data"],             dtype=float)
v      = np.array(streams["velocity_smooth"]["data"],  dtype=float)
g      = np.array(streams["grade_smooth"]["data"],     dtype=float)
alt    = np.array(streams["altitude"]["data"],         dtype=float)
latlng = streams.get("latlng", {}).get("data")
# Use Strava's moving flag when available; fall back to speed threshold
_moving_data = streams.get("moving", {}).get("data")
_strava_moving = np.array(_moving_data, dtype=bool) if _moving_data else None

# ── Wind fetch ────────────────────────────────────────────────────────────────
wind_speed_ms, wind_from_deg, avg_headwind_ms = None, None, None
v_air = None

start_lat = chosen_activity.get("start_latlng", [None, None])[0]
start_lng = chosen_activity.get("start_latlng", [None, None])[1]
start_utc = chosen_activity.get("start_date", "")  # "YYYY-MM-DDTHH:MM:SSZ"

if start_lat and start_lng and start_utc and latlng:
    with st.spinner("Fetching wind data…"):
        date_utc = start_utc[:10]
        hour_utc = int(start_utc[11:13])
        duration_h = chosen_activity["moving_time"] / 3600
        wind_speed_ms, wind_from_deg = get_wind(start_lat, start_lng, date_utc, hour_utc, duration_h)

    if wind_speed_ms is not None:
        v_air = apparent_air_speed(v, latlng, wind_speed_ms, wind_from_deg)
        wind_tail = v - v_air  # positive = tailwind, negative = headwind
        avg_headwind_ms = float(-np.mean(wind_tail))  # positive = net headwind

power = estimate_power(t, v, g, alt, total_mass, CdA, Crr, drive_eff, v_air=v_air)
dt    = np.diff(t, prepend=t[0])
kcal  = energy_to_kcal(power, t, mech_eff)
kJ    = float(np.sum(power * dt)) / 1000.0

# Use Strava's moving flag if present, otherwise fall back to speed threshold
if _strava_moving is not None and len(_strava_moving) == len(v):
    moving_mask = _strava_moving
    moving_source = "Strava"
else:
    moving_mask = v > 0.5
    moving_source = "speed > 0.5 m/s"
stopped_s = int(np.sum(~moving_mask))
power_moving = power[moving_mask]
avg_w = float(np.mean(power_moving)) if len(power_moving) > 0 else 0.0
NP    = normalized_power(power_moving) if len(power_moving) >= 30 else avg_w
avg_v = float(np.mean(v[moving_mask])) * 3.6 if moving_mask.any() else float(np.mean(v)) * 3.6

# ── Metrics ───────────────────────────────────────────────────────────────────

st.divider()
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Avg Power",        f"{avg_w:.0f} W")
m2.metric("Normalized Power", f"{NP:.0f} W")
m3.metric("Mechanical Work",  f"{kJ:.1f} kJ")
m4.metric("Calories Burned",  f"{kcal:.0f} kcal")
m5.metric("Avg Speed",        f"{avg_v:.1f} km/h")

stopped_min = stopped_s // 60
st.caption(
    f":grey[Power & speed on moving samples only · "
    f"**{stopped_min} min stopped** excluded ({stopped_s} s) · "
    f"moving detection: {moving_source}]"
)
if wind_speed_ms is not None:
    compass = deg_to_compass(wind_from_deg)
    head_label = f"+{avg_headwind_ms:.1f} m/s" if avg_headwind_ms > 0 else f"{avg_headwind_ms:.1f} m/s"
    st.caption(
        f":grey[Wind: **{wind_speed_ms:.1f} m/s from {compass}** ({wind_from_deg:.0f}°) · "
        f"avg headwind component: **{head_label}** · power estimate includes wind.]"
    )
else:
    st.caption(":grey[Wind data unavailable (activity too recent or no GPS). Power estimated without wind.]")

# ── Save to history ───────────────────────────────────────────────────────────

if not _LOCAL and "history" not in st.session_state:
    st.session_state["history"] = []

if st.button("💾 Save to history", help="Record this result with the current setup parameters"):
    _record = dict(
        activity_id=chosen_id,
        activity_name=chosen_activity["name"],
        activity_date=chosen_activity["start_date_local"][:10],
        bike_profile=bike_profile_name,
        rider_kg=rider_weight, bike_kg=bike_weight, bags_kg=bags_weight,
        total_kg=total_mass,
        CdA=CdA, Crr=Crr, mech_eff=mech_eff, drive_eff=drive_eff,
        avg_power_w=avg_w, norm_power_w=NP, energy_kj=kJ,
        calories_kcal=kcal, avg_speed_kmh=avg_v,
        duration_s=chosen_activity["moving_time"],
        distance_m=chosen_activity["distance"],
        wind_speed_ms=wind_speed_ms,
        wind_from_deg=wind_from_deg,
        avg_headwind_ms=avg_headwind_ms,
    )
    if _LOCAL:
        save_result(**_record)
    else:
        st.session_state["history"].append({**_record, "recorded_at": datetime.now().isoformat()})
    st.success("Saved!")

# ── Chart 1: Power + Elevation ────────────────────────────────────────────────

t_min    = t / 60.0
smooth_p = pd.Series(power).rolling(30, center=True, min_periods=1).mean().values

fig1 = make_subplots(specs=[[{"secondary_y": True}]])
fig1.add_trace(
    go.Scatter(x=t_min, y=power, name="Power (raw)",
               line=dict(color="rgba(252,76,2,0.2)", width=1)),
    secondary_y=False,
)
fig1.add_trace(
    go.Scatter(x=t_min, y=smooth_p, name="Power (30 s avg)",
               line=dict(color="#FC4C02", width=2)),
    secondary_y=False,
)
fig1.add_trace(
    go.Scatter(
        x=t_min, y=alt, name="Elevation (m)",
        fill="tozeroy", fillcolor="rgba(100,200,100,0.12)",
        line=dict(color="rgba(80,160,80,0.55)", width=1),
    ),
    secondary_y=True,
)
fig1.update_layout(
    title="Power & Elevation", height=400, hovermode="x unified",
    legend=dict(orientation="h", y=1.08, x=1, xanchor="right"),
)
fig1.update_xaxes(title_text="Time (min)")
fig1.update_yaxes(title_text="Power (W)",    secondary_y=False)
fig1.update_yaxes(title_text="Elevation (m)", secondary_y=True, showgrid=False)
st.plotly_chart(fig1, use_container_width=True)

# ── Chart 2: Speed + Grade ────────────────────────────────────────────────────

fig2 = make_subplots(specs=[[{"secondary_y": True}]])
fig2.add_trace(
    go.Scatter(x=t_min, y=v * 3.6, name="Speed (km/h)",
               line=dict(color="#1f77b4", width=2)),
    secondary_y=False,
)
fig2.add_trace(
    go.Scatter(
        x=t_min, y=g, name="Grade (%)",
        fill="tozeroy", fillcolor="rgba(200,80,80,0.1)",
        line=dict(color="rgba(200,80,80,0.65)", width=1.5),
    ),
    secondary_y=True,
)
fig2.update_layout(
    title="Speed & Grade", height=350, hovermode="x unified",
    legend=dict(orientation="h", y=1.08, x=1, xanchor="right"),
)
fig2.update_xaxes(title_text="Time (min)")
fig2.update_yaxes(title_text="Speed (km/h)", secondary_y=False)
fig2.update_yaxes(title_text="Grade (%)",    secondary_y=True, showgrid=False)
st.plotly_chart(fig2, use_container_width=True)

# ── History tab ───────────────────────────────────────────────────────────────

st.divider()
st.subheader("History")

if _LOCAL:
    hist = load_history()
else:
    _saved = st.session_state.get("history", [])
    hist = pd.DataFrame(_saved) if _saved else pd.DataFrame()
if hist.empty:
    st.info("No saved activities yet. Analyse a ride and click 'Save to history'.")
else:
    st.download_button(
        "⬇ Export CSV",
        data=hist.to_csv(index=False).encode(),
        file_name="strava_power_history.csv",
        mime="text/csv",
    )
    # ── Trend charts ──────────────────────────────────────────────────────────
    hc1, hc2 = st.columns(2)

    with hc1:
        fig_pwr = go.Figure()
        fig_pwr.add_trace(go.Scatter(
            x=hist["activity_date"], y=hist["avg_power_w"],
            mode="lines+markers", name="Avg Power",
            line=dict(color="#FC4C02"),
        ))
        fig_pwr.add_trace(go.Scatter(
            x=hist["activity_date"], y=hist["norm_power_w"],
            mode="lines+markers", name="Normalized Power",
            line=dict(color="#FC4C02", dash="dash"),
        ))
        fig_pwr.update_layout(title="Power over time", height=280,
                               yaxis_title="W", hovermode="x unified",
                               legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig_pwr, use_container_width=True)

    with hc2:
        fig_kcal = go.Figure()
        fig_kcal.add_trace(go.Bar(
            x=hist["activity_date"], y=hist["calories_kcal"],
            marker_color="#1f77b4", name="Calories",
        ))
        fig_kcal.update_layout(title="Calories per activity", height=280,
                                yaxis_title="kcal")
        st.plotly_chart(fig_kcal, use_container_width=True)

    # ── Table ─────────────────────────────────────────────────────────────────
    display_cols = {
        "activity_date":    "Date",
        "activity_name":    "Activity",
        "bike_profile":     "Bike",
        "bags_kg":          "Bags (kg)",
        "total_kg":         "Mass (kg)",
        "avg_power_w":      "Avg W",
        "norm_power_w":     "NP (W)",
        "energy_kj":        "Work (kJ)",
        "calories_kcal":    "kcal",
        "avg_speed_kmh":    "Speed (km/h)",
        "wind_speed_ms":    "Wind (m/s)",
        "avg_headwind_ms":  "Headwind (m/s)",
    }
    # fill columns absent in older DB rows
    for col in display_cols:
        if col not in hist.columns:
            hist[col] = None
    tbl = hist[list(display_cols)].rename(columns=display_cols)
    tbl["Avg W"]      = tbl["Avg W"].round(0).astype(int)
    tbl["NP (W)"]     = tbl["NP (W)"].round(0).astype(int)
    tbl["Work (kJ)"]  = tbl["Work (kJ)"].round(1)
    tbl["kcal"]       = tbl["kcal"].round(0).astype(int)
    tbl["Speed (km/h)"]    = tbl["Speed (km/h)"].round(1)
    tbl["Wind (m/s)"]      = tbl["Wind (m/s)"].round(1)
    tbl["Headwind (m/s)"]  = tbl["Headwind (m/s)"].round(2)
    st.dataframe(tbl, use_container_width=True, hide_index=True)

    # ── Statistics & comparison ───────────────────────────────────────────────
    st.divider()
    st.subheader("Statistics & comparison")

    h = hist.copy()

    # ── Records row ───────────────────────────────────────────────────────────
    r1, r2, r3, r4, r5 = st.columns(5)
    r1.metric("Rides saved",    len(h))
    r2.metric("Best NP",        f"{h['norm_power_w'].max():.0f} W")
    r3.metric("Best avg power", f"{h['avg_power_w'].max():.0f} W")
    r4.metric("Best avg speed", f"{h['avg_speed_kmh'].max():.1f} km/h")
    total_km = h["distance_m"].sum() / 1000 if "distance_m" in h.columns else 0
    r5.metric("Total distance", f"{total_km:.0f} km")

    st.divider()

    # ── Scatter: avg speed vs NP, coloured by bike profile ───────────────────
    profiles = h["bike_profile"].fillna("(no profile)").unique().tolist()
    colors   = ["#FC4C02", "#1f77b4", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"]
    color_map = {p: colors[i % len(colors)] for i, p in enumerate(profiles)}

    fig_sc = go.Figure()
    for profile in profiles:
        mask = h["bike_profile"].fillna("(no profile)") == profile
        subset = h[mask]
        fig_sc.add_trace(go.Scatter(
            x=subset["avg_speed_kmh"],
            y=subset["norm_power_w"],
            mode="markers",
            name=profile,
            marker=dict(size=10, color=color_map[profile], opacity=0.85),
            customdata=subset[["activity_name", "activity_date", "avg_power_w",
                                "calories_kcal", "bags_kg"]].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b> · %{customdata[1]}<br>"
                "Speed: %{x:.1f} km/h · NP: %{y:.0f} W · Avg: %{customdata[2]:.0f} W<br>"
                "kcal: %{customdata[3]:.0f} · bags: %{customdata[4]:.1f} kg"
                "<extra></extra>"
            ),
        ))
    fig_sc.update_layout(
        title="Effort map — avg speed vs Normalized Power",
        xaxis_title="Avg speed (km/h, moving)",
        yaxis_title="Normalized Power (W)",
        height=380, hovermode="closest",
        legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(fig_sc, use_container_width=True)

    # ── NP distribution by bike profile (box plot) ────────────────────────────
    if len(profiles) > 1 or len(h) >= 3:
        fig_box = go.Figure()
        for profile in profiles:
            mask   = h["bike_profile"].fillna("(no profile)") == profile
            subset = h[mask]
            fig_box.add_trace(go.Box(
                y=subset["norm_power_w"],
                name=profile,
                marker_color=color_map[profile],
                boxpoints="all",
                jitter=0.3,
                pointpos=-1.5,
            ))
        fig_box.update_layout(
            title="NP distribution by bike profile",
            yaxis_title="Normalized Power (W)",
            height=350,
            showlegend=False,
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # ── Efficiency table: NP per kg ───────────────────────────────────────────
    if "total_kg" in h.columns:
        h["W/kg (NP)"] = (h["norm_power_w"] / h["rider_kg"]).round(2)
        eff_cols = {
            "activity_date": "Date", "activity_name": "Activity",
            "bike_profile": "Bike", "total_kg": "Total kg",
            "norm_power_w": "NP (W)", "W/kg (NP)": "W/kg",
            "avg_speed_kmh": "Speed (km/h)", "calories_kcal": "kcal",
        }
        for col in eff_cols:
            if col not in h.columns:
                h[col] = None
        eff_tbl = h[list(eff_cols)].rename(columns=eff_cols).sort_values("W/kg", ascending=False)
        eff_tbl["NP (W)"]      = eff_tbl["NP (W)"].round(0).astype("Int64")
        eff_tbl["kcal"]        = eff_tbl["kcal"].round(0).astype("Int64")
        eff_tbl["Speed (km/h)"] = eff_tbl["Speed (km/h)"].round(1)
        st.caption("Ranked by W/kg (NP ÷ rider weight)")
        st.dataframe(eff_tbl, use_container_width=True, hide_index=True)

    # ── Delete entry (local mode only) ────────────────────────────────────────
    if _LOCAL:
        with st.expander("🗑️ Delete a saved entry"):
            del_labels = (hist["activity_name"] + " · " + hist["activity_date"]).tolist()
            del_choice = st.selectbox("Select entry", del_labels, key="del_select")
            if st.button("Delete entry", type="secondary"):
                idx = del_labels.index(del_choice)
                delete_result(int(hist.iloc[idx]["activity_id"]))
                st.rerun()
