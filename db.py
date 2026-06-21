import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).parent / "results.db"


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS results (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                activity_id     INTEGER UNIQUE,
                activity_name   TEXT,
                activity_date   TEXT,
                recorded_at     TEXT,
                bike_profile    TEXT,
                rider_kg        REAL,
                bike_kg         REAL,
                bags_kg         REAL,
                total_kg        REAL,
                CdA             REAL,
                Crr             REAL,
                mech_eff        REAL,
                drive_eff       REAL,
                avg_power_w     REAL,
                norm_power_w    REAL,
                energy_kj       REAL,
                calories_kcal   REAL,
                avg_speed_kmh   REAL,
                duration_s      INTEGER,
                distance_m      REAL,
                wind_speed_ms   REAL,
                wind_from_deg   REAL,
                avg_headwind_ms REAL
            )
        """)
        # Non-destructive migration for existing databases
        for col_def in (
            "wind_speed_ms REAL", "wind_from_deg REAL", "avg_headwind_ms REAL",
            "bike_profile TEXT",
        ):
            try:
                c.execute(f"ALTER TABLE results ADD COLUMN {col_def}")
            except Exception:
                pass


def save_result(
    activity_id: int, activity_name: str, activity_date: str,
    bike_profile: str | None = None,
    rider_kg: float = 0.0, bike_kg: float = 0.0, bags_kg: float = 0.0, total_kg: float = 0.0,
    CdA: float = 0.0, Crr: float = 0.0, mech_eff: float = 0.0, drive_eff: float = 0.0,
    avg_power_w: float = 0.0, norm_power_w: float = 0.0, energy_kj: float = 0.0,
    calories_kcal: float = 0.0, avg_speed_kmh: float = 0.0,
    duration_s: int = 0, distance_m: float = 0.0,
    wind_speed_ms: float | None = None,
    wind_from_deg: float | None = None,
    avg_headwind_ms: float | None = None,
) -> None:
    with _conn() as c:
        c.execute("""
            INSERT OR REPLACE INTO results
                (activity_id, activity_name, activity_date, recorded_at,
                 bike_profile,
                 rider_kg, bike_kg, bags_kg, total_kg,
                 CdA, Crr, mech_eff, drive_eff,
                 avg_power_w, norm_power_w, energy_kj, calories_kcal,
                 avg_speed_kmh, duration_s, distance_m,
                 wind_speed_ms, wind_from_deg, avg_headwind_ms)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            activity_id, activity_name, activity_date, datetime.now().isoformat(),
            bike_profile,
            rider_kg, bike_kg, bags_kg, total_kg,
            CdA, Crr, mech_eff, drive_eff,
            avg_power_w, norm_power_w, energy_kj, calories_kcal,
            avg_speed_kmh, duration_s, distance_m,
            wind_speed_ms, wind_from_deg, avg_headwind_ms,
        ))


def load_history() -> pd.DataFrame:
    with _conn() as c:
        rows = c.execute("SELECT * FROM results ORDER BY activity_date DESC").fetchall()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([dict(r) for r in rows])


def delete_result(activity_id: int) -> None:
    with _conn() as c:
        c.execute("DELETE FROM results WHERE activity_id = ?", (activity_id,))


def clear_history() -> None:
    with _conn() as c:
        c.execute("DELETE FROM results")
