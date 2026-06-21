import numpy as np
import pandas as pd

G = 9.81  # m/s²


def air_density(altitude_m: np.ndarray | float) -> np.ndarray | float:
    """Barometric formula — ISA atmosphere. Accepts scalar or array."""
    return 1.225 * (1 - 2.2557e-5 * np.asarray(altitude_m, dtype=float)) ** 5.2559


def compute_heading(latlng: list) -> np.ndarray:
    """Bearing in degrees (0=N, 90=E) per GPS sample, smoothed.
    Scales dlng by cos(lat) so heading is correct regardless of latitude."""
    lat = np.array([p[0] for p in latlng])
    lng = np.array([p[1] for p in latlng])
    dlat = np.diff(lat, append=lat[-1])
    dlng = np.diff(lng, append=lng[-1])
    dlng_scaled = dlng * np.cos(np.radians(lat))
    heading = np.degrees(np.arctan2(dlng_scaled, dlat)) % 360
    return pd.Series(heading).rolling(5, center=True, min_periods=1).mean().values


def apparent_air_speed(
    velocity: np.ndarray,
    latlng: list,
    wind_speed_ms: float,
    wind_from_deg: float,
) -> np.ndarray:
    """
    Apparent air speed at each sample.
    Positive = headwind (air hitting you from front).
    Negative = tailwind exceeding rider speed.
    wind_from_deg: meteorological convention (direction FROM which wind blows).
    """
    heading = compute_heading(latlng)
    wind_to_deg = (wind_from_deg + 180) % 360
    wind_tail = wind_speed_ms * np.cos(np.radians(wind_to_deg - heading))
    return velocity - wind_tail  # sign preserved — no clip


def estimate_power(
    time: np.ndarray,
    velocity: np.ndarray,   # ground speed m/s
    grade: np.ndarray,      # percent
    altitude: np.ndarray,   # metres
    mass: float,            # kg total
    CdA: float = 0.32,
    Crr: float = 0.011,
    drivetrain_eff: float = 0.97,
    v_air: np.ndarray | None = None,  # apparent air speed; None → no wind correction
) -> np.ndarray:
    """Return estimated mechanical power (W) at each sample. Clamped to [0, 2000] W.

    Physics note: aero force magnitude depends on v_air^2, but rider power is
    force × ground speed. These differ when wind ≠ 0; using ground speed here
    is correct (confirmed by wind-tunnel limit: v_ground=0 → P_rider=0).
    """
    rho   = air_density(altitude)  # per-sample, not averaged
    theta = np.arctan(np.clip(grade, -30, 30) / 100.0)
    v_smooth = pd.Series(velocity).rolling(window=10, center=True, min_periods=1).mean().values
    acc   = np.clip(np.gradient(v_smooth, time), -3.0, 3.0)  # clamp GPS noise spikes

    v_aero = v_air if v_air is not None else velocity
    # Signed force: negative v_aero (strong tailwind) → negative (assisting) F_aero
    F_aero = 0.5 * rho * CdA * v_aero * np.abs(v_aero)
    F_roll = Crr * mass * G * np.cos(theta)
    F_grav = mass * G * np.sin(theta)
    F_acc  = mass * acc

    P = (F_aero + F_roll + F_grav + F_acc) * velocity / drivetrain_eff
    return np.clip(P, 0.0, 2000.0)


def normalized_power(power: np.ndarray, window: int = 30) -> float:
    """Standard 30-second rolling average to the 4th power, then 4th root."""
    if len(power) < window:
        return float(np.mean(power))
    rolling = np.convolve(power, np.ones(window) / window, mode="valid")
    return float((np.mean(rolling ** 4)) ** 0.25)


def energy_to_kcal(
    power: np.ndarray,
    time: np.ndarray,
    mech_efficiency: float = 0.23,
) -> float:
    """Mechanical power series → dietary kilocalories."""
    energy_J = float(np.trapezoid(power, time))
    return energy_J / mech_efficiency / 4184.0
