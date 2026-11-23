"""needle_sun.py - estimate where a fixed camera is from WHEN its photos are bright or dark.

How it works
------------
For every candidate point on Earth (lat, lon) and every photo time, the sun's elevation can be
computed. A camera that is dark at night and bright by day produces a brightness curve that only
fits a limited band of latitudes/longitudes:

  * the clock times of dawn/dusk (solar noon)  ->  longitude
  * how long the day lasts                     ->  latitude (and hemisphere, away from equinoxes)

This module turns that idea into a probability map. It does NOT contact any website and it does
NOT submit guesses. It only reads photos you saved yourself plus the release times you typed in.

Important limits (also shown to the user in the notebook)
---------------------------------------------------------
  * Release time is not capture time. A posting delay of 1 hour shifts the longitude by ~15 degrees.
    Latitude (day length) is much less affected by a constant delay.
  * Clouds, tree cover and camera auto-exposure blur the dawn/dusk moment. This is modelled with
    nuisance parameters, and the final probabilities are "tempered" so they are not over-confident.
  * The result is a broad band (typically tens to hundreds of thousands of km2), never a hexagon.
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0088
# The site says hexagons are "about 60 feet across". For a regular hexagon measured across the flats:
HEX_AREA_M2 = (math.sqrt(3.0) / 2.0) * (60 * 0.3048) ** 2

CITIES = {  # only used to orient the reader on the plot
    "New York": (40.71, -74.01), "Chicago": (41.88, -87.63), "Atlanta": (33.75, -84.39),
    "Toronto": (43.65, -79.38), "Montreal": (45.50, -73.57), "Halifax": (44.65, -63.57),
    "Thunder Bay": (48.38, -89.25), "Seattle": (47.61, -122.33), "Los Angeles": (34.05, -118.24),
    "Denver": (39.74, -104.99), "Dallas": (32.78, -96.80), "Miami": (25.76, -80.19),
    "Mexico City": (19.43, -99.13), "London": (51.51, -0.13), "Paris": (48.86, 2.35),
    "Berlin": (52.52, 13.40), "Rome": (41.90, 12.50), "Moscow": (55.76, 37.62),
    "Tokyo": (35.68, 139.69), "Beijing": (39.90, 116.41), "Delhi": (28.61, 77.21),
    "Sydney": (-33.87, 151.21), "Cape Town": (-33.92, 18.42), "Sao Paulo": (-23.55, -46.63),
}


# ----------------------------------------------------------------------------------------------
# Time handling
# ----------------------------------------------------------------------------------------------
def to_utc_timestamps(values, tz="UTC"):
    """Parse time strings into UTC pandas Timestamps.

    * text WITH an offset (e.g. 2026-11-22T14:05:00+01:00, or ending in Z) is converted to UTC;
    * text WITHOUT an offset is read as local time in `tz` (e.g. "Europe/Zurich").
    """
    out = []
    for v in values:
        ts = pd.Timestamp(v)
        if ts.tzinfo is None:
            try:
                ts = ts.tz_localize(tz)
            except Exception:  # ambiguous or non-existent local time around a clock change
                ts = ts.tz_localize(tz, ambiguous=True, nonexistent="shift_forward")
        out.append(ts.tz_convert("UTC"))
    return pd.Series(out, dtype="datetime64[ns, UTC]") if out else pd.Series([], dtype="datetime64[ns, UTC]")


def unix_seconds(utc_series):
    epoch = pd.Timestamp(0, tz="UTC")
    return np.array([(t - epoch).total_seconds() for t in utc_series], dtype=float)


# ----------------------------------------------------------------------------------------------
# Solar position (NOAA low-precision algorithm, good to ~0.01 degree for this purpose)
# ----------------------------------------------------------------------------------------------
def sun_params(unix_s):
    """Return (declination in radians, equation of time in minutes) for UTC unix seconds."""
    unix_s = np.asarray(unix_s, dtype=float)
    jd = unix_s / 86400.0 + 2440587.5
    jc = (jd - 2451545.0) / 36525.0
    l0 = np.mod(280.46646 + jc * (36000.76983 + jc * 0.0003032), 360.0)
    m = 357.52911 + jc * (35999.05029 - 0.0001537 * jc)
    e = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)
    mr = np.radians(m)
    c = (np.sin(mr) * (1.914602 - jc * (0.004817 + 0.000014 * jc))
         + np.sin(2 * mr) * (0.019993 - 0.000101 * jc)
         + np.sin(3 * mr) * 0.000289)
    true_long = l0 + c
    omega = np.radians(125.04 - 1934.136 * jc)
    app_long = np.radians(true_long - 0.00569 - 0.00478 * np.sin(omega))
    mean_obliq = 23.0 + (26.0 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60.0) / 60.0
    obliq = np.radians(mean_obliq + 0.00256 * np.cos(omega))
    decl = np.arcsin(np.sin(obliq) * np.sin(app_long))
    y = np.tan(obliq / 2.0) ** 2
    l0r = np.radians(l0)
    eot = 4.0 * np.degrees(y * np.sin(2 * l0r) - 2 * e * np.sin(mr)
                           + 4 * e * y * np.sin(mr) * np.cos(2 * l0r)
                           - 0.5 * y * y * np.sin(4 * l0r) - 1.25 * e * e * np.sin(2 * mr))
    return decl, eot


def solar_elevation_deg(unix_s, lat_deg, lon_deg):
    """Sun elevation above the horizon in degrees (no refraction). Arrays broadcast together."""
    unix_s = np.asarray(unix_s, dtype=float)
    decl, eot = sun_params(unix_s)
    minutes = np.mod(unix_s, 86400.0) / 60.0
    hour_angle = np.radians((minutes + eot) / 4.0 + np.asarray(lon_deg, dtype=float) - 180.0)
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    sin_e = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(hour_angle)
    return np.degrees(np.arcsin(np.clip(sin_e, -1.0, 1.0)))


# ----------------------------------------------------------------------------------------------
# Reading photos / tables
# ----------------------------------------------------------------------------------------------
def read_photo_table(csv_path, tz="UTC"):
    """CSV columns: photo, released_at, and optionally brightness (0 = night ... 1 = bright)."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "photo" not in df.columns or "released_at" not in df.columns:
        raise ValueError("The CSV needs the columns: photo, released_at (and optionally brightness).")
    df["utc"] = to_utc_timestamps(df["released_at"].astype(str), tz)
    df["photo"] = df["photo"].astype(int)
    return df.sort_values("utc").reset_index(drop=True)


def collect_images(folder):
    """Map photo number -> file path, using the last run of digits in each file name."""
    found = {}
    for root, _dirs, files in os.walk(folder):
        for name in files:
            if not name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                continue
            digits = re.findall(r"\d+", os.path.splitext(name)[0])
            if digits:
                found[int(digits[-1])] = os.path.join(root, name)
    return found


def image_mean_luminance(path, crop=(0.25, 0.30, 0.75, 0.90), size=192):
    """Average brightness (0-255) of the chosen part of the image (fractions left, top, right, bottom)."""
    from PIL import Image  # imported here so the solver works without Pillow when brightness is typed in
    im = Image.open(path).convert("RGB")
    w, h = im.size
    box = (int(crop[0] * w), int(crop[1] * h), max(int(crop[2] * w), int(crop[0] * w) + 1),
           max(int(crop[3] * h), int(crop[1] * h) + 1))
    im = im.crop(box).resize((size, size))
    a = np.asarray(im, dtype=float)
    lum = 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]
    return float(lum.mean())


def add_brightness(table, images_by_photo, crop=(0.25, 0.30, 0.75, 0.90)):
    """Add a `lum` column (mean luminance 0-255). Photos with no image file are dropped and counted."""
    table = table.copy()
    lums, missing = [], []
    for p in table["photo"]:
        path = images_by_photo.get(int(p))
        if path is None:
            lums.append(np.nan)
            missing.append(int(p))
        else:
            lums.append(image_mean_luminance(path, crop))
    table["lum"] = lums
    return table.dropna(subset=["lum"]).reset_index(drop=True), missing


def normalise_brightness(table, lo_q=2, hi_q=98):
    """Create column `s` in [0, 1] from `lum` (log scale, robust percentiles) or from typed `brightness`."""
    table = table.copy()
    notes = []
    if "brightness" in table.columns and table["brightness"].notna().all():
        table["s"] = table["brightness"].astype(float).clip(0, 1)
        notes.append("Used the brightness values you typed in.")
        return table, notes
    if "lum" not in table.columns:
        raise ValueError("No brightness available: provide photos or a `brightness` column.")
    lg = np.log1p(table["lum"].to_numpy(dtype=float))
    lo, hi = np.percentile(lg, lo_q), np.percentile(lg, hi_q)
    if hi - lo < 1.0:
        notes.append("WARNING: your photos barely change in brightness (they may not include both day and night). "
                     "The result will be unreliable.")
        hi = lo + 1.0
    table["s"] = np.clip((lg - lo) / (hi - lo), 0, 1)
    return table, notes


def check_coverage(table):
    """Plain-language warnings about data that cannot support the method."""
    w = []
    n = len(table)
    if n < 40:
        w.append(f"Only {n} photos. Aim for at least ~100 spanning several days.")
    span_days = (table["utc"].max() - table["utc"].min()).total_seconds() / 86400.0 if n > 1 else 0.0
    if span_days < 1.5:
        w.append(f"The photos span only {span_days:.1f} days. You need at least two sunrises and two sunsets.")
    if (table["s"] < 0.2).sum() < 5 or (table["s"] > 0.8).sum() < 5:
        w.append("Too few clearly dark or clearly bright photos. Include both night and day frames.")
    if table["utc"].duplicated().mean() > 0.3:
        w.append("Many photos share the same release time (batch posting). The timing information is weak.")
    return w


# ----------------------------------------------------------------------------------------------
# The probability model
# ----------------------------------------------------------------------------------------------
@dataclass
class Result:
    lat_grid: np.ndarray
    lon_grid: np.ndarray
    delays_min: np.ndarray
    logL: np.ndarray            # (delay, lat, lon) log-likelihood, nuisance parameters marginalised
    P: np.ndarray               # (lat, lon) tempered posterior, sums to 1
    temper: float
    n_photos: int
    summary: dict = field(default_factory=dict)
    per_delay_best: list = field(default_factory=list)


def make_grid(region, step):
    lat_min, lat_max, lon_min, lon_max = region
    lat_min, lat_max = max(lat_min, -85.0), min(lat_max, 85.0)
    lat = np.arange(lat_min, lat_max + 1e-9, step)
    lon = np.arange(lon_min, lon_max - 1e-9 if (lon_max - lon_min) >= 360 else lon_max + 1e-9, step)
    return lat, lon


def locate_loglik(unix_s, s, lat_grid, lon_grid, delays_min=(0.0,),
                  e0s=(-10.0, -7.0, -4.0, -1.0, 2.0), ws=(1.5, 3.0, 5.0), sigma=0.2, eps=0.05,
                  progress=None):
    """Log-likelihood of the brightness series for each (delay, lat, lon).

    Model: brightness s = sigmoid((sun_elevation - e0) / w) + noise, where e0 (the sun elevation at which
    the camera 'turns on') and w (how sharp the change is) are unknown and shared by all photos. They are
    averaged over. A small fraction `eps` of photos may be arbitrary outliers (clouds, glitches).
    """
    unix_s = np.asarray(unix_s, dtype=float)
    s = np.asarray(s, dtype=float)
    delays = np.asarray(delays_min, dtype=float)
    t = unix_s[None, :] - 60.0 * delays[:, None]                      # capture time = release - delay
    decl, eot = sun_params(t)
    minutes = np.mod(t, 86400.0) / 60.0
    h0 = (minutes + eot) / 4.0 - 180.0
    sd, cd = np.sin(decl).astype(np.float32), np.cos(decl).astype(np.float32)
    cos_h = np.cos(np.radians(h0[..., None] + np.asarray(lon_grid)[None, None, :])).astype(np.float32)
    params = [(e0, w) for e0 in e0s for w in ws]
    s_ = s.astype(np.float32)[None, :, None]
    lognorm = -math.log(sigma * math.sqrt(2.0 * math.pi))
    out = np.empty((len(delays), len(lat_grid), len(lon_grid)), dtype=float)
    for i, lat in enumerate(lat_grid):
        sp, cp = math.sin(math.radians(lat)), math.cos(math.radians(lat))
        sin_e = sp * sd[..., None] + cp * cd[..., None] * cos_h
        elev = np.degrees(np.arcsin(np.clip(sin_e, -1.0, 1.0)))
        acc = np.empty((len(params), len(delays), len(lon_grid)))
        for k, (e0, w) in enumerate(params):
            mu = 1.0 / (1.0 + np.exp(-(elev - e0) / w))
            g = np.exp(lognorm - 0.5 * ((s_ - mu) / sigma) ** 2)
            acc[k] = np.log((1.0 - eps) * g + eps).sum(axis=1)
        m = acc.max(axis=0)
        out[:, i, :] = m + np.log(np.exp(acc - m).mean(axis=0))
        if progress is not None and (i % 10 == 0 or i == len(lat_grid) - 1):
            progress(i + 1, len(lat_grid))
    return out


def _haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def posterior_from_loglik(logL, lat_grid, temper):
    m = logL.max()
    p = np.exp(temper * (logL - m)).mean(axis=0)               # average over unknown posting delays
    p = p * np.cos(np.radians(lat_grid))[:, None]               # equal area on the sphere
    return p / p.sum()


def summarise(P, lat_grid, lon_grid, mass=0.9, n_modes=3, mode_radius_km=800.0):
    dlat = float(np.median(np.diff(lat_grid))) if len(lat_grid) > 1 else 1.0
    dlon = float(np.median(np.diff(lon_grid))) if len(lon_grid) > 1 else 1.0
    cell_km2 = (EARTH_RADIUS_KM ** 2) * np.cos(np.radians(lat_grid))[:, None] * np.radians(dlat) * np.radians(dlon)
    cell_km2 = np.broadcast_to(cell_km2, P.shape)
    order = np.argsort(P.ravel())[::-1]
    cum = np.cumsum(P.ravel()[order])
    k = int(np.searchsorted(cum, mass) + 1)
    mask = np.zeros(P.size, dtype=bool)
    mask[order[:k]] = True
    mask = mask.reshape(P.shape)
    ii, jj = np.where(mask)
    area_km2 = float(cell_km2[mask].sum())
    half = np.sort(P.ravel())[::-1]
    k50 = int(np.searchsorted(np.cumsum(half), 0.5) + 1)
    m50 = np.zeros(P.size, dtype=bool)
    m50[order[:k50]] = True
    area50 = float(cell_km2[m50.reshape(P.shape)].sum())
    best = np.unravel_index(np.argmax(P), P.shape)
    lat_m, lon_m = P.sum(axis=1), P.sum(axis=0)

    def interval(grid, marg, lo=0.05, hi=0.95):
        c = np.cumsum(marg)
        return float(grid[np.searchsorted(c, lo)]), float(grid[min(np.searchsorted(c, hi), len(grid) - 1)])

    modes, remaining = [], P.copy()
    LA, LO = np.meshgrid(lat_grid, lon_grid, indexing="ij")
    for _ in range(n_modes):
        if remaining.max() <= 0:
            break
        b = np.unravel_index(np.argmax(remaining), remaining.shape)
        near = _haversine_km(lat_grid[b[0]], lon_grid[b[1]], LA, LO) <= mode_radius_km
        modes.append({"lat": float(lat_grid[b[0]]), "lon": float(lon_grid[b[1]]), "mass": float(P[near].sum())})
        remaining = np.where(near, 0.0, remaining)
    return {
        "best_lat": float(lat_grid[best[0]]), "best_lon": float(lon_grid[best[1]]),
        "region_mass": mass,
        "region_lat_range": (float(lat_grid[ii].min()), float(lat_grid[ii].max())),
        "region_lon_range": (float(lon_grid[jj].min()), float(lon_grid[jj].max())),
        "region_area_km2": area_km2, "region50_area_km2": area50,
        "region_hexagons": area_km2 * 1e6 / HEX_AREA_M2,
        "lat_90": interval(lat_grid, lat_m), "lon_90": interval(lon_grid, lon_m),
        "lat_weak": (interval(lat_grid, lat_m)[1] - interval(lat_grid, lat_m)[0]) > 12.0,
        "modes": modes,
        "mask": mask,
    }


def solve(table, region=(-60, 75, -180, 180), step=2.0, max_delay_min=30.0, delay_step_min=15.0,
          temper=0.05, max_photos=400, progress=None, **kw):
    """Run the whole model on a table that has columns `utc` and `s`."""
    table = table.sort_values("utc").reset_index(drop=True)
    if len(table) > max_photos:
        idx = np.linspace(0, len(table) - 1, max_photos).round().astype(int)
        table = table.iloc[np.unique(idx)]
    unix_s, s = unix_seconds(table["utc"]), table["s"].to_numpy(dtype=float)
    lat_grid, lon_grid = make_grid(region, step)
    delays = np.arange(0.0, max_delay_min + 1e-9, delay_step_min) if max_delay_min > 0 else np.array([0.0])
    logL = locate_loglik(unix_s, s, lat_grid, lon_grid, delays, progress=progress, **kw)
    P = posterior_from_loglik(logL, lat_grid, temper)
    res = Result(lat_grid, lon_grid, delays, logL, P, temper, len(table))
    res.summary = summarise(P, lat_grid, lon_grid)
    for d, delay in enumerate(delays):
        b = np.unravel_index(np.argmax(logL[d]), logL[d].shape)
        res.per_delay_best.append((float(delay), float(lat_grid[b[0]]), float(lon_grid[b[1]]), float(logL[d][b])))
    return res


def refine(res, table, step=0.5, pad_deg=3.0, **kw):
    """Re-run on a finer grid around the current 90% region."""
    la, lo = res.summary["region_lat_range"], res.summary["region_lon_range"]
    region = (la[0] - pad_deg, la[1] + pad_deg, lo[0] - pad_deg, lo[1] + pad_deg)
    return solve(table, region=region, step=step, max_delay_min=float(res.delays_min.max()),
                 delay_step_min=float(res.delays_min[1] - res.delays_min[0]) if len(res.delays_min) > 1 else 15.0,
                 temper=res.temper, **kw)


# ----------------------------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------------------------
def describe(res):
    s = res.summary
    lines = [
        f"Photos used: {res.n_photos}",
        f"Most likely longitude: {s['best_lon']:.1f}   (90% range {s['lon_90'][0]:.1f} to {s['lon_90'][1]:.1f})",
        f"Latitude 90% range: {s['lat_90'][0]:.1f} to {s['lat_90'][1]:.1f}",
        f"90% region: about {s['region_area_km2']:,.0f} km2  (~{s['region_hexagons']:.1e} hexagons of 60 ft)",
        f"50% region: about {s['region50_area_km2']:,.0f} km2",
    ]
    if not s["lat_weak"]:
        lines.append("Main candidate areas (share of probability within 800 km):")
        for m in s["modes"]:
            lines.append(f"   lat {m['lat']:.1f}, lon {m['lon']:.1f}  ->  {100 * m['mass']:.0f}%")
    if s["lat_weak"]:
        lines.append("NOTE: latitude is only weakly determined. Day length cannot be separated from the camera's "
                     "unknown exposure threshold, so treat the latitude range as 'could be anywhere in here'. "
                     "It improves as the photos span more weeks (especially across the December solstice).")
    if len(res.delays_min) > 1:
        lines.append("Best fit if photos were posted this long after being taken (shows how much the delay moves it):")
        for d, la, lo, _ll in res.per_delay_best:
            lines.append(f"   delay {d:>4.0f} min -> lat {la:.1f}, lon {lo:.1f}")
    return "\n".join(lines)


def plot_diagnostics(table, show=True):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 3.6))
    ax[0].scatter(table["utc"], table["s"], s=6)
    ax[0].set_title("Brightness over time (UTC)")
    ax[0].set_ylabel("brightness (0 dark, 1 bright)")
    ax[0].tick_params(axis="x", rotation=30)
    hours = table["utc"].dt.hour + table["utc"].dt.minute / 60.0
    day = (table["utc"] - table["utc"].min()).dt.total_seconds() / 86400.0
    sc = ax[1].scatter(hours, table["s"], c=day, s=6, cmap="viridis")
    ax[1].set_title("Brightness by hour of day (UTC)")
    ax[1].set_xlabel("hour of day (UTC)")
    ax[1].set_xlim(0, 24)
    plt.colorbar(sc, ax=ax[1], label="days since first photo")
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_posterior(res, show=True, true_point=None):
    import matplotlib.pyplot as plt
    lat, lon, P = res.lat_grid, res.lon_grid, res.P
    fig = plt.figure(figsize=(13, 5.2))
    ax = fig.add_axes([0.05, 0.1, 0.6, 0.82])
    im = ax.imshow(P / P.max(), origin="lower", aspect="auto", cmap="magma",
                   extent=[lon.min(), lon.max(), lat.min(), lat.max()])
    ax.set_title("Where the day/night timing fits (brighter = more likely)")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.grid(alpha=0.25)
    for name, (la, lo) in CITIES.items():
        if lat.min() <= la <= lat.max() and lon.min() <= lo <= lon.max():
            ax.plot(lo, la, "c.", ms=4)
            ax.annotate(name, (lo, la), fontsize=7, color="c", xytext=(3, 3), textcoords="offset points")
    if true_point is not None:
        ax.plot(true_point[1], true_point[0], "g*", ms=14, label="true location (demo)")
        ax.legend(loc="lower left")
    plt.colorbar(im, ax=ax, label="relative probability")
    ax2 = fig.add_axes([0.72, 0.58, 0.26, 0.32])
    ax2.plot(P.sum(axis=1), lat)
    ax2.set_title("Latitude", fontsize=9)
    ax2.set_ylabel("lat")
    ax3 = fig.add_axes([0.72, 0.12, 0.26, 0.32])
    ax3.plot(lon, P.sum(axis=0))
    ax3.set_title("Longitude (depends on posting delay)", fontsize=9)
    ax3.set_xlabel("lon")
    if show:
        plt.show()
    return fig


# ----------------------------------------------------------------------------------------------
# Simulation (demo and tests)
# ----------------------------------------------------------------------------------------------
def simulate_photos(true_lat, true_lon, start="2026-11-01 00:00", days=21, every_min=60, delay_min=0.0,
                    e0=-3.0, w=2.5, noise=0.08, outlier_frac=0.04, cloudy=True, seed=0):
    """Fake brightness series for a camera at (true_lat, true_lon). `utc` is the RELEASE time."""
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp(start, tz="UTC")
    n = int(days * 24 * 60 / every_min)
    release = [t0 + pd.Timedelta(minutes=every_min * i + float(rng.uniform(0, 3))) for i in range(n)]
    ux = unix_seconds(pd.Series(release))
    capture = ux - 60.0 * delay_min
    elev = solar_elevation_deg(capture, true_lat, true_lon)
    s = 1.0 / (1.0 + np.exp(-(elev - e0) / w))
    if cloudy:
        day_factor = rng.uniform(0.55, 1.0, size=days + 2)
        s = s * day_factor[((ux - ux[0]) // 86400).astype(int)]
    s = np.clip(s + rng.normal(0, noise, n), 0, 1)
    bad = rng.random(n) < outlier_frac
    s[bad] = rng.uniform(0, 1, bad.sum())
    return pd.DataFrame({"photo": np.arange(1, n + 1), "utc": pd.Series(release, dtype="datetime64[ns, UTC]"), "s": s})
