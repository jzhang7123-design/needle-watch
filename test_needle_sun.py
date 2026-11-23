"""Tests for needle_sun.py. Run:  python test_needle_sun.py   (or: pytest -q)"""
import math
import os
import sys
import tempfile
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import needle_sun as ns  # noqa: E402


def U(text):
    return ns.unix_seconds(pd.Series([pd.Timestamp(text, tz="UTC")]))[0]


# ------------------------------------------------------------------ astronomy checks
def test_declination_at_solstices():
    d_jun = math.degrees(ns.sun_params(U("2026-06-21 12:00"))[0])
    d_dec = math.degrees(ns.sun_params(U("2026-12-21 12:00"))[0])
    assert abs(d_jun - 23.43) < 0.1, d_jun
    assert abs(d_dec + 23.43) < 0.1, d_dec


def test_equation_of_time_known_values():
    assert abs(ns.sun_params(U("2026-11-03 12:00"))[1] - 16.4) < 0.4      # yearly maximum
    assert abs(ns.sun_params(U("2026-02-11 12:00"))[1] + 14.2) < 0.4      # yearly minimum
    assert abs(ns.sun_params(U("2026-04-15 12:00"))[1]) < 1.0             # crosses zero
    assert abs(ns.sun_params(U("2026-11-22 12:00"))[1] - 13.9) < 0.4


def test_noon_elevation_london_winter_solstice():
    e = float(ns.solar_elevation_deg(U("2026-12-21 11:58"), 51.5, 0.0))
    assert abs(e - (90 - 51.5 - 23.44)) < 0.3, e


def test_equator_day_length_about_12h():
    t = U("2026-11-22 00:00") + 60 * np.arange(0, 1440)          # every minute
    up = (ns.solar_elevation_deg(t, 0.0, 10.0) > -0.833).sum() / 60.0
    assert abs(up - 12.0) < 0.2, up


def test_zurich_day_length_late_november():
    # With the usual -0.833 deg sunrise definition the day is ~9 h 05 min long
    t = U("2026-11-22 00:00") + 30 * np.arange(0, 2880)
    up = (ns.solar_elevation_deg(t, 47.37, 8.54) > -0.833).sum() * 30 / 3600.0
    assert abs(up - 9.08) < 0.12, up


def test_antimeridian_wraps():
    t = U("2026-11-22 09:30")
    a = float(ns.solar_elevation_deg(t, 20.0, 179.9))
    b = float(ns.solar_elevation_deg(t, 20.0, -180.1))
    assert abs(a - b) < 1e-9


# ------------------------------------------------------------------ time parsing
def test_timezone_parsing():
    got = ns.to_utc_timestamps(["2026-11-22 14:05", "2026-11-22T14:05:00-05:00", "2026-11-22T14:05:00Z"], "Europe/Zurich")
    assert [t.strftime("%H:%M") for t in got] == ["13:05", "19:05", "14:05"]


# ------------------------------------------------------------------ the model
def _solve(df, **kw):
    kw.setdefault("region", (-60, 75, -180, 180))
    kw.setdefault("step", 3.0)
    kw.setdefault("max_delay_min", 0.0)
    return ns.solve(df, **kw)


def _lon_err(res, lon):
    return ((res.summary["best_lon"] - lon + 180) % 360) - 180


def _contains(res, lat, lon, slack=1):
    ii, jj = np.argmin(abs(res.lat_grid - lat)), np.argmin(abs(res.lon_grid - lon))
    return bool(res.summary["mask"][max(ii - slack, 0):ii + slack + 1, max(jj - slack, 0):jj + slack + 1].any())


def test_longitude_is_recovered_on_a_clean_series():
    df = ns.simulate_photos(35.5, -83.5, seed=1, cloudy=False, outlier_frac=0.0)
    res = ns.solve(df, region=(20, 55, -110, -60), step=1.0, max_delay_min=0.0)
    assert abs(_lon_err(res, -83.5)) <= 2.0, _lon_err(res, -83.5)
    assert _contains(res, 35.5, -83.5, slack=0)


def test_longitude_and_containment_with_clouds_and_outliers():
    for seed, (la, lo) in enumerate([(44.5, -71.0), (35.5, -83.5), (51.0, 10.0), (-34.0, 18.4)]):
        res = _solve(ns.simulate_photos(la, lo, seed=seed + 10))
        assert abs(_lon_err(res, lo)) <= 6.0, (la, lo, _lon_err(res, lo))
        assert _contains(res, la, lo), (la, lo)


def test_latitude_is_reported_as_weak_not_falsely_precise():
    res = _solve(ns.simulate_photos(35.5, -83.5, seed=1))
    lo, hi = res.summary["lat_90"]
    assert hi - lo > 12.0, (lo, hi)                      # honest: wide
    assert res.summary["lat_weak"]
    assert lo <= 35.5 <= hi, (lo, hi)                    # ...but it contains the truth
    text = ns.describe(res)
    assert "weakly determined" in text and "Main candidate areas" not in text   # no fake-precise peaks


def test_hemisphere_is_only_weakly_separated_and_reported_honestly():
    # Northern sites are clearly northern; southern sites are NOT firmly resolved, and the code must not claim they are.
    north = _solve(ns.simulate_photos(34.0, 18.4, seed=5))
    assert north.P[north.lat_grid > 0].sum() > 0.9
    south = _solve(ns.simulate_photos(-50.0, 18.4, seed=5))
    assert south.P[south.lat_grid > 0].sum() < 0.2
    south_mid = _solve(ns.simulate_photos(-15.0, 18.4, seed=5))
    assert south_mid.summary["lat_weak"]                       # wide range flagged as weak


def test_posting_delay_shifts_longitude_about_15_degrees_per_hour():
    df = ns.simulate_photos(40.0, -80.0, delay_min=60.0, seed=3, outlier_frac=0.02)
    naive = _solve(df)                                   # wrongly assumes photos are posted instantly
    assert 8.0 <= abs(_lon_err(naive, -80.0)) <= 20.0, _lon_err(naive, -80.0)
    assert naive.summary["lat_90"][0] <= 40.0 <= naive.summary["lat_90"][1]


def test_delay_prior_that_covers_the_truth_contains_it():
    la, lo = 40.0, -80.0
    df = ns.simulate_photos(la, lo, delay_min=60.0, seed=3, outlier_frac=0.02)
    res = _solve(df, max_delay_min=90.0, delay_step_min=15.0)
    assert _contains(res, la, lo), "true location should lie inside the 90% region"
    delays = [d for d, _la, _lo, _ll in res.per_delay_best]
    assert max(delays) == 90.0 and len(delays) == 7


def test_robust_to_many_bad_frames():
    df = ns.simulate_photos(38.0, -90.0, seed=7, outlier_frac=0.15)
    res = _solve(df)
    assert abs(_lon_err(res, -90.0)) <= 6.0, _lon_err(res, -90.0)
    assert _contains(res, 38.0, -90.0)


def test_calibration_of_90_percent_region():
    rng = np.random.default_rng(42)
    hits, n = 0, 16
    for k in range(n):
        la, lo = float(rng.uniform(-45, 65)), float(rng.uniform(-170, 170))
        e0, w = float(rng.uniform(-8, 2)), float(rng.uniform(1.2, 6.0))
        df = ns.simulate_photos(la, lo, seed=100 + k, delay_min=float(rng.choice([0, 15, 30])), e0=e0, w=w,
                                days=int(rng.choice([10, 21])))
        res = _solve(df, max_delay_min=30.0, delay_step_min=15.0)
        hits += _contains(res, la, lo)
    assert hits >= int(0.85 * n), f"only {hits}/{n} 90% regions contained the truth"


def test_csv_loading_path_keeps_timezones_and_solves():
    sim = ns.simulate_photos(36.0, -82.0, days=14, every_min=30, delay_min=10, seed=3)
    csvdf = pd.DataFrame({"photo": sim["photo"], "released_at": sim["utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                          "lum": np.expm1(np.log1p(4) + sim["s"] * (np.log1p(130) - np.log1p(4))).round(2)})
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "photos.csv")
        csvdf.to_csv(p, index=False)
        table = ns.read_photo_table(p, tz="Europe/Zurich")      # 'Z' times must ignore the local tz
        assert str(table["utc"].dt.tz) == "UTC"
        assert table["utc"].iloc[0] == sim["utc"].iloc[0].floor("s")
        table, _ = ns.normalise_brightness(table)
        res = ns.solve(table, region=(20, 55, -110, -60), step=2.0, max_delay_min=30.0)
        assert abs(_lon_err(res, -82.0)) <= 8.0, _lon_err(res, -82.0)
        assert _contains(res, 36.0, -82.0)


def test_local_times_without_offset_use_the_given_timezone():
    t = ns.to_utc_timestamps(["2026-11-22 08:00"], "America/New_York")[0]
    assert t.strftime("%H:%M") == "13:00"


def test_flat_data_is_flagged():
    df = ns.simulate_photos(40.0, -80.0, days=3)
    df["lum"] = 120.0
    tab, notes = ns.normalise_brightness(df)
    assert any("barely change" in n for n in notes)
    assert ns.check_coverage(tab)


def test_short_series_is_flagged():
    df = ns.simulate_photos(40.0, -80.0, days=1, every_min=120)
    assert any("span only" in w or "Only" in w for w in ns.check_coverage(df))


def test_typed_brightness_path():
    df = ns.simulate_photos(40.0, -80.0, days=10, seed=2)
    df["brightness"] = np.round(df["s"] * 2) / 2.0         # someone rating photos 0 / 0.5 / 1 by eye
    tab, notes = ns.normalise_brightness(df.drop(columns=["s"]))
    assert "typed" in notes[0]
    res = ns.solve(tab, region=(20, 60, -110, -50), step=2.0, max_delay_min=0.0)
    assert abs(_lon_err(res, -80.0)) <= 8.0, _lon_err(res, -80.0)
    assert _contains(res, 40.0, -80.0)


# ------------------------------------------------------------------ image pipeline
def test_image_pipeline_end_to_end():
    from PIL import Image
    sim = ns.simulate_photos(41.0, -77.0, days=8, every_min=90, seed=4)
    with tempfile.TemporaryDirectory() as d:
        rng = np.random.default_rng(0)
        for _, row in sim.iterrows():
            lum = float(np.expm1(np.log1p(4) + row["s"] * (np.log1p(130) - np.log1p(4))))
            base = np.clip(rng.normal(lum, 6, (60, 80, 3)), 0, 255).astype("uint8")
            canvas = np.zeros((150, 200, 3), dtype="uint8")        # dark "browser frame" around the photo
            canvas[30:140, 40:190] = np.asarray(Image.fromarray(base).resize((150, 110)))
            Image.fromarray(canvas).save(os.path.join(d, f"photo_{int(row['photo'])}.png"))
        imgs = ns.collect_images(d)
        assert len(imgs) == len(sim)
        tab, missing = ns.add_brightness(sim.drop(columns=["s"]), imgs, crop=(0.3, 0.3, 0.9, 0.9))
        assert not missing
        tab, notes = ns.normalise_brightness(tab)
        corr = np.corrcoef(tab["s"], sim["s"])[0, 1]
        assert corr > 0.95, corr
        res = ns.solve(tab, region=(20, 60, -110, -50), step=2.0, max_delay_min=0.0)
        assert abs(_lon_err(res, -77.0)) <= 8.0, _lon_err(res, -77.0)
        assert _contains(res, 41.0, -77.0)


def test_missing_images_are_reported():
    sim = ns.simulate_photos(41.0, -77.0, days=2, every_min=240)
    tab, missing = ns.add_brightness(sim.drop(columns=["s"]), {}, crop=(0, 0, 1, 1))
    assert len(tab) == 0 and len(missing) == len(sim)


def test_posterior_is_a_distribution_and_summary_is_sane():
    res = _solve(ns.simulate_photos(40.0, -80.0, seed=9))
    assert abs(res.P.sum() - 1.0) < 1e-9 and (res.P >= 0).all()
    s = res.summary
    assert s["region_area_km2"] > 0 and s["region50_area_km2"] <= s["region_area_km2"]
    assert s["region_hexagons"] > 1e6          # a timing-only fit never gets near a 60 ft hexagon


def test_runtime_of_global_two_degree_search_is_reasonable():
    df = ns.simulate_photos(40.0, -80.0, seed=9)
    t0 = time.time()
    ns.solve(df, region=(-60, 75, -180, 180), step=2.0, max_delay_min=30.0, delay_step_min=15.0, max_photos=400)
    assert time.time() - t0 < 120


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        t0 = time.time()
        try:
            fn()
            print(f"PASS  {name}  ({time.time() - t0:.1f}s)")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
