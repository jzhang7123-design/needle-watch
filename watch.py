#!/usr/bin/env python3
"""watch.py - polite photo watcher for the Needle hunt. One run = one check, then exit.

What it does
  * asks the site's public photo list which photo numbers exist (ONE request per run);
  * for photos it has not seen before, downloads the small thumbnail and records
      photo number, the time WE first saw it, and the average brightness;
  * appends rows to data/photos.csv. That file feeds the Colab notebook.

What it never does: log in, submit guesses, buy anything, or hit the site more than a few requests per run.
It refuses to start unless PERMISSION_CONFIRMED=yes (you attest the site said OK). CONTACT_EMAIL is optional;
if set, it is added to the User-Agent so the site can reach you.

Honesty about timing: "first seen" is when this script noticed a photo, NOT when it was released or taken.
It lags the real release by up to one polling interval (plus scheduler delay). The first run can't know when the
existing photos were released, so they go to data/backfill.csv (numbers only) and are never used as timestamps.
"""
from __future__ import annotations

import csv
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

EXIT_OK, EXIT_CONFIG, EXIT_FORBIDDEN, EXIT_FORMAT, EXIT_NETWORK = 0, 2, 3, 4, 5
LIVE_FIELDS = ["photo", "released_at", "lum", "prev_poll_utc", "source_url"]
BACKFILL_FIELDS = ["photo", "noted_at", "reason"]


class Forbidden(Exception):
    pass


def now_utc():
    return datetime.now(timezone.utc)


def iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg):
    print(f"[{iso(now_utc())}] {msg}", flush=True)


# ------------------------------------------------------------------ HTTP (polite)
def http_get(url, user_agent, timeout=20, max_tries=3):
    """GET with a clear User-Agent. Respects Retry-After on 429/503. Stops (Forbidden) on 401/403."""
    sleep_cap = float(os.environ.get("RETRY_SLEEP_CAP", "60"))
    last = None
    for attempt in range(1, max_tries + 1):
        req = urllib.request.Request(url, headers={"User-Agent": user_agent, "Accept": "*/*"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise Forbidden(f"{url} -> HTTP {e.code}")
            if e.code == 404:
                return 404, b""
            if e.code in (429, 500, 502, 503, 504):
                try:
                    wait = float(e.headers.get("Retry-After", "") or 0)
                except ValueError:
                    wait = 0.0
                wait = min(max(wait, 2.0 ** attempt if sleep_cap > 0 else 0.0), sleep_cap)
                last = f"HTTP {e.code}"
                log(f"{url} -> HTTP {e.code}; waiting {wait:.0f}s (try {attempt}/{max_tries})")
                time.sleep(wait)
                continue
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = str(e)
            wait = min(2.0 ** attempt, sleep_cap)
            log(f"{url} -> {e}; waiting {wait:.0f}s (try {attempt}/{max_tries})")
            time.sleep(wait)
    raise ConnectionError(f"{url}: gave up ({last})")


# ------------------------------------------------------------------ parsing the photo list
def parse_photo_ids(payload):
    """Return a sorted list of photo numbers from whatever JSON shape the site uses, or None if unrecognised."""
    def as_int(v):
        if isinstance(v, bool):
            return None
        if isinstance(v, int):
            return v
        if isinstance(v, str) and v.strip().isdigit():
            return int(v.strip())
        if isinstance(v, dict):
            for k in ("id", "n", "num", "number", "photo", "image", "index"):
                if k in v:
                    return as_int(v[k])
        return None

    def from_list(lst):
        ids = [as_int(x) for x in lst]
        return sorted(set(i for i in ids if i is not None)) if lst and all(i is not None for i in ids) else None

    if isinstance(payload, list):
        return from_list(payload)
    if isinstance(payload, dict):
        for key in ("images", "photos", "ids", "list", "items", "published", "data"):
            if key in payload and isinstance(payload[key], list):
                got = from_list(payload[key])
                if got is not None:
                    return got
        for key in ("count", "total", "n", "latest", "last", "max"):    # only a count: assume 1..N
            if key in payload and isinstance(payload[key], int) and not isinstance(payload[key], bool):
                return list(range(1, payload[key] + 1))
    return None


def brightness(image_bytes):
    from PIL import Image
    import needle_sun as ns
    return ns.image_mean_luminance(io.BytesIO(image_bytes), crop=(0.15, 0.15, 0.85, 0.85))


# ------------------------------------------------------------------ storage
def read_state(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"known": [], "last_ok_utc": None, "runs": 0}


def write_state(path, state):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, path)


def append_rows(path, fields, rows):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerows(rows)


# ------------------------------------------------------------------ one run
def main():
    base = os.environ.get("SITE_BASE", "https://needle.global").rstrip("/")
    contact = os.environ.get("CONTACT_EMAIL", "").strip()      # optional
    if contact and "@" not in contact:
        log("CONTACT_EMAIL does not look like an email address. Fix it or leave it empty.")
        return EXIT_CONFIG
    if os.environ.get("PERMISSION_CONFIRMED", "").strip().lower() != "yes":
        log("Set PERMISSION_CONFIRMED=yes only if the site owners agreed to this monitoring. Not running.")
        return EXIT_CONFIG

    data_dir = os.environ.get("DATA_DIR", os.path.join(HERE, "..", "data"))
    os.makedirs(data_dir, exist_ok=True)
    state_path = os.path.join(data_dir, "state.json")
    state = read_state(state_path)
    known = set(state["known"])
    max_gap = float(os.environ.get("MAX_GAP_MIN", "45"))
    max_live = int(os.environ.get("MAX_LIVE_PER_RUN", "25"))
    pause = float(os.environ.get("REQUEST_PAUSE_S", "1.5"))
    ua = "NeedlePhotoWatcher/1.0 (personal research; low rate" + (f"; contact {contact})" if contact else ")")

    started = now_utc()
    try:
        status, body = http_get(f"{base}/api/images", ua)
    except Forbidden as e:
        log(f"The site refused the request ({e}). Stopping. Do not retry until you have talked to the site owners.")
        return EXIT_FORBIDDEN
    except (ConnectionError, urllib.error.URLError) as e:
        log(f"Network problem: {e}. Nothing recorded; will try again next run.")
        return EXIT_NETWORK
    if status != 200:
        log(f"Photo list returned HTTP {status}. Nothing recorded.")
        return EXIT_NETWORK

    try:
        payload = json.loads(body.decode("utf-8"))
        ids = parse_photo_ids(payload)
    except (ValueError, UnicodeDecodeError):
        payload, ids = None, None
    if ids is None:
        sample = os.path.join(data_dir, "raw_images_sample.txt")
        with open(sample, "wb") as f:
            f.write(body[:20000])
        log(f"Could not understand the photo list format. Saved a sample to {sample}. Send it to me so I can adapt the parser.")
        return EXIT_FORMAT

    new_ids = [i for i in ids if i not in known]
    prev_poll = state["last_ok_utc"]
    reason = None
    if not known:
        reason = "first run: existing photos, release time unknown"
    elif prev_poll:
        gap = (started - datetime.strptime(prev_poll, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)).total_seconds() / 60.0
        if gap > max_gap and new_ids:
            reason = f"gap of {gap:.0f} min since last good poll (> {max_gap:.0f}): release times unreliable"
    if new_ids and not reason and len(new_ids) > max_live:
        reason = f"{len(new_ids)} new photos at once (> {max_live}): looks like a catch-up, not live"

    live_rows, back_rows = [], []
    if new_ids and reason:
        back_rows = [{"photo": i, "noted_at": iso(started), "reason": reason} for i in new_ids]
        log(f"{len(new_ids)} photos recorded as backfill ({reason}).")
    else:
        for i in new_ids:
            lum, used = None, None
            for suffix in (f"{i}.t.webp", f"{i}.webp", f"{i}.png"):
                time.sleep(pause)
                try:
                    st, img = http_get(f"{base}/img/{suffix}", ua)
                except Forbidden as e:
                    log(f"Image request refused ({e}). Stopping without recording photo {i}.")
                    _commit(data_dir, state_path, state, known, live_rows, back_rows, prev_poll, started, partial=True)
                    return EXIT_FORBIDDEN
                if st == 200 and img:
                    try:
                        lum, used = brightness(img), f"{base}/img/{suffix}"
                        break
                    except Exception as e:  # noqa: BLE001  (unreadable image: try the next size)
                        log(f"photo {i}: could not decode {suffix}: {e}")
            if lum is None:
                log(f"photo {i}: image not available yet; will retry next run (not marked as seen).")
                continue
            live_rows.append({"photo": i, "released_at": iso(now_utc()), "lum": f"{lum:.2f}",
                              "prev_poll_utc": prev_poll or "", "source_url": used})
            log(f"new photo {i}: brightness {lum:.1f}")

    _commit(data_dir, state_path, state, known, live_rows, back_rows, prev_poll, started)
    log(f"done: {len(ids)} photos listed, {len(live_rows)} new live, {len(back_rows)} backfill.")
    return EXIT_OK


def _commit(data_dir, state_path, state, known, live_rows, back_rows, prev_poll, started, partial=False):
    if live_rows:
        append_rows(os.path.join(data_dir, "photos.csv"), LIVE_FIELDS, live_rows)
    if back_rows:
        append_rows(os.path.join(data_dir, "backfill.csv"), BACKFILL_FIELDS, back_rows)
    for r in live_rows + back_rows:
        known.add(int(r["photo"]))
    state["known"] = sorted(known)
    if not partial:
        state["last_ok_utc"] = iso(started)
    state["runs"] = state.get("runs", 0) + 1
    write_state(state_path, state)


if __name__ == "__main__":
    sys.exit(main())
