"""Tests for watcher/watch.py against a local fake site. Run: python test_watch.py"""
import csv
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
WATCH = os.path.join(ROOT, "watcher", "watch.py")


class Site:
    """Controllable fake of the public parts of the site."""
    ids = [1, 2, 3]
    list_format = "dict"          # "dict" | "list" | "count" | "garbage"
    status_list = 200
    forbid_images = False
    throttle_first = 0            # number of 429s to send before succeeding
    seen_uas = []
    hits = []
    lum = {}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body=b"", ctype="application/json", extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        Site.hits.append(self.path)
        Site.seen_uas.append(self.headers.get("User-Agent", ""))
        if self.path == "/api/images":
            if Site.throttle_first > 0:
                Site.throttle_first -= 1
                return self._send(429, b"slow down", "text/plain", {"Retry-After": "0"})
            if Site.status_list != 200:
                return self._send(Site.status_list)
            if Site.list_format == "dict":
                return self._send(200, json.dumps({"images": Site.ids}).encode())
            if Site.list_format == "list":
                return self._send(200, json.dumps(Site.ids).encode())
            if Site.list_format == "count":
                return self._send(200, json.dumps({"count": max(Site.ids)}).encode())
            return self._send(200, b"<html>not json</html>", "text/html")
        if self.path.startswith("/img/"):
            if Site.forbid_images:
                return self._send(403)
            name = self.path.split("/")[-1]
            n = int(name.split(".")[0])
            if n not in Site.ids:
                return self._send(404)
            if name.endswith(".t.webp"):
                lum = Site.lum.get(n, 100)
                buf = io.BytesIO()
                Image.new("RGB", (64, 64), (lum, lum, lum)).save(buf, "WEBP")
                return self._send(200, buf.getvalue(), "image/webp")
            return self._send(404)
        return self._send(404)


def start():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def reset():
    Site.ids, Site.list_format, Site.status_list = [1, 2, 3], "dict", 200
    Site.forbid_images, Site.throttle_first, Site.seen_uas, Site.hits, Site.lum = False, 0, [], [], {}


def run(port, data_dir, **env):
    e = dict(os.environ)
    e.update({"SITE_BASE": f"http://127.0.0.1:{port}", "CONTACT_EMAIL": "me@example.com",
              "PERMISSION_CONFIRMED": "yes", "DATA_DIR": data_dir, "REQUEST_PAUSE_S": "0",
              "RETRY_SLEEP_CAP": "0", "no_proxy": "127.0.0.1,localhost", "NO_PROXY": "127.0.0.1,localhost"})
    e.update({k: str(v) for k, v in env.items()})
    for k in [k for k, v in e.items() if v is None]:
        e.pop(k)
    p = subprocess.run([sys.executable, WATCH], env=e, capture_output=True, text=True, timeout=60)
    return p.returncode, p.stdout + p.stderr


def rows(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def test_all():
    srv = start()
    port = srv.server_address[1]
    results = []

    def check(name, cond, extra=""):
        results.append((name, bool(cond)))
        print(("PASS  " if cond else "FAIL  ") + name + (f"   {extra}" if extra and not cond else ""))

    # 1. refuses without permission / contact
    reset()
    with tempfile.TemporaryDirectory() as d:
        code, out = run(port, d, PERMISSION_CONFIRMED="no")
        check("refuses without PERMISSION_CONFIRMED=yes", code == 2 and not Site.hits, out)
        code, out = run(port, d, CONTACT_EMAIL="")
        check("refuses without a contact email", code == 2 and not Site.hits, out)

    # 2. first run -> everything is backfill, nothing is timestamped
    reset()
    with tempfile.TemporaryDirectory() as d:
        code, out = run(port, d)
        check("first run exits 0", code == 0, out)
        check("first run: existing photos go to backfill only", len(rows(f"{d}/backfill.csv")) == 3 and not rows(f"{d}/photos.csv"), out)
        check("first run downloads no images", not any(h.startswith("/img/") for h in Site.hits))
        check("User-Agent carries the contact email", all("me@example.com" in u for u in Site.seen_uas))
        check("one list request per run", Site.hits.count("/api/images") == 1)

        # 3. a new photo appears
        Site.ids = [1, 2, 3, 4]
        Site.lum[4] = 200
        code, out = run(port, d)
        live = rows(f"{d}/photos.csv")
        check("new photo recorded once as live", code == 0 and [r["photo"] for r in live] == ["4"], out)
        check("brightness is measured (~200)", live and abs(float(live[0]["lum"]) - 200) < 3, str(live))
        check("released_at is an ISO UTC time", live and live[0]["released_at"].endswith("Z"))
        check("previous poll time stored", live and live[0]["prev_poll_utc"].endswith("Z"))

        # 4. rerun adds no duplicates
        code, out = run(port, d)
        check("rerun creates no duplicate", len(rows(f"{d}/photos.csv")) == 1, out)

        # 5. two new photos, one is not yet downloadable -> only the ready one recorded, other retried
        Site.ids = [1, 2, 3, 4, 5, 6]
        Site.lum[5] = 30
        orig = Site.lum
        saved = list(Site.ids)
        Site.ids = [1, 2, 3, 4, 5, 6]
        # make photo 6 listed but image 404: serve by temporarily removing from image availability
        Site.lum.pop(6, None)
        H_img_orig = H.do_GET

        def patched(self):
            if self.path.startswith("/img/6."):
                return self._send(404)
            return H_img_orig(self)
        H.do_GET = patched
        code, out = run(port, d)
        H.do_GET = H_img_orig
        got = [r["photo"] for r in rows(f"{d}/photos.csv")]
        check("unavailable image is skipped and not marked seen", got == ["4", "5"], str(got) + out)
        code, out = run(port, d)
        got = [r["photo"] for r in rows(f"{d}/photos.csv")]
        check("skipped photo is picked up next run", got == ["4", "5", "6"], str(got) + out)

        # 6. an outage gap -> new photos are treated as backfill, not live
        state = json.load(open(f"{d}/state.json"))
        state["last_ok_utc"] = "2020-01-01T00:00:00Z"
        json.dump(state, open(f"{d}/state.json", "w"))
        Site.ids = [1, 2, 3, 4, 5, 6, 7]
        code, out = run(port, d)
        check("after a long gap, new photo is backfill (times unreliable)",
              "7" not in [r["photo"] for r in rows(f"{d}/photos.csv")]
              and "7" in [r["photo"] for r in rows(f"{d}/backfill.csv")], out)

        # 7. burst of many new photos -> backfill
        Site.ids = list(range(1, 60))
        code, out = run(port, d, MAX_LIVE_PER_RUN=10)
        check("a burst larger than MAX_LIVE_PER_RUN is treated as a catch-up",
              "30" in [r["photo"] for r in rows(f"{d}/backfill.csv")] and "30" not in [r["photo"] for r in rows(f"{d}/photos.csv")], out)

    # 8. 403 on the list -> stop, nothing written
    reset()
    with tempfile.TemporaryDirectory() as d:
        Site.status_list = 403
        code, out = run(port, d)
        check("403 on the photo list stops with exit 3 and writes nothing",
              code == 3 and not os.path.exists(f"{d}/state.json"), out)

    # 9. 403 on an image -> stop, photo not marked seen
    reset()
    with tempfile.TemporaryDirectory() as d:
        run(port, d)                                   # first run
        Site.ids = [1, 2, 3, 4]
        Site.forbid_images = True
        code, out = run(port, d)
        check("403 on an image stops with exit 3", code == 3, out)
        check("refused photo is not marked as seen", 4 not in json.load(open(f"{d}/state.json"))["known"])
        check("no live row written", not rows(f"{d}/photos.csv"))

    # 10. 429 then success -> retried politely
    reset()
    with tempfile.TemporaryDirectory() as d:
        Site.throttle_first = 1
        code, out = run(port, d)
        check("429 is retried and then succeeds", code == 0 and Site.hits.count("/api/images") == 2, out)

    # 11. alternate list formats
    for fmt in ("list", "count"):
        reset()
        with tempfile.TemporaryDirectory() as d:
            Site.list_format = fmt
            code, out = run(port, d)
            check(f"parses '{fmt}' photo-list format", code == 0 and len(rows(f"{d}/backfill.csv")) == 3, out)

    # 12. unrecognised format -> exit 4 + sample saved
    reset()
    with tempfile.TemporaryDirectory() as d:
        Site.list_format = "garbage"
        code, out = run(port, d)
        check("unrecognised list format: exit 4 and a sample is saved",
              code == 4 and os.path.exists(f"{d}/raw_images_sample.txt") and not os.path.exists(f"{d}/state.json"), out)

    # 13. unit test of the parser
    sys.path.insert(0, os.path.join(ROOT, "watcher"))
    import watch
    check("parser: dicts with ids", watch.parse_photo_ids([{"id": 2}, {"id": 1}]) == [1, 2])
    check("parser: string numbers", watch.parse_photo_ids({"photos": ["1", "2"]}) == [1, 2])
    check("parser: nonsense -> None", watch.parse_photo_ids({"x": "y"}) is None and watch.parse_photo_ids([]) is None)
    srv.shutdown()

    failed = [n for n, ok in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(test_all())
