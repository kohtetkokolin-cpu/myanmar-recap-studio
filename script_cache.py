"""script_cache.py - cache generated scene scripts for Myanmar Recap Studio.

A recap's scenes_data (timestamps + narration) is deterministic given the
source movie and the analysis settings. When only *presentation* changes
(blur, subtitles, FX, voice, output resolution), the script is identical, so
re-running would waste Gemini calls re-writing the same text.

This cache keys scenes_data by (movie content + engine + all analysis params)
and reuses it, so FX/voice-only re-renders cost **zero** Gemini calls.

Safe by construction: any parameter that affects the text is part of the key,
and a miss simply recomputes exactly as before. Disable with MRS_SCRIPT_CACHE=0.
"""

from __future__ import annotations

import hashlib
import json
import os
import time

DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "script_cache")
ENABLED = str(os.environ.get("MRS_SCRIPT_CACHE", "1")).strip().lower() not in ("0", "false", "no", "off")


def movie_key(path):
    """Content id of the source (same scheme as the Movie Index)."""
    try:
        import movie_index
        return movie_index.cache_key(path)
    except Exception:
        try:
            return hashlib.sha1(os.path.abspath(path).encode()).hexdigest()[:16]
        except Exception:
            return "unknown"


def make_sig(movie_key, engine, **params):
    """Deterministic signature over every input that changes the script."""
    h = hashlib.sha1()
    h.update(str(movie_key).encode())
    h.update(b"|")
    h.update(str(engine).encode())
    for k in sorted(params):
        h.update(f"|{k}={params[k]}".encode("utf-8", "replace"))
    return h.hexdigest()[:20]


def _path(movie_key, sig, root=None):
    root = root or DEFAULT_ROOT
    return os.path.join(root, str(movie_key), f"{sig}.json")


def load(movie_key, sig, root=None):
    if not ENABLED:
        return None
    p = _path(movie_key, sig, root)
    if not os.path.isfile(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        scenes = d.get("scenes")
        return scenes if isinstance(scenes, list) and scenes else None
    except Exception:
        return None


def save(movie_key, sig, scenes, root=None, meta=None):
    if not ENABLED or not scenes:
        return
    try:
        p = _path(movie_key, sig, root)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                       "meta": meta or {}, "scenes": scenes}, f, ensure_ascii=False)
    except Exception:
        pass


if __name__ == "__main__":
    import tempfile
    root = tempfile.mkdtemp()
    mk = movie_key(__file__)
    sig = make_sig(mk, "ranked_local_v2", style="cinematic", minutes=40, ratio=40, lang="Burmese")
    sig2 = make_sig(mk, "ranked_local_v2", style="hook", minutes=40, ratio=40, lang="Burmese")
    assert sig != sig2, "signature must change with style"
    assert load(mk, sig, root) is None, "empty cache should miss"
    scenes = [{"start_time": "00:00:00.000", "end_time": "00:00:05.000", "script": "x"}]
    save(mk, sig, scenes, root=root)
    got = load(mk, sig, root)
    assert got == scenes, "round-trip failed"
    assert load(mk, sig2, root) is None, "different style must miss"
    print("SELFTEST PASS")
