"""movie_index.py - Analyze-once "Movie Index" for Myanmar Recap Studio.

Builds a reusable, disk-cached index for a source video:

    {
      "key":       <stable file id>,
      "source":    <path>,
      "duration":  <seconds>,
      "shots":     [{"start":..,"end":..}, ...],      # PySceneDetect
      "transcript":[{"start":..,"end":..,"text":..}],  # faster-whisper
      "units":     [[start,end], ...],                 # 5s address grid
      "characters":[{"name","gender","role"}],         # AI (optional)
      "summary":   "...",                              # AI (optional)
      "meta":      {"detail","created","local_only"}
    }

Why this exists
---------------
The expensive step in every recap is *understanding the movie*. Today each
mode (Auto / Dialogue / Story) and each style re-runs that work. The index is
built **once** and then reused by every later render, so re-rendering in a new
style/voice costs **zero** analysis and zero API calls.

Design rules
------------
* Additive module. It does NOT import main.py (avoids circular imports).
* Local-first: shots + transcript + units are produced locally (free).
* AI is optional and injected via a `deps` object exposing:
      llm_text(prompt, schema, label) -> dict      (required for AI detail)
      llm_video(prompt, clip, schema, label) -> dict  (optional, "deep")
      log(msg), progress(done, total, label)          (optional)
  With deps=None the index is built local-only and still fully usable.
* Any AI failure degrades gracefully to a local-only index.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import time

# ----------------------------------------------------------------------------
# defaults / tiny helpers
# ----------------------------------------------------------------------------

DEFAULT_CACHE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "movie_index")
UNIT_QUANTUM = 5.0
VALID_DETAILS = ("local", "balanced", "deep")

_whisper_cache = {}


def _log(deps, msg):
    fn = getattr(deps, "log", None)
    if callable(fn):
        try:
            fn(msg)
            return
        except Exception:
            pass
    try:
        print(f"[movie_index] {msg}")
    except UnicodeEncodeError:
        print(f"[movie_index] {msg.encode('ascii', 'replace').decode('ascii')}")


def _progress(deps, done, total, label=""):
    fn = getattr(deps, "progress", None)
    if callable(fn):
        try:
            fn(int(done), int(total), label)
        except Exception:
            pass


def cache_key(path):
    """Stable id of a video file (size + head/tail bytes) for the disk cache."""
    try:
        size = os.path.getsize(path)
    except OSError:
        size = 0
    h = hashlib.sha1()
    h.update(str(size).encode())
    try:
        with open(path, "rb") as f:
            h.update(f.read(65536))
            if size > 131072:
                f.seek(-65536, os.SEEK_END)
                h.update(f.read(65536))
    except OSError:
        pass
    return h.hexdigest()[:16]


def _run(cmd, label=""):
    return subprocess.run(cmd, capture_output=True, text=True)


# ----------------------------------------------------------------------------
# local evidence: duration, shots, audio, transcript, units
# ----------------------------------------------------------------------------

def probe_duration(path):
    """Return media duration in seconds (0.0 on failure)."""
    r = _run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
              "-of", "default=noprint_wrappers=1:nokey=1", path])
    try:
        return max(0.0, float((r.stdout or "").strip()))
    except (TypeError, ValueError):
        return 0.0


def _extract_audio(path, wav_path, sr=16000):
    _run(["ffmpeg", "-y", "-i", path, "-vn", "-ac", "1", "-ar", str(sr), wav_path])
    return wav_path


def detect_shots(path, min_cut_sec=1.5, fps_fallback=24.0):
    """PySceneDetect content cuts -> [{"start","end"}]. Never raises."""
    try:
        from scenedetect import open_video, SceneManager
        from scenedetect.detectors import ContentDetector
        video = open_video(path)
        sm = SceneManager()
        try:
            fps = float(video.frame_rate)
        except (AttributeError, TypeError, ValueError):
            fps = fps_fallback
        min_len = max(1, int(round(max(1.0, float(min_cut_sec)) * fps)))
        sm.add_detector(ContentDetector(threshold=27.0, min_scene_len=min_len))
        sm.detect_scenes(video=video)
        shots = []
        for s, e in sm.get_scene_list():
            shots.append({"start": round(s.get_seconds(), 3), "end": round(e.get_seconds(), 3)})
        return shots
    except Exception as e:  # noqa: BLE001
        print(f"[movie_index] shot detection failed: {e}")
        return []


def build_units(duration_sec, quantum=UNIT_QUANTUM):
    """Deterministic contiguous ~5s address grid: [[start,end], ...]."""
    try:
        duration = max(0.1, float(duration_sec))
        q = max(2.0, float(quantum))
    except (TypeError, ValueError):
        duration, q = 0.1, 5.0
    units, start = [], 0.0
    while start < duration - 0.01:
        end = min(duration, start + q)
        units.append([round(start, 3), round(end, 3)])
        start = end
    return units


def _get_whisper():
    from faster_whisper import WhisperModel
    key = "cpu_int8"
    if key not in _whisper_cache:
        name = os.environ.get("WHISPER_MODEL", "base")
        _whisper_cache[key] = WhisperModel(name, device="cpu", compute_type="int8")
    return _whisper_cache[key]


def transcribe(wav_path, lang=None):
    """faster-whisper -> [{"start","end","text"}]. Never raises."""
    try:
        model = _get_whisper()
        segs, _info = model.transcribe(wav_path, beam_size=1, vad_filter=True, language=lang or None)
        out = []
        for x in segs:
            t = str(getattr(x, "text", "") or "").strip()
            if t:
                out.append({"start": round(float(x.start), 3), "end": round(float(x.end), 3), "text": t})
        return out
    except Exception as e:  # noqa: BLE001
        print(f"[movie_index] transcription failed: {e}")
        return []


# ----------------------------------------------------------------------------
# optional AI understanding (Gemini via injected deps)
# ----------------------------------------------------------------------------

INDEX_SCHEMA = {
    "type": "object",
    "properties": {
        "title_guess": {"type": "string"},
        "summary": {"type": "string"},
        "characters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "gender": {"type": "string"},
                    "role": {"type": "string"},
                },
                "required": ["name"],
            },
        },
        "beats": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "number"},
                    "end": {"type": "number"},
                    "note": {"type": "string"},
                },
                "required": ["note"],
            },
        },
    },
    "required": ["summary"],
}


def _transcript_text(transcript, limit_chars=None):
    parts = []
    for seg in transcript:
        ts = f"[{int(seg['start']) // 60:02d}:{int(seg['start']) % 60:02d}]"
        parts.append(f"{ts} {seg['text']}")
    text = "\n".join(parts)
    if limit_chars and len(text) > limit_chars:
        text = text[:limit_chars]
    return text


def _ai_understand(deps, video, duration, transcript, detail):
    """One/few batched text calls -> {summary, characters, beats}. Never raises."""
    llm_text = getattr(deps, "llm_text", None)
    if not callable(llm_text) or detail == "local" or not transcript:
        return {}
    body = _transcript_text(transcript)
    if not body:
        return {}
    prompt = (
        "You are a film analyst. Below is a timestamped transcript of a movie.\n"
        "Return a JSON object with: a concise plot summary, the main characters "
        "(name, gender, role), and the major story beats with their start/end "
        "seconds taken ONLY from the given timestamps.\n\n"
        f"TRANSCRIPT:\n{body}"
    )
    try:
        _log(deps, f"Movie Index: AI understanding ({detail}) ...")
        obj = llm_text(prompt, INDEX_SCHEMA, "index")
        if isinstance(obj, dict):
            return obj
    except Exception as e:  # noqa: BLE001
        _log(deps, f"Movie Index: AI step skipped ({str(e)[:120]})")
    return {}


# ----------------------------------------------------------------------------
# build / load
# ----------------------------------------------------------------------------

def index_path_for(video, cache_root=None):
    root = cache_root or DEFAULT_CACHE_ROOT
    return os.path.join(root, cache_key(video), "index.json")


def index_exists(video, cache_root=None):
    return os.path.isfile(index_path_for(video, cache_root))


def load_index(video, cache_root=None):
    p = index_path_for(video, cache_root)
    if not os.path.isfile(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("meta", {})["loaded_from_cache"] = True
        return data
    except Exception:
        return None


def save_index(video, units, transcript, shots=None, cache_root=None,
               detail="local", summary="", characters=None, duration=None):
    """Persist an index from already-computed evidence (no recompute).

    Used by the pipelines so the first run's local work (scene detect +
    Whisper) is reused by every later run of any mode.
    """
    key = cache_key(video)
    if duration is None:
        duration = units[-1][1] if units else probe_duration(video)
    index = {
        "key": key,
        "source": os.path.abspath(video),
        "duration": round(float(duration), 3),
        "shots": shots or [],
        "transcript": transcript or [],
        "units": [[float(a), float(b)] for a, b in units],
        "characters": characters or [],
        "summary": summary or "",
        "beats": [],
        "meta": {
            "detail": detail,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "local_only": not bool(summary),
            "shot_count": len(shots or []),
            "unit_count": len(units or []),
            "segment_count": len(transcript or []),
            "saved_by": "pipeline",
        },
    }
    p = index_path_for(video, cache_root)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=1)
    return index


def build_index(video, deps=None, cache_root=None, duration=None, lang=None,
                detail="balanced", rebuild=False, quantum=UNIT_QUANTUM):
    """Build (or reuse) the movie index. Returns the index dict.

    detail: "local" (no AI) | "balanced" (one batched text call) | "deep".
    """
    detail = detail if detail in VALID_DETAILS else "balanced"
    if not rebuild:
        cached = load_index(video, cache_root)
        if cached is not None:
            _log(deps, "Movie Index: cache မှ ပြန်သုံးပါမည် (analysis ၀)")
            return cached

    key = cache_key(video)
    _log(deps, f"Movie Index: တည်ဆောက်နေပါသည် ({detail}) ...")
    if duration is None:
        duration = probe_duration(video)

    tmp = tempfile.mkdtemp(prefix="mv_idx_")
    try:
        wav = os.path.join(tmp, "audio.wav")
        _extract_audio(video, wav)
        _progress(deps, 1, 4, "shots")
        shots = detect_shots(video)
        _progress(deps, 2, 4, "transcript")
        wlang = "my" if str(lang).lower() in ("my", "burmese") else (lang or None)
        transcript = transcribe(wav, lang=wlang)
    finally:
        try:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
        except Exception:
            pass

    units = build_units(duration, quantum=quantum)
    _progress(deps, 3, 4, "ai")
    ai = _ai_understand(deps, video, duration, transcript, detail)
    _progress(deps, 4, 4, "done")

    index = {
        "key": key,
        "source": os.path.abspath(video),
        "duration": round(float(duration), 3),
        "shots": shots,
        "transcript": transcript,
        "units": units,
        "characters": ai.get("characters", []) if isinstance(ai, dict) else [],
        "summary": ai.get("summary", "") if isinstance(ai, dict) else "",
        "beats": ai.get("beats", []) if isinstance(ai, dict) else [],
        "meta": {
            "detail": detail,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "local_only": not bool(ai),
            "shot_count": len(shots),
            "unit_count": len(units),
            "segment_count": len(transcript),
        },
    }
    p = index_path_for(video, cache_root)
    try:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=1)
        _log(deps, f"Movie Index: သိမ်းပြီး ({len(shots)} shots, {len(transcript)} segs)")
    except Exception as e:  # noqa: BLE001
        _log(deps, f"Movie Index: သိမ်းလို့မရ ({str(e)[:120]})")
    return index


# ----------------------------------------------------------------------------
# self-test (no API key, no app needed)
# ----------------------------------------------------------------------------

def _selftest():
    tmp = tempfile.mkdtemp(prefix="mv_idx_test_")
    clip = os.path.join(tmp, "sample.mp4")
    print("[selftest] generating 12s synthetic clip ...")
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc=size=320x240:rate=15:duration=12",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=12",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", clip,
    ], capture_output=True)
    assert os.path.isfile(clip), "test clip not created (ffmpeg missing?)"

    cache_root = os.path.join(tmp, "cache")
    idx = build_index(clip, deps=None, cache_root=cache_root, detail="local")
    assert idx["duration"] > 10, f"bad duration {idx['duration']}"
    assert idx["units"], "no units"
    assert len(idx["units"]) >= 2, "expected >=2 five-second units"
    assert idx["meta"]["local_only"] is True, "local build should be local_only"

    cached = build_index(clip, deps=None, cache_root=cache_root, detail="local")
    assert cached.get("meta", {}).get("loaded_from_cache") is True, "cache not reused"

    print("[selftest] PASS", {
        "duration": idx["duration"],
        "shots": len(idx["shots"]),
        "units": len(idx["units"]),
        "segments": len(idx["transcript"]),
        "cached_reused": True,
    })
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print(__doc__)
