"""media_utils.py - ffmpeg helpers for Myanmar Recap Studio.

Currently provides hardware-accelerated H.264 encoder detection so renders can
use the GPU (NVENC / QSV / AMF / VideoToolbox) when available, and fall back to
libx264 (CPU) otherwise. Output is visually equivalent; HW is just faster.

This module is additive and does not import main.py.
"""

from __future__ import annotations

import functools
import subprocess

# Preference order: fastest/most-compatible first.
_HW_CANDIDATES = [
    "h264_nvenc",         # NVIDIA
    "h264_qsv",           # Intel Quick Sync
    "h264_amf",           # AMD
    "h264_videotoolbox",  # Apple
]

_QUALITY = {"fast": "fast", "balanced": "balanced", "quality": "quality"}


def _encoders_output():
    try:
        r = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                           capture_output=True, text=True)
        return r.stdout or ""
    except Exception:
        return ""


def _runtime_ok(encoder):
    """Actually try a 1-frame encode; a listed encoder can still fail at runtime."""
    try:
        r = subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=size=128x96:rate=5:duration=0.4",
            "-frames:v", "1", "-c:v", encoder, "-an", "-f", "null", "-",
        ], capture_output=True, text=True, timeout=30)
        return r.returncode == 0
    except Exception:
        return False


@functools.lru_cache(maxsize=1)
def available_hw_encoders():
    """Hardware H.264 encoders that are listed AND pass a runtime encode test."""
    text = _encoders_output()
    listed = [e for e in _HW_CANDIDATES if e in text]
    return [e for e in listed if _runtime_ok(e)]


def pick_h264_encoder(prefer="auto"):
    """Return an encoder name. prefer='cpu' forces libx264; 'auto' uses HW if any."""
    if str(prefer).lower() in ("cpu", "libx264", "software"):
        return "libx264"
    hw = available_hw_encoders()
    if str(prefer).lower() in ("nvenc", "qsv", "amf", "videotoolbox"):
        want = f"h264_{str(prefer).lower()}"
        return want if want in hw else "libx264"
    return hw[0] if hw else "libx264"


def h264_encode_args(crf=23, quality="balanced", prefer="auto"):
    """ffmpeg output args for H.264 that work with both HW and CPU encoders.

    Returns e.g. ['-c:v','h264_nvenc','-preset','p4','-cq','23','-pix_fmt','yuv420p'].
    """
    enc = pick_h264_encoder(prefer)
    q = _QUALITY.get(str(quality).lower(), "balanced")
    crf = int(crf)
    if enc == "h264_nvenc":
        preset = {"fast": "p1", "balanced": "p4", "quality": "p6"}[q]
        return ["-c:v", enc, "-preset", preset, "-rc", "vbr", "-cq", str(crf), "-pix_fmt", "yuv420p"]
    if enc == "h264_qsv":
        return ["-c:v", enc, "-global_quality", str(crf), "-pix_fmt", "nv12"]
    if enc == "h264_amf":
        amf_q = {"fast": "speed", "balanced": "balanced", "quality": "quality"}[q]
        return ["-c:v", enc, "-quality", amf_q, "-qp_i", str(crf), "-qp_p", str(crf), "-pix_fmt", "yuv420p"]
    if enc == "h264_videotoolbox":
        return ["-c:v", enc, "-q:v", str(crf), "-pix_fmt", "yuv420p"]
    # CPU fallback
    preset = {"fast": "veryfast", "balanced": "medium", "quality": "slow"}[q]
    return ["-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p"]


def summary():
    hw = available_hw_encoders()
    return {
        "hw_encoders": hw,
        "chosen": pick_h264_encoder("auto"),
        "hw_available": bool(hw),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(summary(), indent=2))
