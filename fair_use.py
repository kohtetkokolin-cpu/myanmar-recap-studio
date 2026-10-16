"""fair_use.py - Fair-use / copyright transform presets for Myanmar Recap Studio.

Reuses the exact FX keys already understood by the frontend (`FX` in index.html)
and the render graph, so a preset is just a partial FX dict to merge.

Provides:
  * PRESETS              - named transform presets ("safe", "strong", ...)
  * apply_preset(fx, n)  - merge a preset into an fx dict (returns new dict)
  * auto_scene_fx(i)     - per-scene transform deltas that VARY across the movie
                           (mirror flip / punch / hue drift ...) to strengthen
                           the transformative character of the output
  * legal_note()         - a short disclaimer string

This module is additive and does not import main.py. These tools only *help*
transform the footage; they are not a legal guarantee.
"""

from __future__ import annotations

import copy

# Full default FX shape (mirror of the frontend default) so presets merge safely.
FX_DEFAULT = {
    "freeze": {"on": False, "interval": 5, "duration": 3, "zoom": 1.5,
               "mode": "lock", "zoom_dir": "in", "snap": True},
    "mirror": "off", "color": False, "noise": 0, "rotation": False,
    "punch": 0, "edge": 0, "kenburns": 0, "shake": 0, "vignette": 0,
    "sharpen": 0, "denoise": 0, "brightness": 0, "contrast": 0,
    "saturation": 0, "gamma": 0, "hue": 0, "tone": "none",
    "bgfill": False, "bgblur": 25, "chroma": 0, "lens": 0,
    "border": {"on": False, "thick": 0.7, "color": "White"},
    "wm": {"text": "", "x": 90, "y": 7, "size": 3.2, "opacity": 70, "color": "White"},
}

PRESETS = {
    "off": {
        "label": "မထည့် (Original)",
        "help": "transform မထည့် — မူရင်းအတိုင်း။",
        "fx": {},
    },
    "safe": {
        "label": "အနည်းငယ် ပြောင်း (Light)",
        "help": "မူရင်းနဲ့ မတူအောင် အနည်းငယ် — mirror + crop + vignette + color။",
        "fx": {"mirror": "alt", "color": True, "punch": 4, "edge": 2,
               "kenburns": 3, "vignette": 15, "brightness": 2,
               "contrast": 6, "saturation": 6},
    },
    "strong": {
        "label": "သိသိသာသာ ပြောင်း (Strong)",
        "help": "transform ပိုပြင်းထင် — တရားမျှတမှု (fair-use) ပိုထင်ရှား။",
        "fx": {"mirror": "alt", "color": True, "punch": 8, "edge": 4,
               "kenburns": 6, "shake": 0.3, "vignette": 35, "brightness": 5,
               "contrast": 14, "saturation": 15, "hue": 6, "tone": "cinematic",
               "chroma": 1.5, "lens": 15,
               "border": {"on": True, "thick": 0.7, "color": "White"}},
    },
    "cinematic": {
        "label": "Cinematic ဆန်ဆန်",
        "help": "teal-orange tone + vignette + letterbox border။",
        "fx": {"tone": "cinematic", "vignette": 25, "contrast": 10,
               "kenburns": 4,
               "border": {"on": True, "thick": 1.2, "color": "Black"}},
    },
    "shorts_9x16": {
        "label": "Shorts / Reels (9:16)",
        "help": "မှုန်နောက်ခံဖြည့် — 16:9 ကို 9:16 အတွက်။",
        "fx": {"bgfill": True, "bgblur": 30},
    },
}


def apply_preset(fx, name):
    """Merge preset `name` into fx (new dict). Unknown name -> unchanged copy."""
    base = copy.deepcopy(fx) if isinstance(fx, dict) else copy.deepcopy(FX_DEFAULT)
    preset = PRESETS.get(str(name).lower())
    if not preset:
        return base
    for k, v in preset["fx"].items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k].update(v)
        else:
            base[k] = copy.deepcopy(v)
    return base


def auto_scene_fx(scene_index):
    """Deterministic per-scene transform deltas that vary across the movie.

    Alternate mirror direction and drift punch/hue so consecutive scenes are
    not byte-identical transforms of the source.
    """
    i = int(scene_index)
    return {
        "mirror": "all" if (i % 2 == 0) else "off",
        "punch": round(3.0 + (i % 4) * 1.5, 1),      # 3.0, 4.5, 6.0, 7.5
        "kenburns": round(2.0 + (i % 3) * 1.5, 1),
        "hue": ((i * 7) % 13) - 6,                   # -6 .. +6
        "brightness": (i % 3) - 1,                   # -1 .. +1
    }


def legal_note():
    return ("Fair-use အတွက် transform များ ထည့်ပေးထားပါသည်။ "
            "ဤအရာများသည် အကူအညီသာဖြစ်ပြီး ဥပဒေအရ အာမခံချက် မဟုတ်ပါ။ "
            "အသုံးပြုမှုတိုင်းအတွက် တာဝန်ကို အသုံးပြုသူ ကိုယ်တိုင်ယူရပါမည်။")


def list_presets():
    return [{"name": k, "label": v["label"], "help": v["help"]}
            for k, v in PRESETS.items()]


if __name__ == "__main__":
    import json
    f = apply_preset(FX_DEFAULT, "safe")
    assert f["mirror"] == "alt" and f["punch"] == 4, "safe preset merge failed"
    s = apply_preset(FX_DEFAULT, "strong")
    assert s["border"]["on"] is True, "nested border merge failed"
    assert FX_DEFAULT["mirror"] == "off", "default mutated!"
    a0, a1 = auto_scene_fx(0), auto_scene_fx(1)
    assert a0["mirror"] != a1["mirror"], "auto transform not varying"
    print("SELFTEST PASS")
    print(json.dumps(list_presets(), ensure_ascii=False, indent=1))
