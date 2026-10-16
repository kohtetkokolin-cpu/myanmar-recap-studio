"""styles_config.py - file-based style management for Myanmar Recap Studio.

Lets recap / story / dialogue styles be edited, added or removed from a single
`styles.yaml` file instead of code. Built-in styles stay the source of truth;
the YAML only OVERRIDES or ADDS on top, and any parse error is ignored so the
app always starts.

YAML shape (all sections optional):

    recap:
      cinematic:
        label: "My Cinematic"
        label_mm: "ကျွန်တော့် ဆင်နီမာ"
        voice: "STYLE RULES ..."        # full prompt string (replaces builtin)
      my_new_style:                      # a brand-new style
        label: "..."
        label_mm: "..."
        voice: "..."
    story:
      case_file:
        label: "၄။ အမှုဖိုင် (ပြင်ပြီး)"
        rules: "PERSONA: ..."           # replaces the builtin rules
      delete_me:
        delete: true                     # remove a builtin style
    dialogue:
      recap_dialogue:
        label: "..."
"""

from __future__ import annotations

import copy
import os

STYLES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "styles.yaml")

_cache = {"path": None, "data": None, "mtime": None}


def load_overrides(path=None):
    """Return parsed overrides {section: {name: {...}}}. Never raises."""
    path = path or STYLES_FILE
    if not os.path.isfile(path):
        return {}
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception as e:  # noqa: BLE001
        try:
            print(f"[styles_config] styles.yaml ဖတ်လို့မရ ({str(e)[:120]}) - builtin သုံးမည်")
        except UnicodeEncodeError:
            print("[styles_config] styles.yaml parse failed - using builtins")
        return {}


def _load_cached(path=None):
    path = path or STYLES_FILE
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = None
    if _cache["path"] == path and _cache["mtime"] == mtime and _cache["data"] is not None:
        return _cache["data"]
    data = load_overrides(path)
    _cache.update({"path": path, "mtime": mtime, "data": data})
    return data


def merge(builtin, section, path=None):
    """Return a new dict = builtin merged with overrides for `section`.

    * an override dict updates the matching builtin entry's fields
    * a name not in builtin is ADDED
    * an entry with {"delete": true} is REMOVED
    """
    out = copy.deepcopy(builtin) if isinstance(builtin, dict) else {}
    overrides = _load_cached(path).get(section, {})
    if not isinstance(overrides, dict):
        return out
    for name, spec in overrides.items():
        if not isinstance(spec, dict):
            continue
        if spec.get("delete"):
            out.pop(name, None)
            continue
        if name in out and isinstance(out[name], dict):
            out[name].update({k: v for k, v in spec.items() if k != "delete"})
        else:
            out[name] = {k: v for k, v in spec.items() if k != "delete"}
    return out


def apply_recap_overrides(builtin, path=None):
    return merge(builtin, "recap", path)


def apply_story_overrides(builtin, path=None):
    return merge(builtin, "story", path)


def apply_dialogue_overrides(builtin, path=None):
    return merge(builtin, "dialogue", path)


def export_template(path=None, recap=None, story=None, dialogue=None):
    """Write a starter styles.yaml from the current built-ins (edit-friendly)."""
    path = path or STYLES_FILE
    try:
        import yaml
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"PyYAML required: {e}")
    data = {"recap": recap or {}, "story": story or {}, "dialogue": dialogue or {}}
    header = ("# styles.yaml - Myanmar Recap Studio\n"
              "# Edit / add / remove styles here. Fields you set OVERRIDE the builtin;\n"
              "# a name not built-in is ADDED; add `delete: true` to remove one.\n"
              "# Restart (or reload) the app to apply.\n\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False, width=100)
    return path


if __name__ == "__main__":
    import tempfile
    import yaml
    sample = {
        "cinematic": {"label": "Natural Cinematic", "voice": "BUILTIN"},
        "old_style": {"label": "Old", "voice": "x"},
    }
    tmp = os.path.join(tempfile.mkdtemp(), "styles.yaml")
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.safe_dump({
            "recap": {
                "cinematic": {"label": "Edited Label"},
                "brand_new": {"label": "New", "voice": "NEWVOICE"},
                "old_style": {"delete": True},
            }
        }, f, allow_unicode=True)

    merged = merge(sample, "recap", path=tmp)
    assert merged["cinematic"]["label"] == "Edited Label", "override failed"
    assert merged["cinematic"]["voice"] == "BUILTIN", "should keep unset fields"
    assert "brand_new" in merged and merged["brand_new"]["voice"] == "NEWVOICE", "add failed"
    assert "old_style" not in merged, "delete failed"
    assert sample["cinematic"]["label"] == "Natural Cinematic", "builtin mutated!"
    assert load_overrides("Z:/nope/styles.yaml") == {}, "missing file should be empty"
    print("SELFTEST PASS", sorted(merged.keys()))
