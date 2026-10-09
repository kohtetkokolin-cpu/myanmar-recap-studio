"""dialogue_v2 - plan / write / fit pipeline for the three Dialogue styles.

Styles
  recap           narrator-only story recap
  recap_dialogue  narrator + source-grounded male/female character lines
  dialogue_only   translated dialogue, one single TTS voice, no narrator

Design (why sync and length are reliable)
  1. ANALYZE once per movie (cached on disk): local Whisper transcript, local
     shot cuts, and one Gemini *video* call per 5-minute window that returns
     small scene cards (time range, summary, importance, speaker genders).
  2. PLAN deterministically in Python: every beat gets a time budget in
     SECONDS (not "N words") and a footage window that cannot overlap the
     next beat.
  3. WRITE with text-only Gemini calls, giving each beat its own max_chars
     (calibrated from a real TTS sample of the chosen voice).
  4. FIT: synthesize every beat, measure it, rewrite only the beats that run
     too long, then (if still long) speed that beat up slightly.  The audio
     duration - not a Gemini guess - decides how much footage is shown.
  5. LENGTH control: drop the least important beats or add the next most
     important ones until the measured total is within a few percent of the
     target.

Everything that touches the outside world (Gemini, TTS, Whisper, ffmpeg
helpers) is injected through `Deps`, so the planning/fit logic is testable
without network access.
"""
import bisect
import hashlib
import json
import math
import os
import re
import shutil
import subprocess

STYLE_RECAP = "recap"
STYLE_RECAP_DIALOGUE = "recap_dialogue"
STYLE_DIALOGUE_ONLY = "dialogue_only"
STYLES = (STYLE_RECAP, STYLE_RECAP_DIALOGUE, STYLE_DIALOGUE_ONLY)

WINDOW_SEC = 300.0          # one Gemini video call per 5 minutes of movie
CARD_MAX = 8.0              # scene cards are split to <= this many seconds
S_MIN, S_MAX = 3.0, 12.0    # narrator beat slot bounds (seconds)
LEAD_MAX, LEAD_MIN = 5.0, 2.0
IMP_W = {1: 0.55, 2: 0.75, 3: 0.95, 4: 1.1, 5: 1.3}
TOLERANCE = 0.04            # accept total within +/-4% of target
MAX_OVER = 1.10             # a beat may run 10% over its slot before rewrite
MAX_TEMPO = 1.15            # last-resort audio speed-up for one beat
BATCH = 24                  # beats per Gemini writing call (bigger batches = fewer API calls)
PAUSE_SEC = 0.15            # pause between lines inside one beat
DUP_SIM = 0.22              # char-3gram similarity above which two narration beats count as repeats
DIALOGUE_SHARE = 0.40       # share of the target length given to character dialogue (style 2)
LITERARY_RE = re.compile(r"(သည်(?!း)|၏|တွင်(?!း)|၍|၎င်း|သို့)")   # book-style Burmese markers
_GOLD = 0.6180339887498949


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
class Deps:
    """Injected capabilities.  All callables are synchronous."""
    def __init__(self, llm_video, llm_text, tts, transcribe, probe, run,
                 log=None, progress=None, fatal=(), llm_light=None):
        self.llm_video = llm_video      # (prompt, clip_path, schema, label) -> obj
        self.llm_text = llm_text        # (prompt, schema, label) -> obj
        self.tts = tts                  # (text, voice_id, out_path) -> [word timings]
        self.transcribe = transcribe    # (media_path) -> [{start,end,text}]
        self.probe = probe              # (media_path) -> seconds
        self.run = run                  # (ffmpeg_cmd_list, label) -> None (raises)
        self.llm_light = llm_light or llm_text   # optional cheaper model for rewrite/shorten calls
        self.log = log or (lambda m: None)
        self.progress = progress or (lambda done, total, label="": None)
        self.fatal = tuple(fatal)       # exception types that must abort the job


class PipelineError(RuntimeError):
    pass


_PUNCT_RE = re.compile(r"[\s\.,!?;:\"'()\[\]{}\-\u2013\u2014\u2026\u104a\u104b\u201c\u201d\u2018\u2019]")


def speak_len(text):
    """Number of speakable characters (the unit used for all length budgets)."""
    return len(_PUNCT_RE.sub("", str(text or "")))


def _gold_key(i):
    return (i * _GOLD) % 1.0


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def fmt_time(sec):
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec - h * 3600 - m * 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def cache_key(path):
    """Stable id of a video file (size + head/tail bytes) for the analysis cache."""
    size = os.path.getsize(path)
    h = hashlib.sha1(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(1 << 20))
        if size > (2 << 20):
            f.seek(max(0, size - (1 << 20)))
            h.update(f.read(1 << 20))
    return h.hexdigest()[:20]


def _jload(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _jsave(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


def detect_cuts(video, threshold=0.35, timeout=3600):
    """Local shot-boundary times (seconds).  Never raises; [] on failure."""
    try:
        cmd = ["ffmpeg", "-hide_banner", "-nostats", "-i", video, "-an", "-vf",
               f"fps=4,scale=160:-2,select='gt(scene,{threshold})',showinfo",
               "-f", "null", "-"]
        p = subprocess.run(cmd, capture_output=True, timeout=timeout)
        err = p.stderr.decode("utf-8", errors="replace")
        return sorted({round(float(x), 3) for x in re.findall(r"pts_time:([0-9]+(?:\.[0-9]+)?)", err)})
    except Exception:
        return []


def snap(t, cuts, tol):
    if not cuts:
        return t
    i = bisect.bisect_left(cuts, t)
    best, bd = t, tol + 1e-9
    for j in (i - 1, i):
        if 0 <= j < len(cuts) and abs(cuts[j] - t) < bd:
            best, bd = cuts[j], abs(cuts[j] - t)
    return best


def transcript_text(rows, a, b, limit=400):
    out = []
    for r in rows:
        if r["end"] > a and r["start"] < b:
            out.append(r["text"].strip())
    return " ".join(out)[:limit]


# --------------------------------------------------------------------------
# 1. ANALYZE
# --------------------------------------------------------------------------
ANALYSIS_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "scenes": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
            "start": {"type": "NUMBER"}, "end": {"type": "NUMBER"},
            "summary": {"type": "STRING"}, "importance": {"type": "INTEGER"},
            "has_dialogue": {"type": "BOOLEAN"},
            "speakers": {"type": "ARRAY", "items": {"type": "STRING"}},
        }, "required": ["start", "end", "summary", "importance"]}},
        "characters": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
            "name": {"type": "STRING"}, "gender": {"type": "STRING"}, "age": {"type": "STRING"},
            "role": {"type": "STRING"},
        }, "required": ["name", "gender"]}},
    },
    "required": ["scenes"],
}


def _analysis_prompt(win_len, rows, known, recent):
    return f"""You are the story analyst for a movie-recap editor. Watch this {win_len:.0f}-second clip (one window of a longer movie).
All timestamps you return are SECONDS measured from the START OF THIS CLIP (0 to {win_len:.0f}) as plain numbers.

Return JSON: {{"scenes": [...], "characters": [...]}}

scenes: chronological, NON-overlapping, covering every story-relevant moment of the clip.
- One scene = one continuous action or conversation, between 3 and 8 seconds long. Split anything longer into consecutive scenes.
- Each scene's summary must describe ONLY what is NEW in that part. Never repeat or paraphrase the previous scene's summary; if nothing new happens, merge the two scenes.
- Skip black frames, credits and pure filler.
- start, end: seconds (numbers).
- summary: ONE factual English sentence: what is visibly happening and what is said. Do not invent names, motives or events.
- importance: 1-5. 5 = turning point / twist / reveal / climax, 4 = important plot progress, 3 = normal progress, 2 = minor, 1 = filler. Use the full range; most scenes must NOT be 3.
- has_dialogue: true only if people clearly speak on screen.
- speakers: the NAMES of the people who speak in this scene, in order of first speaking. Use the exact names from KNOWN CHARACTERS or from "characters" below; for a person with no name use a short stable label such as "old woman" and list that same label in "characters".

characters: every important person visible in this clip: name (or a short visual label if unnamed), gender ("male"/"female"/"unknown"), age ("child"/"teen"/"adult"/"elder"), role (max 8 words). Re-use the exact names from KNOWN CHARACTERS for the same person.

KNOWN CHARACTERS: {json.dumps(known, ensure_ascii=False)}
PREVIOUS EVENTS: {json.dumps(recent, ensure_ascii=False)}
LOCAL SPEECH TRANSCRIPT (ASR, may contain errors; times relative to this clip): {json.dumps(rows, ensure_ascii=False)}
"""


def _normalize_cards(raw_scenes, win_start, win_len, cuts, rows_abs, prefix):
    cards = []
    items = []
    for s in raw_scenes or []:
        try:
            a, b = float(s.get("start")), float(s.get("end"))
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(a) and math.isfinite(b)):
            continue
        a = _clamp(a, 0.0, win_len)
        b = _clamp(b, 0.0, win_len)
        if b - a < 0.8:
            continue
        items.append((a, b, s))
    items.sort(key=lambda x: x[0])
    prev_end = 0.0
    for a, b, s in items:
        a = max(a, prev_end)
        if b - a < 0.8:
            continue
        A, B = win_start + a, win_start + b
        A = snap(A, cuts, 0.8)
        B = snap(B, cuts, 0.8)
        if B - A < 0.8:
            A, B = win_start + a, win_start + b
        gid = len(cards)
        pieces = max(1, int(math.ceil((B - A) / CARD_MAX - 1e-9)))
        step = (B - A) / pieces
        try:
            imp = int(s.get("importance", 3))
        except (TypeError, ValueError):
            imp = 3
        imp = int(_clamp(imp, 1, 5))
        for k in range(pieces):
            ca, cb = A + k * step, A + (k + 1) * step
            cards.append({
                "id": f"{prefix}_{len(cards):03d}",
                "start": round(ca, 3), "end": round(cb, 3), "dur": round(cb - ca, 3),
                "summary": str(s.get("summary", "")).strip()[:300],
                "imp": imp,
                "has_dialogue": bool(s.get("has_dialogue")),
                "group": f"{prefix}g{gid}", "cont": k > 0,
                "speakers_raw": [str(g).strip() for g in (s.get("speakers") or []) if str(g).strip()][:4],
            })
        prev_end = b
    return cards


def analyze(video, duration, deps, cache_dir, transcript, cuts, work_dir, w_range=None, known_init=None, recent_init=None):
    """Return (cards, characters, warnings).  Resumable: one cache file per window."""
    os.makedirs(cache_dir, exist_ok=True)
    n_win = max(1, int(math.ceil(duration / WINDOW_SEC - 1e-9)))
    cards, known, warnings = [], {}, []
    for ent in (known_init or []):
        nm = str(ent.get("name", "")).strip()
        if nm:
            known[nm.lower()] = dict(ent)
    recent_seed = list(recent_init or [])
    w_first, w_last = (0, n_win) if not w_range else (max(0, int(w_range[0])), min(n_win, int(w_range[1])))
    for w in range(w_first, w_last):
        w0 = w * WINDOW_SEC
        wl = min(WINDOW_SEC, duration - w0)
        if wl < 2.0:
            continue
        cpath = os.path.join(cache_dir, f"win_{w:03d}.json")
        data = _jload(cpath)
        if data is None:
            rows = [{"start": round(r["start"] - w0, 2), "end": round(r["end"] - w0, 2), "text": r["text"][:160]}
                    for r in transcript if r["end"] > w0 and r["start"] < w0 + wl][:120]
            recent = ([c["summary"] for c in cards[-6:]] or recent_seed)[-6:]
            clip = os.path.join(work_dir, f"win_{w:03d}.mp4")
            deps.log(f"🔎 Analysis {w + 1}/{n_win} — {w0 / 60:.0f}-{(w0 + wl) / 60:.0f} မိနစ် ကို ခွဲခြမ်းစိတ်ဖြာနေပါသည်...")
            deps.run(["ffmpeg", "-y", "-ss", str(w0), "-t", str(wl), "-i", video,
                      "-vf", "scale=-2:360,fps=2", "-c:v", "libx264", "-preset", "veryfast", "-crf", "32",
                      "-c:a", "aac", "-b:a", "48k", "-ac", "1", clip], f"dv2 window {w + 1}")
            prompt = _analysis_prompt(wl, rows, list(known.values())[:30], recent)
            last_err = None
            for attempt in (1, 2):
                try:
                    data = deps.llm_video(prompt, clip, ANALYSIS_SCHEMA, f"Analysis {w + 1}/{n_win}")
                    if isinstance(data, list):
                        data = {"scenes": data, "characters": []}
                    if not isinstance(data, dict) or not data.get("scenes"):
                        raise ValueError("analysis returned no scenes")
                    break
                except deps.fatal:
                    raise
                except Exception as e:  # noqa: BLE001
                    last_err = e
                    data = None
                    deps.log(f"⚠️ Analysis {w + 1}/{n_win} attempt {attempt} မအောင်မြင်ပါ: {e}")
            try:
                os.remove(clip)
            except OSError:
                pass
            if data is None:
                warnings.append(f"window {w + 1}/{n_win} skipped: {last_err}")
                deps.progress(w + 1, n_win, "analysis")
                continue
            _jsave(cpath, data)
        for ch in data.get("characters") or []:
            nm = str(ch.get("name", "")).strip()
            if nm and nm.lower() not in ("male", "female"):
                old = known.get(nm.lower(), {})
                g = str(ch.get("gender", "unknown")).lower()
                a = str(ch.get("age", "adult")).lower()
                known[nm.lower()] = {"name": old.get("name", nm),
                                     "gender": g if g in ("male", "female") else old.get("gender", "unknown"),
                                     "age": a if a in ("child", "teen", "adult", "elder") else old.get("age", "adult"),
                                     "role": str(ch.get("role", ""))[:60] or old.get("role", "")}
        new_cards = _normalize_cards(data.get("scenes"), w0, wl, cuts, transcript, f"w{w:02d}")
        for c in new_cards:
            c["speakers"], c["genders"] = [], []
            for nm in c.pop("speakers_raw", []):
                if nm.lower() in ("male", "female"):          # legacy cache: gender only
                    c["genders"].append(nm.lower())
                    continue
                ent = known.get(nm.lower())
                if ent is None:
                    ent = known[nm.lower()] = {"name": nm, "gender": "unknown", "age": "adult", "role": ""}
                if ent["name"] not in c["speakers"]:
                    c["speakers"].append(ent["name"])
                if ent["gender"] in ("male", "female"):
                    c["genders"].append(ent["gender"])
        if cards and new_cards and new_cards[0]["start"] < cards[-1]["end"]:
            shift = cards[-1]["end"]
            if new_cards[0]["end"] - shift < 0.8:
                new_cards = new_cards[1:]
            else:
                new_cards[0]["start"] = round(shift, 3)
                new_cards[0]["dur"] = round(new_cards[0]["end"] - shift, 3)
        cards.extend(new_cards)
        deps.progress(w + 1, n_win, "analysis")
    for i, c in enumerate(cards):
        c["idx"] = i
    if not cards:
        raise PipelineError("Gemini analysis produced no usable scenes"
                            + (": " + "; ".join(warnings[:3]) if warnings else ""))
    return cards, list(known.values()), warnings


# --------------------------------------------------------------------------
# speech units (Whisper clusters)
# --------------------------------------------------------------------------
def make_units(transcript, max_gap=0.7, max_len=12.0, min_len=0.3):
    units, cur = [], None
    for r in sorted(transcript, key=lambda x: x["start"]):
        a, b, t = float(r["start"]), float(r["end"]), str(r["text"]).strip()
        if b <= a or not t:
            continue
        if cur and a - cur["end"] <= max_gap and (b - cur["start"]) <= max_len:
            cur["end"] = max(cur["end"], b)
            cur["text"] += " " + t
        else:
            if cur:
                units.append(cur)
            cur = {"start": a, "end": b, "text": t}
    if cur:
        units.append(cur)
    return [u for u in units if u["end"] - u["start"] >= min_len]


def merge_units(units, max_gap=1.5, short=2.0, max_len=9.0, join_gap=3.0):
    """Merge rapid-fire / very short utterances into 'turns' so each dubbing beat has room to breathe."""
    units = [dict(u) for u in units]
    out = []
    for u in units:
        if out:
            p = out[-1]
            gap = u["start"] - p["end"]
            plen, ulen = p["end"] - p["start"], u["end"] - u["start"]
            fits = (u["end"] - p["start"]) <= max_len
            if fits and (gap <= max_gap or ((plen < short or ulen < short) and gap <= join_gap)):
                p["end"] = max(p["end"], u["end"])
                p["text"] = (p["text"] + " " + u["text"]).strip()
                continue
        out.append(u)
    return [u for u in out if u["end"] - u["start"] >= 0.8]


def attach_units(cards, units):
    """Give every card the speech units that start inside it."""
    for c in cards:
        c["units"] = []
    starts = [c["start"] for c in cards]
    for u in units:
        i = bisect.bisect_right(starts, u["start"] + 1e-6) - 1
        if i < 0:
            i = 0
        cards[i]["units"].append(u)
    for c in cards:
        c["has_dialogue"] = bool(c["units"])


# --------------------------------------------------------------------------
# 2. PLAN
# --------------------------------------------------------------------------
class Beat(dict):
    pass


def _new_beat(counter, kind, card, src_start, cap, slot, **kw):
    b = Beat(id=f"b{counter[0]:04d}", kind=kind, card=card["id"], card_idx=card["idx"],
             src_start=round(float(src_start), 3), cap=round(float(cap), 3), slot=round(float(slot), 3),
             imp=card["imp"], summary=card["summary"], source_text="", genders=list(card.get("genders", [])),
             lead=False, lines=[], path=None, dur=0.0, words=[], pos="middle",
             group=card.get("group"), speakers=list(card.get("speakers", [])),
             cont=bool(card.get("cont")), card_end=card["end"])
    counter[0] += 1
    b.update(kw)
    return b


def plan_narration(cards, target, src_dur, counter, blockers=(), exclude=()):
    """Choose cards and give each a slot (seconds) so the slots sum to `target`.

    Footage for a beat starts at its card start and may not run into the next
    selected card (or any blocker start).
    """
    idxs = [i for i in range(len(cards)) if i not in exclude]
    if not idxs or target <= 0:
        return []
    blockers = sorted(blockers)
    top = sorted((i for i in idxs if cards[i]["imp"] >= 5), key=lambda i: _gold_key(i))
    mandatory = {idxs[0], idxs[-1]} | set(top[:max(0, int(target / (2 * S_MIN)))])

    def caps_for(sel):
        out = []
        for k, i in enumerate(sel):
            s = cards[i]["start"]
            nxt = cards[sel[k + 1]]["start"] if k + 1 < len(sel) else src_dur
            j = bisect.bisect_right(blockers, s + 1e-6)
            if j < len(blockers):
                nxt = min(nxt, blockers[j])
            limit = max(S_MIN, cards[i]["dur"] * 1.10 + 0.5)      # never show much footage beyond the card
            out.append(_clamp(min(nxt - s, limit), 0.5, S_MAX))
        return out

    def base(i):
        return _clamp(cards[i]["dur"], S_MIN, S_MAX) * IMP_W[cards[i]["imp"]]

    def slots_at(sel, caps, scale):
        return [min(c, max(min(S_MIN, c), base(i) * scale)) for i, c in zip(sel, caps)]

    sel = set(mandatory)
    s = sorted(sel)
    caps = caps_for(s)
    order = sorted((i for i in idxs if i not in sel), key=lambda i: (-cards[i]["imp"], _gold_key(i)))
    pos = 0
    while sum(slots_at(s, caps, 1.0)) < target and pos < len(order):
        sel.add(order[pos])
        pos += 1
        s = sorted(sel)
        caps = caps_for(s)
    # too many mandatory/selected cards for the budget -> drop the least important
    while sum(min(S_MIN, c) for c in caps) > target:
        removable = [i for i in s if i not in mandatory]
        if not removable:
            break
        victim = min(removable, key=lambda i: (cards[i]["imp"], -_gold_key(i)))
        sel.discard(victim)
        s = sorted(sel)
        caps = caps_for(s)
    lo, hi = 0.05, 20.0
    if sum(slots_at(s, caps, hi)) <= target:
        scale = hi
    elif sum(slots_at(s, caps, lo)) >= target:
        scale = lo
    else:
        for _ in range(50):
            mid = (lo + hi) / 2
            if sum(slots_at(s, caps, mid)) < target:
                lo = mid
            else:
                hi = mid
        scale = (lo + hi) / 2
    slots = slots_at(s, caps, scale)
    beats = []
    for i, cap, slot in zip(s, caps, slots):
        beats.append(_new_beat(counter, "narr", cards[i], cards[i]["start"], cap, round(slot, 2)))
    return beats


def _unit_beats(card, counter, units_all, src_dur, kind="dlg"):
    out = []
    ustarts = [u["start"] for u in units_all]
    for u in card["units"]:
        slot = (u["end"] - u["start"]) + 0.16
        start = max(0.0, u["start"] - 0.08)
        j = bisect.bisect_right(ustarts, u["start"] + 1e-6)
        nxt = units_all[j]["start"] if j < len(units_all) else src_dur
        cap = max(slot, min(slot * 1.15, nxt - start))
        out.append(_new_beat(counter, kind, card, start, cap, round(slot, 2), source_text=u["text"][:500]))
    return out


def plan_dialogue_only(cards, units_all, target, src_dur, counter):
    dcards = [c for c in cards if c["units"]]
    if not dcards:
        raise PipelineError("Dialogue only mode အတွက် စကားပြော (Whisper transcript) မတွေ့ပါ")

    def total(c):
        return sum((u["end"] - u["start"]) + 0.16 for u in c["units"])

    mand = {dcards[0]["idx"], dcards[-1]["idx"]}
    chosen = set(mand)
    acc = sum(total(cards[i]) for i in chosen)
    for c in sorted(dcards, key=lambda c: (-c["imp"], _gold_key(c["idx"]))):
        if c["idx"] in chosen:
            continue
        if acc >= target * (1 - TOLERANCE):
            break
        t = total(c)
        if acc + t <= target * (1 + TOLERANCE):
            chosen.add(c["idx"])
            acc += t
    beats = []
    for c in dcards:
        if c["idx"] in chosen:
            beats.extend(_unit_beats(c, counter, units_all, src_dur))
    return beats


def plan_mixed(cards, units_all, target, src_dur, counter, share=DIALOGUE_SHARE, exclude_extra=()):
    """Narrator beats + dialogue blocks (optional narrator lead-in)."""
    exclude_extra = set(exclude_extra)
    cand = [c for c in cards if c["units"] and c["imp"] >= 3 and c["idx"] not in exclude_extra]
    cand.sort(key=lambda c: (-c["imp"], _gold_key(c["idx"])))
    picked, fixed = [], 0.0
    for c in cand:
        if fixed >= share * target:
            break
        # alternate: never two neighbouring cards as dialogue blocks (narration must bridge them)
        if any(abs(c["idx"] - p["idx"]) <= 1 for p in picked):
            continue
        t = sum((u["end"] - u["start"]) + 0.16 for u in c["units"]) + LEAD_MAX * 0.8
        picked.append(c)
        fixed += t
    # keep at least 30% of the budget for narration
    while picked and fixed > 0.70 * target:
        v = min(picked, key=lambda c: (c["imp"], -_gold_key(c["idx"])))
        picked.remove(v)
        fixed -= sum((u["end"] - u["start"]) + 0.16 for u in v["units"]) + LEAD_MAX * 0.8
    picked.sort(key=lambda c: c["idx"])
    dlg_beats, blockers, fixed_real = [], [], 0.0
    prev_unit_end = 0.0
    for c in picked:
        ub = _unit_beats(c, counter, units_all, src_dur)
        if not ub:
            continue
        first = ub[0]
        lead_start = max(c["start"], first["src_start"] - LEAD_MAX, prev_unit_end)
        room = first["src_start"] - 0.1 - lead_start
        block_start = first["src_start"]
        group = []
        if room >= LEAD_MIN:
            slot = _clamp(room, LEAD_MIN, LEAD_MAX)
            lead = _new_beat(counter, "narr", c, lead_start, room, round(slot, 2), lead=True)
            group.append(lead)
            block_start = lead_start
        group.extend(ub)
        dlg_beats.extend(group)
        blockers.append(block_start)
        fixed_real += sum(b["slot"] for b in group)
        prev_unit_end = ub[-1]["src_start"] + ub[-1]["cap"]
    # narrator cards must not start inside a dialogue block
    spans = []
    for c in picked:
        us = [b for b in dlg_beats if b["card_idx"] == c["idx"]]
        if us:
            spans.append((min(b["src_start"] for b in us), max(b["src_start"] + b["cap"] for b in us)))
    exclude = {c["idx"] for c in picked} | exclude_extra
    for c in cards:
        if any(a - 1e-6 <= c["start"] < b for a, b in spans):
            exclude.add(c["idx"])
    narr = plan_narration(cards, max(0.0, target - fixed_real), src_dur, counter,
                          blockers=blockers, exclude=exclude)
    return narr + dlg_beats


def order_beats(beats):
    beats.sort(key=lambda b: (b["src_start"], b["id"]))
    for i, b in enumerate(beats):
        b["pos"] = "opening" if i == 0 else ("closing" if i == len(beats) - 1 else "middle")
    return beats


# --------------------------------------------------------------------------
# 3. WRITE
# --------------------------------------------------------------------------
WRITE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "beats": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
            "id": {"type": "STRING"},
            "lines": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
                "role": {"type": "STRING"}, "text": {"type": "STRING"}}, "required": ["role", "text"]}},
        }, "required": ["id", "lines"]}},
        "names": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
            "source": {"type": "STRING"}, "burmese": {"type": "STRING"}}, "required": ["source", "burmese"]}},
    },
    "required": ["beats"],
}

_RULES = {
    STYLE_RECAP: """MODE: STORY RECAP (narrator only).
- Every beat has exactly ONE line with role "NARRATOR": a spoken storyteller paragraph in natural, lively Burmese (not textbook, not literal translation).
- Tell the STORY: who wants what, what blocks them, what happens next and why it matters. Keep cause -> effect between neighbouring beats; never repeat a point.
- position "opening": start with a strong hook (a shocking fact or a question). position "closing": finish with a satisfying closing line.
- Never say "scene", "clip", "video", "timestamp" or describe the camera.""",
    STYLE_RECAP_DIALOGUE: """MODE: RECAP + CHARACTER DIALOGUE.
- kind "narr": ONE line, role "NARRATOR": spoken storyteller narration in natural Burmese. If lead_in is true it is a short set-up that leads into the conversation that follows.
- kind "dlg": the characters speak. Write 1-3 short lines. Each line's role is the EXACT name of the character who says it, chosen from speakers_in_scene (if unsure use "MALE" or "FEMALE"). Base the lines on source_dialogue: convey its real meaning as natural spoken Burmese, never invent plot facts.
- A dlg line is ONLY what the character says out loud, in first or second person ("ငါ...", "မင်း..."). NEVER describe what someone does, feels or thinks in the third person inside a dlg line ("သူမ ... လုပ်တယ်") - that is narration and belongs in a NARRATOR beat. Never write lines like "she did not explain".
- Two different characters must never speak inside one line; use one line per speaker, in the order they speak.
- Keep cause -> effect between neighbouring beats; never repeat a point. opening: strong hook. closing: closing line.""",
    STYLE_DIALOGUE_ONLY: """MODE: DIALOGUE ONLY (dubbing, one single voice).
- Every beat is ONE spoken source utterance. Write exactly ONE line with role "NARRATOR" containing that utterance as natural, colloquial spoken Burmese, as an actor would say it. Keep the meaning and emotional tone of source_dialogue.
- No narration, no explanations, no speaker names, no quotation marks. Do not add information that is not in source_dialogue.
- If source_dialogue is empty or pure noise, return an empty "lines" array for that beat.
- The line MUST fit max_chars. Dubbing rule: translate the meaning, not every word. Drop filler, repeated words and interjections; prefer a short complete natural sentence over a long one. NEVER output a cut-off word or half sentence - every line must be a complete, natural utterance.""",
}


def _write_prompt(style, items, style_text, bible, names, prev_tail, lang, rules=None, context=""):
    return f"""You write the {lang} spoken script of a movie-recap video. Each beat is a timed slot: the spoken text must fit its seconds.

{rules or _RULES[style]}

{context}

SPOKEN BURMESE (applies to every line): write the way a person talks out loud, never book style.
- Use colloquial endings: ...တယ်၊ ...ပါတယ်၊ ...ခဲ့တယ်၊ ...တော့၊ ...ပြီး၊ ...ရဲ့၊ ...ထဲမှာ.
- NEVER use written/literary forms: ...သည်၊ ...ခဲ့သည်၊ ၏၊ တွင်၊ ၍၊ ၎င်း၊ သို့.
- Short, punchy, fluent sentences - like a fast, confident storyteller talking to friends.
NO REPETITION: each beat must move the story forward with NEW information taken from its own what_happens / source_dialogue. If continues_previous_beat is true the previous beat already narrated the start of the same moment: say what happens NEXT, never restate or paraphrase it. Two beats must never say the same thing in different words.

LENGTH IS A HARD CONSTRAINT. For each beat, the total speakable characters over all its lines (spaces and punctuation not counted) MUST be between min_chars and max_chars. A line that is too long gets cut by the editor.
Write ONLY {lang}. Keep character names consistent: use NAME MAP entries exactly as given and add new names to "names" (source name -> {lang} spelling).

{('WRITING STYLE GUIDE:\n' + style_text) if style_text else ''}

CHARACTERS: {json.dumps(bible, ensure_ascii=False)}
NAME MAP: {json.dumps(names, ensure_ascii=False)}
PREVIOUS NARRATION (continue naturally from it): {json.dumps(prev_tail, ensure_ascii=False)}

BEATS (JSON): {json.dumps(items, ensure_ascii=False)}

Return JSON {{"beats":[{{"id":..., "lines":[{{"role":"NARRATOR|MALE|FEMALE","text":...}}]}}], "names":[{{"source":...,"burmese":...}}]}} with one entry per input beat id, in the same order."""


def _parse_beats(obj):
    if isinstance(obj, dict):
        arr = obj.get("beats")
        names = obj.get("names") or []
    elif isinstance(obj, list):
        arr, names = obj, []
    else:
        return {}, []
    out = {}
    for it in arr or []:
        if isinstance(it, dict) and it.get("id") is not None:
            out[str(it["id"])] = it.get("lines") or []
    return out, names


_LOCAL_SPOKEN = [
    (re.compile(r"သည်(?!း)(?=\s*[။၊]|\s*$)"), "တယ်"),    # sentence-final verb ending  (…ခဲ့သည်။ -> …ခဲ့တယ်။)
    (re.compile(r"သည့်"), "တဲ့"),
    (re.compile(r"၏"), "ရဲ့"),
    (re.compile(r"တွင်(?!း)"), "မှာ"),
    (re.compile(r"၍"), "ပြီး"),
    (re.compile(r"၎င်း"), "အဲဒီ"),
]


def localize_spoken(text):
    """Cheap, deterministic book-style -> spoken Burmese. Whatever is left is handled by one LLM pass."""
    out = str(text or "")
    for rx, rep_ in _LOCAL_SPOKEN:
        out = rx.sub(rep_, out)
    return out


def _plain(text):
    return _PUNCT_RE.sub("", str(text or ""))


def _grams(t, n=3):
    return {t[i:i + n] for i in range(max(1, len(t) - n + 1))}


def similarity(a, b):
    A, B = _grams(_plain(a)), _grams(_plain(b))
    return len(A & B) / max(1, len(A | B))


def beat_text(b):
    return " ".join(t for _r, t in b["lines"])


class Writer:
    def __init__(self, deps, style, cps, style_text, bible, lang="Burmese", roster=(), rules=None,
                 role_names=None):
        self.deps, self.style, self.cps = deps, style, cps
        self.rules, self.context, self.tail_seed = rules, "", []
        self.role_names = list(role_names or [])
        self.style_text, self.bible, self.lang = (style_text or "")[:2500], bible, lang
        self.roster = list(roster)
        self.names = {}
        self.banned = set()           # cards whose beat was dropped as a repeat
        self.prev_of = {}
        self.stats = {"dedupe_rewritten": 0, "dedupe_dropped": 0, "register_fixed": 0, "local_fixed": 0}

    def _role(self, b, raw):
        r = str(raw or "").strip()
        u = r.upper()
        if self.role_names:
            for nm in self.role_names:
                if nm.upper() == u:
                    return nm
            return self.role_names[0]
        if self.style != STYLE_RECAP_DIALOGUE or b["kind"] == "narr" or u == "NARRATOR":
            return "NARRATOR"
        for nm in self.roster:
            if nm.lower() == r.lower():
                return nm
        if u in ("MALE", "FEMALE"):
            return u
        for nm in self.roster:
            if nm.lower() in r.lower():
                return nm
        g = (b.get("genders") or ["male"])[0]
        return "FEMALE" if g == "female" else "MALE"

    def budget(self, b):
        c = self.cps.get("NARRATOR", 11.0)
        if self.role_names:
            vals = [self.cps.get(r, c) for r in self.role_names]
            c = sum(vals) / len(vals)
        if b["kind"] == "dlg" and self.style == STYLE_RECAP_DIALOGUE:
            c = (self.cps.get("MALE", c) + self.cps.get("FEMALE", c)) / 2.0
        hi = max(4, int(c * b["slot"] * 0.97))
        lo = max(2, int(c * b["slot"] * (0.35 if b["kind"] == "dlg" else 0.82)))
        return lo, hi

    def _item(self, b, rows):
        lo, hi = self.budget(b)
        it = {"id": b["id"], "kind": b["kind"], "seconds": round(b["slot"], 1),
              "min_chars": lo, "max_chars": hi, "what_happens": b["summary"], "position": b["pos"]}
        if b["lead"]:
            it["lead_in"] = True
        txt = b["source_text"] or transcript_text(rows, b["src_start"], b["src_start"] + b["slot"], 260)
        if txt:
            it["source_dialogue"] = txt
        if b["kind"] == "dlg":
            sp = [n for n in b["speakers"] if n in self.roster]
            if sp:
                it["speakers_in_scene"] = sp
            elif b["genders"]:
                it["speaker_genders"] = b["genders"]
        pv = self.prev_of.get(b["id"])
        if b["kind"] == "narr" and pv is not None and pv["kind"] == "narr" and pv["group"] == b["group"]:
            it["continues_previous_beat"] = True
        return it

    def write(self, beats, rows, label="Write"):
        """Fill b['lines'] for every beat; beats that could not be written get lines=[]."""
        todo = [b for b in beats]
        for k, b in enumerate(todo):
            self.prev_of[b["id"]] = todo[k - 1] if k else None
        tail = list(self.tail_seed)
        for i in range(0, len(todo), BATCH):
            batch = todo[i:i + BATCH]
            pending = list(batch)
            for attempt in (1, 2):
                if not pending:
                    break
                items = [self._item(b, rows) for b in pending]
                prompt = _write_prompt(self.style, items, self.style_text, self.bible,
                                       self.names, tail[-3:], self.lang, rules=self.rules, context=self.context)
                try:
                    obj = self.deps.llm_text(prompt, WRITE_SCHEMA, f"{label} {i // BATCH + 1}")
                except self.deps.fatal:
                    raise
                except Exception as e:  # noqa: BLE001
                    self.deps.log(f"⚠️ {label} batch {i // BATCH + 1} attempt {attempt}: {e}")
                    continue
                got, names = _parse_beats(obj)
                for n in names:
                    if isinstance(n, dict) and n.get("source") and n.get("burmese"):
                        self.names[str(n["source"])] = str(n["burmese"])
                still = []
                for b in pending:
                    lines = self._clean(b, got.get(b["id"]))
                    if lines is None:
                        still.append(b)
                    else:
                        b["lines"] = lines
                pending = still
            for b in pending:
                b["lines"] = []
            for b in batch:
                if b["lines"]:
                    tail.append(" ".join(t for _r, t in b["lines"]))
            self.deps.progress(min(i + BATCH, len(todo)), len(todo), "write")

    def _clean(self, b, raw):
        """-> list[(role,text)] ; [] for an intentionally empty beat; None when missing/invalid."""
        if raw is None:
            return None
        out = []
        for ln in raw:
            if not isinstance(ln, dict):
                continue
            t = re.sub(r"\s+", " ", str(ln.get("text", "")).replace("\u200b", "")).strip()
            t = re.sub(r"^\[[A-Za-z_]+\]\s*", "", t)
            if self.lang == "Burmese":
                t2 = localize_spoken(t)
                if t2 != t:
                    self.stats["local_fixed"] += 1
                    t = t2
            if speak_len(t) >= 1:
                out.append((self._role(b, ln.get("role")), t))
        if not out and self.style != STYLE_DIALOGUE_ONLY:
            return None
        return out

    # -- quality passes --------------------------------------------------
    def _rewrite(self, items, label):
        """items: [(beat, instruction_dict)] -> returns {id: lines}"""
        out = {}
        for i in range(0, len(items), BATCH):
            chunk = items[i:i + BATCH]
            payload = []
            for b, extra in chunk:
                lo, hi = self.budget(b)
                it = {"id": b["id"], "kind": b["kind"], "seconds": round(b["slot"], 1), "min_chars": lo,
                      "max_chars": hi, "what_happens": b["summary"],
                      "current": [{"role": r, "text": t} for r, t in b["lines"]]}
                it.update(extra)
                payload.append(it)
            prompt = f"""You are fixing the spoken {self.lang} script of a movie-recap video. Rewrite each beat according to its "fix" instruction.
Keep the same roles, keep every beat inside min_chars..max_chars speakable characters (spaces and punctuation not counted), keep NAME MAP spellings: {json.dumps(self.names, ensure_ascii=False)}.
Write natural SPOKEN Burmese only (colloquial endings ...တယ်/...ပါတယ်/...ခဲ့တယ်/...တော့; never ...သည်, ၏, တွင်, ၍, ၎င်း, သို့).
BEATS: {json.dumps(payload, ensure_ascii=False)}
Return JSON {{"beats":[{{"id":..., "lines":[{{"role":..., "text":...}}]}}]}} with every id."""
            try:
                obj = self.deps.llm_light(prompt, WRITE_SCHEMA, f"{label} {i // BATCH + 1}")
            except self.deps.fatal:
                raise
            except Exception as e:  # noqa: BLE001
                self.deps.log(f"⚠️ {label}: {e}")
                continue
            got, _ = _parse_beats(obj)
            for b, _x in chunk:
                new = self._clean(b, got.get(b["id"]))
                if new:
                    out[b["id"]] = new
        return out

    def dedupe(self, beats, droppable=None):
        """Detect narration beats that repeat a neighbour; rewrite once, else drop."""
        def narr(seq):
            return [b for b in seq if b["lines"] and b["kind"] == "narr"]

        def worst(seq, k):
            b = seq[k]
            return max([similarity(beat_text(b), beat_text(seq[j])) for j in range(max(0, k - 2), k)] or [0.0])

        for rnd in (1, 2):
            seq = narr(order_beats(list(beats)))
            bad = [seq[k] for k in range(1, len(seq))
                   if worst(seq, k) >= DUP_SIM and (droppable is None or seq[k]["id"] in droppable)]
            if not bad:
                return
            if rnd == 1:
                req = []
                for b in bad:
                    k = seq.index(b)
                    prev = [beat_text(seq[j]) for j in range(max(0, k - 2), k)]
                    req.append((b, {"fix": "This beat repeats the previous narration. Rewrite it so it says ONLY what is new "
                                           "in its own what_happens, continuing the story forward. Do not reuse the "
                                           "wording or the facts of previous_narration.", "previous_narration": prev}))
                got = self._rewrite(req, "Dedupe")
                for b in bad:
                    if b["id"] in got:
                        b["lines"] = got[b["id"]]
                        self.stats["dedupe_rewritten"] += 1
            else:
                total = max(1, len(seq))
                limit = max(2, int(total * 0.20))       # never gut the script because of a detector false alarm
                bad.sort(key=lambda b: -worst(seq, seq.index(b)))
                for b in bad[:limit]:
                    b["lines"] = []
                    self.banned.add(b["card"])
                    self.stats["dedupe_dropped"] += 1

    def fix_register(self, beats):
        """Rewrite beats that contain book-style Burmese into spoken Burmese."""
        flagged = [b for b in beats if b["lines"] and any(LITERARY_RE.search(t) for _r, t in b["lines"])]
        if not flagged:
            return
        got = self._rewrite([(b, {"fix": "Rewrite in colloquial spoken Burmese: replace written forms "
                                         "(သည်, ၏, တွင်, ၍, ၎င်း, သို့) with spoken ones (တယ်/ပါတယ်, ရဲ့, ထဲမှာ, ပြီး, သူ/ဒီ, ကို). "
                                         "Keep the meaning."}) for b in flagged], "Spoken")
        for b in flagged:
            new = got.get(b["id"])
            if new and not any(LITERARY_RE.search(t) for _r, t in new):
                b["lines"] = new
                self.stats["register_fixed"] += 1

    def polish(self, beats, droppable=None):
        self.dedupe(beats, droppable)
        self.fix_register([b for b in beats if droppable is None or b["id"] in droppable])

    def shorten(self, rows_beats, label="Shorten"):
        """rows_beats: [(beat, target_chars)] -> rewrites b['lines'] in place when valid."""
        if not rows_beats:
            return
        for i in range(0, len(rows_beats), BATCH):
            chunk = rows_beats[i:i + BATCH]
            items = [{"id": b["id"], "kind": b["kind"], "seconds": round(b["slot"], 1),
                      "max_chars": int(tc), "current": [{"role": r, "text": t} for r, t in b["lines"]]}
                     for b, tc in chunk]
            prompt = f"""Shorten the spoken {self.lang} lines of each beat so that the total speakable characters (spaces and punctuation not counted) is AT MOST max_chars. Keep the same roles, meaning, tone and the most important facts; remove detail, not the point. Keep NAME MAP spellings: {json.dumps(self.names, ensure_ascii=False)}
BEATS: {json.dumps(items, ensure_ascii=False)}
Return JSON {{"beats":[{{"id":..., "lines":[{{"role":..., "text":...}}]}}]}} with every id."""
            try:
                obj = self.deps.llm_light(prompt, WRITE_SCHEMA, f"{label} {i // BATCH + 1}")
            except self.deps.fatal:
                raise
            except Exception as e:  # noqa: BLE001
                self.deps.log(f"⚠️ {label}: {e}")
                continue
            got, _ = _parse_beats(obj)
            for b, tc in chunk:
                new = self._clean(b, got.get(b["id"]))
                if new:
                    cur = sum(speak_len(t) for _r, t in b["lines"])
                    nl = sum(speak_len(t) for _r, t in new)
                    if nl < cur:
                        b["lines"] = new


# --------------------------------------------------------------------------
# 4. SYNTHESIZE + FIT
# --------------------------------------------------------------------------
def assign_voices(bible, cards, base, pool, preset=None):
    """Give each recurring speaker their own voice.  Returns (voices, roster)."""
    from collections import Counter
    count = Counter()
    for c in cards:
        if c.get("units"):
            for nm in c.get("speakers", []):
                count[nm] += 1
    info = {str(b["name"]).lower(): b for b in bible}
    voices, roster = dict(base), []
    used = {base["NARRATOR"], base["MALE"], base["FEMALE"]}
    first = {"male": False, "female": False}
    pool = {g: list((pool or {}).get(g, [])) for g in ("male", "female")}
    preset = dict(preset or {})
    for nm, vid in preset.items():                 # voices fixed in earlier parts stay fixed
        used.add(vid)
    for nm, _n in count.most_common(8):
        ent = info.get(nm.lower(), {})
        g = ent.get("gender")
        if nm in preset:
            voices[nm] = preset[nm]
            if g in first:
                first[g] = True
            roster.append(nm)
            continue
        if g not in ("male", "female"):
            continue
        age = ent.get("age", "adult")
        if not first[g]:
            voices[nm] = base["MALE" if g == "male" else "FEMALE"]
            first[g] = True
        else:
            cand = [v for v in pool[g] if v not in used]
            if age in ("child", "teen", "elder"):
                pref = [v for v in cand if age in v]
            else:
                pref = [v for v in cand if not any(k in v for k in ("child", "teen", "elder"))]
            pick = (pref or cand or [base["MALE" if g == "male" else "FEMALE"]])[0]
            voices[nm] = pick
            used.add(pick)
        roster.append(nm)
    return voices, roster


def calibrate(deps, voices, work_dir, lang_sample=None):
    """Measure speakable-characters-per-second for each distinct voice."""
    sample = lang_sample or ("ဒီနေ့ မနက်ပိုင်းမှာ သူက မြို့ထဲကို အလျင်စလို ထွက်သွားပြီး ခရီးသွားလာနေတဲ့ လူတွေကြားထဲမှာ "
                             "သူ့ရဲ့ မိတ်ဆွေဟောင်းကို လိုက်ရှာနေပါတယ်။ ဒါပေမယ့် သူရှာတွေ့တဲ့အခါမှာ အားလုံးက ပြောင်းလဲသွားပြီးပါပြီ။")
    cps, seen = {}, {}
    for role, vid in voices.items():
        if vid in seen:
            cps[role] = seen[vid]
            continue
        p = os.path.join(work_dir, f"cal_{len(seen)}.mp3")
        try:
            deps.tts(sample, vid, p)
            d = deps.probe(p)
            v = speak_len(sample) / d if d > 0.3 else 11.0
        except deps.fatal:
            raise
        except Exception as e:  # noqa: BLE001
            deps.log(f"⚠️ voice calibration failed for {vid}: {e} — default speed သုံးပါမည်")
            v = 11.0
        v = _clamp(v, 4.0, 30.0)
        seen[vid] = v
        cps[role] = v
        try:
            os.remove(p)
        except OSError:
            pass
    return cps


def _ensure_pause(deps, work_dir):
    p = os.path.join(work_dir, "pause.mp3")
    if not os.path.exists(p):
        deps.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono", "-t", str(PAUSE_SEC),
                  "-c:a", "libmp3lame", "-b:a", "64k", p], "dv2 pause")
    return p


def synth_beat(deps, b, voices, work_dir):
    """TTS every line of the beat, join with short pauses.  Sets path/dur/words."""
    paths, words, cursor = [], [], 0.0
    pause = _ensure_pause(deps, work_dir) if len(b["lines"]) > 1 else None
    for li, (role, text) in enumerate(b["lines"]):
        lp = os.path.join(work_dir, f"{b['id']}_l{li}.mp3")
        w = deps.tts(text, voices.get(role, voices["NARRATOR"]), lp) or []
        d = deps.probe(lp)
        for x in w:
            words.append({**x, "start": x["start"] + cursor, "end": x["end"] + cursor})
        paths.append(lp)
        cursor += d
        if pause and li < len(b["lines"]) - 1:
            paths.append(pause)
            cursor += PAUSE_SEC
    if len(paths) == 1:
        out = paths[0]
    else:
        out = os.path.join(work_dir, f"{b['id']}_joined.mp3")
        lst = os.path.join(work_dir, f"{b['id']}_list.txt")
        with open(lst, "w", encoding="utf-8") as f:
            for p in paths:
                f.write("file '" + p.replace("'", "'\\''") + "'\n")
        deps.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c:a", "libmp3lame",
                  "-b:a", "128k", out], f"dv2 join {b['id']}")
    b["path"], b["words"] = out, words
    b["dur"] = deps.probe(out)


def _allowed(b):
    """Longest audio (seconds) accepted for a beat without rewriting it."""
    return max(b["slot"], min(b["cap"], b["slot"] * MAX_OVER))


def hard_trim(lines, budget):
    """Last resort: keep whole lines / whole phrases that fit in `budget` speakable chars.

    Cuts only at a phrase boundary (space, ၊ or ။) - never inside a word.  Returns None when not even
    the first phrase fits; the caller must then drop the beat instead of playing a word fragment.
    """
    lines = [(r, t) for r, t in lines]
    while len(lines) > 1 and sum(speak_len(t) for _r, t in lines) > budget:
        lines.pop()
    role, text = lines[-1]
    if speak_len(text) > budget:
        out = ""
        for tk in re.split(r"(?<=[\u104a\u104b\s])", text):
            if speak_len(out + tk) > budget:
                break
            out += tk
        out = out.strip()
        if not out:
            return None
        lines[-1] = (role, out)
    return lines


def _tempo(deps, b, factor, work_dir):
    out = os.path.join(work_dir, f"{b['id']}_t.mp3")
    deps.run(["ffmpeg", "-y", "-i", b["path"], "-af", f"atempo={factor:.4f}", "-c:a", "libmp3lame",
              "-b:a", "128k", out], f"dv2 tempo {b['id']}")
    b["path"] = out
    b["words"] = [{**w, "start": w["start"] / factor, "end": w["end"] / factor} for w in b["words"]]
    b["dur"] = deps.probe(out)


def fit(deps, writer, beats, voices, work_dir, rounds=2, max_tempo=MAX_TEMPO):
    """Synthesize + rewrite-too-long + last-resort tempo.  Drops unwritable beats."""
    stats = {"rewritten": 0, "tempo": 0, "dropped": 0, "overlong_after": 0, "hard_trimmed": 0, "dropped_unfittable": 0}
    todo = [b for b in beats if b["lines"]]
    for b in beats:
        if not b["lines"]:
            stats["dropped"] += 1
    for rnd in range(rounds + 1):
        for k, b in enumerate(todo):
            if b["path"] is None:
                try:
                    synth_beat(deps, b, voices, work_dir)
                except deps.fatal:
                    raise
                except Exception as e:  # noqa: BLE001
                    deps.log(f"⚠️ TTS {b['id']} မအောင်မြင်ပါ: {e}")
                    b["lines"] = []
            if k % 10 == 0:
                deps.progress(k, len(todo), "tts")
        todo = [b for b in todo if b["lines"] and b["path"]]
        over = [b for b in todo if b["dur"] > _allowed(b) + 0.02]
        if not over or rnd == rounds:
            break
        req = []
        for b in over:
            chars = sum(speak_len(t) for _r, t in b["lines"])
            tc = max(3, int(chars * (b["slot"] * 0.96) / b["dur"]))
            req.append((b, tc))
        deps.log(f"✂️ Beat {len(over)} ခုက slot ထက်ရှည်နေလို့ စာသားချုံ့ပါမည် (round {rnd + 1})")
        writer.shorten(req, f"Shorten r{rnd + 1}")
        for b in over:
            b["path"] = None   # re-synthesize with the (possibly) shorter text
            stats["rewritten"] += 1
    for b in todo:
        lim = _allowed(b)
        if b["dur"] > lim + 0.02:
            f = b["dur"] / lim
            if f > max_tempo:
                # the writer did not shorten enough: cut at a phrase boundary, or drop the beat
                chars = sum(speak_len(t) for _r, t in b["lines"])
                budget = max(3, int(chars * (lim * max_tempo * 0.97) / b["dur"]))
                trimmed = hard_trim(b["lines"], budget)
                if trimmed is None:
                    b["lines"], b["path"] = [], None
                    stats["dropped_unfittable"] += 1
                    deps.log(f"⚠️ {b['id']}: စာသားက slot ထဲ မဆန့်လို့ ဖယ်ထုတ်ပါသည် (စကားလုံးအပိုင်းအစ မဖြစ်စေရန်)")
                    continue
                b["lines"] = trimmed
                synth_beat(deps, b, voices, work_dir)
                stats["hard_trimmed"] += 1
                f = b["dur"] / lim
            if f > 1.0 + 0.02 / lim:
                _tempo(deps, b, min(max_tempo, f), work_dir)
                stats["tempo"] += 1
        if b["dur"] > max(lim, b["cap"]) + 0.05:
            stats["overlong_after"] += 1
    kept = [b for b in beats if b["lines"] and b["path"]]
    stats["dropped"] = len(beats) - len(kept)
    return kept, stats


# --------------------------------------------------------------------------
# 5. LENGTH CONTROL
# --------------------------------------------------------------------------
def total_dur(beats):
    return sum(b["dur"] for b in beats)


def trim_to_target(beats, target):
    """Drop least-important beats (never first/last) while total > target*(1+TOL)."""
    dropped = 0
    beats = order_beats(list(beats))
    while total_dur(beats) > target * (1 + TOLERANCE) and len(beats) > 3:
        cands = beats[1:-1]
        v = min(cands, key=lambda b: (b["imp"] + (0.5 if b["kind"] == "dlg" else 0.0), -b["dur"]))
        beats.remove(v)
        dropped += 1
    return order_beats(beats), dropped


def find_topup(beats, cards, src_dur, need, counter, banned=(), allow=None):
    """New narrator beats in the unused footage gaps, most important cards first."""
    used = sorted((b["src_start"], b["src_start"] + max(b["dur"], b["slot"])) for b in beats)
    starts = [u[0] for u in used]
    taken = {b["card_idx"] for b in beats}
    cand = [c for c in cards if c["idx"] not in taken and c["id"] not in banned
            and (allow is None or c["idx"] in allow)]
    cand.sort(key=lambda c: (-c["imp"], _gold_key(c["idx"])))
    new, got = [], 0.0
    reserved = []
    for c in cand:
        if got >= need:
            break
        a = c["start"]
        j = bisect.bisect_right(starts, a + 1e-6)
        prev_end = used[j - 1][1] if j > 0 else 0.0
        nxt = used[j][0] if j < len(used) else src_dur
        for ra, rb in reserved:
            if ra <= a < rb:
                prev_end = max(prev_end, rb)
            if a < ra < nxt:
                nxt = min(nxt, ra)
        a = max(a, prev_end)
        room = min(S_MAX, nxt - a, max(S_MIN, c["dur"] * 1.10 + 0.5))
        if room < S_MIN * 0.7:
            continue
        slot = round(min(room, max(S_MIN, min(S_MAX, c["dur"] * IMP_W[c["imp"]]))), 2)
        nb = _new_beat(counter, "narr", c, a, room, slot)
        new.append(nb)
        reserved.append((a, a + room))
        got += slot
    return new


def find_topup_dialogue(beats, cards, units_all, need, src_dur, counter, banned=()):
    """Dialogue-only: add whole unused dialogue cards, most important first."""
    taken = {b["card_idx"] for b in beats}
    cand = [c for c in cards if c["units"] and c["idx"] not in taken]
    cand.sort(key=lambda c: (-c["imp"], _gold_key(c["idx"])))
    new, got = [], 0.0
    for c in cand:
        if got >= need:
            break
        t = sum((u["end"] - u["start"]) + 0.16 for u in c["units"]) * 0.9
        if got > 0 and got + t > need * 1.3:
            continue
        new.extend(_unit_beats(c, counter, units_all, src_dur))
        got += t
    return new


# --------------------------------------------------------------------------
# 6. OUTPUT
# --------------------------------------------------------------------------
def build_scenes(beats, src_dur):
    """Chronological, non-overlapping scenes for the render engine."""
    beats = order_beats(list(beats))
    scenes, prev_end, dropped = [], 0.0, 0
    for k, b in enumerate(beats):
        nxt = beats[k + 1]["src_start"] if k + 1 < len(beats) else src_dur
        fs = max(b["src_start"], prev_end)
        room = min(nxt, src_dur) - fs
        if room < 0.25:
            dropped += 1
            continue
        fe = min(fs + b["dur"] + 0.15, fs + room, src_dur)
        prev_end = fe
        tagged = " ".join(f"[{r}] {t}" for r, t in b["lines"])
        scenes.append({
            "start_time": fmt_time(fs), "end_time": fmt_time(fe),
            "script": tagged, "narration": tagged, "raw_script": b["source_text"] or b["summary"],
            "_tts_path": b["path"], "_tts_words": b["words"],
            "_beat": b["id"], "_kind": b["kind"], "_dur": round(b["dur"], 3),
            "_drift": round(max(0.0, fe - b["card_end"]), 2) if b["kind"] == "narr" else 0.0,
        })
    return scenes, dropped


def prepare_evidence(video, src_dur, deps, cache_root, work_dir, limit_sec=None, want_units=True):
    """Transcript + cuts + scene cards for the first `limit_sec` seconds (whole movie when None).

    Everything is cached on disk, so a pilot run and the later full run share the work.
    Returns dict(cards, bible, transcript, units, cdir, warnings).
    """
    os.makedirs(work_dir, exist_ok=True)
    key = cache_key(video)
    cdir = os.path.join(cache_root, key)
    os.makedirs(cdir, exist_ok=True)
    limit = float(limit_sec) if limit_sec and limit_sec < src_dur else float(src_dur)
    partial = limit < src_dur - 1.0
    tfull = os.path.join(cdir, "transcript.json")
    transcript = _jload(tfull)
    if transcript is None and partial:
        tpart = os.path.join(cdir, f"transcript_part_{int(limit)}.json")
        transcript = _jload(tpart)
        if transcript is None:
            deps.log("🎙️ Pilot အတွက် ရှေ့ပိုင်း transcript ကိုသာ ထုတ်နေပါသည်...")
            clip = os.path.join(work_dir, "pilot_audio.wav")
            deps.run(["ffmpeg", "-y", "-i", video, "-t", f"{limit:.2f}", "-vn", "-ac", "1", "-ar", "16000", clip], "pilot audio")
            transcript = [r for r in (deps.transcribe(clip) or []) if r.get("text")]
            _jsave(tpart, transcript)
    elif transcript is None:
        deps.log("🎙️ Local Whisper transcript ထုတ်နေပါသည် (တစ်ကြိမ်သာ၊ နောက်ပိုင်း cache သုံးပါမည်)...")
        transcript = [r for r in (deps.transcribe(video) or []) if r.get("text")]
        if transcript:
            _jsave(tfull, transcript)
    transcript = [r for r in transcript if r["start"] < limit]
    cpath = os.path.join(cdir, "cuts.json")
    cuts = _jload(cpath)
    if cuts is None:
        cuts = detect_cuts(video)
        _jsave(cpath, cuts)
    cards, bible, warns = analyze(video, limit, deps, os.path.join(cdir, "a2"), transcript, cuts, work_dir)
    units = make_units(transcript)
    units = merge_units(units) if want_units else [u for u in units if u["end"] - u["start"] >= 0.8]
    attach_units(cards, units)
    return {"cards": cards, "bible": bible, "transcript": transcript, "units": units, "cdir": cdir,
            "warnings": warns, "limit": limit}


def prepare_range(video, src_dur, deps, cache_root, work_dir, start_sec, end_sec, memory_chars=None, recent=None):
    """Transcript + cards for ONLY [start_sec, end_sec) of the movie (a project 'part').

    Analysis windows are cached on the 5-minute grid, so neighbouring parts share the window that straddles
    their boundary and never pay for it twice.  A full-movie transcript is reused if a pre-scan already made one.
    """
    os.makedirs(work_dir, exist_ok=True)
    cdir = os.path.join(cache_root, cache_key(video))
    os.makedirs(cdir, exist_ok=True)
    full = _jload(os.path.join(cdir, "transcript.json"))
    if full is None:
        rpath = os.path.join(cdir, f"transcript_{int(start_sec)}_{int(end_sec)}.json")
        full = _jload(rpath)
        if full is None:
            deps.log(f"🎙️ အပိုင်း ({start_sec / 60:.0f}-{end_sec / 60:.0f} မိနစ်) transcript ထုတ်နေပါသည်...")
            clip = os.path.join(work_dir, "part_audio.wav")
            deps.run(["ffmpeg", "-y", "-ss", f"{start_sec:.2f}", "-t", f"{end_sec - start_sec:.2f}", "-i", video,
                      "-vn", "-ac", "1", "-ar", "16000", clip], "part audio")
            rows = [r for r in (deps.transcribe(clip) or []) if r.get("text")]
            full = [{**r, "start": r["start"] + start_sec, "end": r["end"] + start_sec} for r in rows]
            _jsave(rpath, full)
    transcript = [r for r in full if r["end"] > start_sec and r["start"] < end_sec]
    cpath = os.path.join(cdir, "cuts.json")
    cuts = _jload(cpath)
    if cuts is None:
        cuts = detect_cuts(video)
        _jsave(cpath, cuts)
    w0 = int(start_sec // WINDOW_SEC)
    w1 = int(math.ceil(end_sec / WINDOW_SEC - 1e-9))
    cards, chars, warns = analyze(video, src_dur, deps, os.path.join(cdir, "a2"), transcript, cuts, work_dir,
                                  w_range=(w0, w1), known_init=memory_chars, recent_init=recent)
    cards = [c for c in cards if start_sec - 0.5 <= c["start"] < end_sec - 0.5]
    for i, c in enumerate(cards):
        c["idx"] = i
    if not cards:
        raise PipelineError(f"no scene cards for {start_sec / 60:.0f}-{end_sec / 60:.0f} min " + "; ".join(warns[:2]))
    units = merge_units(make_units(transcript))
    attach_units(cards, units)
    return {"cards": cards, "bible": chars, "transcript": transcript, "units": units, "cdir": cdir, "warnings": warns,
            "limit": end_sec}


def find_break(video, nominal, src_dur, deps=None, back=90.0, fwd=60.0):
    """A natural cut point near `nominal`: the middle of the longest quiet gap in [nominal-back, nominal+fwd].

    Falls back to `nominal` itself.  Never returns past the end of the movie.
    """
    nominal = min(max(0.0, nominal), src_dur)
    if nominal >= src_dur - 5.0:
        return src_dur
    a = max(0.0, nominal - back)
    b = min(src_dur, nominal + fwd)
    try:
        p = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-ss", f"{a:.2f}", "-t", f"{b - a:.2f}", "-i", video,
                            "-vn", "-af", "silencedetect=noise=-32dB:d=0.7", "-f", "null", "-"],
                           capture_output=True, timeout=600)
        err = p.stderr.decode("utf-8", errors="replace")
        starts = [float(x) for x in re.findall(r"silence_start: ([0-9.]+)", err)]
        ends = [float(x) for x in re.findall(r"silence_end: ([0-9.]+)", err)]
        best, bd = None, 0.0
        for st, en in zip(starts, ends):
            if en - st > bd and en - st >= 0.7:
                best, bd = a + (st + en) / 2.0, en - st
        if best is not None:
            return round(min(max(best, a), b), 2)
    except Exception:
        pass
    return round(nominal, 2)


def run_pipeline(video, src_dur, style, target_sec, voices, deps, cache_root, work_dir,
                 style_text="", lang="Burmese", voice_pool=None):
    """Run the whole pipeline.  Returns {"scenes": [...], "qa": {...}}."""
    if style not in STYLES:
        raise PipelineError(f"unknown dialogue style {style!r}")
    if target_sec <= 5:
        raise PipelineError("target length is too short")
    os.makedirs(work_dir, exist_ok=True)
    key = cache_key(video)
    cdir = os.path.join(cache_root, key)
    os.makedirs(cdir, exist_ok=True)
    warnings = []

    # --- evidence -----------------------------------------------------------
    tpath = os.path.join(cdir, "transcript.json")
    transcript = _jload(tpath)
    if transcript is None:
        deps.log("🎙️ Local Whisper transcript ထုတ်နေပါသည် (တစ်ကြိမ်သာ၊ နောက်ပိုင်း cache သုံးပါမည်)...")
        transcript = [r for r in (deps.transcribe(video) or []) if r.get("text")]
        if transcript:
            _jsave(tpath, transcript)
    if not transcript and style == STYLE_DIALOGUE_ONLY:
        raise PipelineError("Whisper transcript မရပါ — Dialogue only mode အတွက် faster-whisper လိုအပ်ပါသည်")
    eff_style = style
    if not transcript and style == STYLE_RECAP_DIALOGUE:
        warnings.append("no transcript: recap_dialogue fell back to narrator-only recap")
        deps.log("⚠️ Transcript မရလို့ Recap+Dialogue ကို Recap (narrator) အဖြစ် ပြောင်းလုပ်ပါမည်။")
        eff_style = STYLE_RECAP
    cpath = os.path.join(cdir, "cuts.json")
    cuts = _jload(cpath)
    if cuts is None:
        cuts = detect_cuts(video)
        _jsave(cpath, cuts)

    cards, bible, w_analysis = analyze(video, src_dur, deps, os.path.join(cdir, "a2"), transcript, cuts, work_dir)
    warnings.extend(w_analysis)
    units = make_units(transcript)
    if eff_style in (STYLE_RECAP_DIALOGUE, STYLE_DIALOGUE_ONLY):
        units = merge_units(units)
    else:
        units = [u for u in units if u["end"] - u["start"] >= 0.8]
    attach_units(cards, units)

    # --- voices / calibration ----------------------------------------------
    roster = []
    if eff_style == STYLE_RECAP_DIALOGUE:
        use_voices, roster = assign_voices(bible, cards, dict(voices), voice_pool)
    else:
        use_voices = {"NARRATOR": voices["NARRATOR"]}
    cps = calibrate(deps, use_voices, work_dir)
    deps.log("🔊 Voice speed: " + ", ".join(f"{k}={v:.1f} chars/s" for k, v in cps.items()))

    # --- plan ---------------------------------------------------------------
    counter = [1]
    if eff_style == STYLE_RECAP:
        beats = plan_narration(cards, target_sec, src_dur, counter)
    elif eff_style == STYLE_RECAP_DIALOGUE:
        beats = plan_mixed(cards, units, target_sec, src_dur, counter)
    else:
        beats = plan_dialogue_only(cards, units, target_sec / 0.92, src_dur, counter)
    if not beats:
        raise PipelineError("planner selected no beats")
    beats = order_beats(beats)
    deps.log(f"📐 Plan: {len(beats)} beats, budget {sum(b['slot'] for b in beats):.0f}s / target {target_sec:.0f}s")

    # --- write + fit --------------------------------------------------------
    writer = Writer(deps, eff_style, cps, style_text, bible, lang, roster=roster)
    writer.write(beats, transcript)
    writer.polish(beats)
    dub = eff_style == STYLE_DIALOGUE_ONLY
    kept, stats = fit(deps, writer, beats, use_voices, work_dir, rounds=3 if dub else 2,
                      max_tempo=1.22 if dub else MAX_TEMPO)
    if not kept:
        raise PipelineError("no beat could be written and synthesized")

    # --- global length control ---------------------------------------------
    topups = 0
    for rnd in range(2):
        tot = total_dur(kept)
        if tot >= target_sec * (1 - TOLERANCE):
            break
        if eff_style == STYLE_DIALOGUE_ONLY:
            extra = find_topup_dialogue(kept, cards, units, target_sec - tot, src_dur, counter, writer.banned)
        else:
            extra = find_topup(kept, cards, src_dur, target_sec - tot, counter, writer.banned)
        if not extra:
            break
        deps.log(f"➕ Length {tot:.0f}s < target {target_sec:.0f}s — beat {len(extra)} ခု ထပ်ထည့်ပါမည်")
        writer.write(extra, transcript, label="TopUp")
        writer.polish(order_beats(kept + extra), droppable={b["id"] for b in extra})
        add, st2 = fit(deps, writer, extra, use_voices, work_dir, rounds=3 if dub else 2,
                       max_tempo=1.22 if dub else MAX_TEMPO)
        for k in ("rewritten", "tempo", "dropped", "overlong_after", "hard_trimmed", "dropped_unfittable"):
            stats[k] += st2[k]
        kept = order_beats(kept + add)
        topups += len(add)
    kept, trimmed = trim_to_target(kept, target_sec)

    scenes, dropped_ovl = build_scenes(kept, src_dur)
    if not scenes:
        raise PipelineError("no scenes remained after timeline validation")
    audio_total = sum(s["_dur"] for s in scenes)
    qa = {
        "dv2_style": eff_style, "dv2_requested_style": style,
        "dv2_target_sec": round(target_sec, 2), "dv2_audio_sec": round(audio_total, 2),
        "dv2_length_ratio": round(audio_total / target_sec, 4),
        "dv2_beats": len(scenes), "dv2_cards": len(cards), "dv2_cps": {k: round(v, 2) for k, v in cps.items()},
        "dv2_rewritten": stats["rewritten"], "dv2_tempo_beats": stats["tempo"],
        "dv2_dropped_unwritable": stats["dropped"], "dv2_topup_beats": topups,
        "dv2_trimmed_beats": trimmed, "dv2_dropped_overlap": dropped_ovl,
        "dv2_overlong_after_fit": stats["overlong_after"], "dv2_hard_trimmed": stats["hard_trimmed"], "dv2_dropped_unfittable": stats["dropped_unfittable"],
        "dv2_dialogue_beats": sum(1 for s in scenes if s["_kind"] == "dlg"),
        "dv2_drift_beats": sum(1 for s in scenes if s.get("_drift", 0) > 2.0),
        "dv2_dedupe_rewritten": writer.stats["dedupe_rewritten"], "dv2_dedupe_dropped": writer.stats["dedupe_dropped"],
        "dv2_register_fixed": writer.stats["register_fixed"],
        "dv2_voices": {k: v for k, v in use_voices.items()},
        "dv2_warnings": warnings,
    }
    seq = [s for s in scenes if s["_kind"] == "narr"]
    left = sum(1 for k in range(1, len(seq))
               if similarity(re.sub(r"\[[^\]]+\]", "", seq[k]["script"]), re.sub(r"\[[^\]]+\]", "", seq[k - 1]["script"])) >= DUP_SIM)
    qa["dv2_repeats_left"] = left
    if left:
        qa["dv2_warnings"].append(f"{left} narration beat(s) may still repeat their neighbour - please review the script")
    if abs(audio_total - target_sec) / target_sec > 0.10:
        qa["dv2_warnings"].append(
            f"final audio {audio_total:.0f}s differs from target {target_sec:.0f}s by more than 10% "
            "(not enough supported story/dialogue material, or Gemini returned short text)")
    return {"scenes": scenes, "qa": qa}
