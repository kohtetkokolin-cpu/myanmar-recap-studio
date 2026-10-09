"""story_engine - long-form, audio-first story retelling on top of dialogue_v2.

Flow
  1. prepare_evidence()   (dialogue_v2)  transcript + shot cuts + scene cards, cached on disk
  2. build_bible()        one text call: logline, themes, turning points, hidden details, characters
  3. build_outline()      one text call: N chapters (hook / tension / end question / scene list)
  4. produce_chapter()    per chapter: plan -> write -> polish -> TTS fit -> top-up/trim -> footage windows
  5. make_meta()          YouTube titles / description / tags (+ chapter list built in Python)

Ten selectable styles share this engine; they differ only in persona rules, voices and structure.
Pilot mode makes a ~3 minute sample from the first part of the movie (cheap, shares the cache).
"""
import hashlib
import json
import math
import os
import re

import dialogue_v2 as dv

CHAPTER_MINUTES = 6.0
PILOT_SEC = 180.0

AUDIO_FIRST = """AUDIO-FIRST RULES (the listener may NOT be looking at the screen):
- Say clearly who does what and where. Never leave 'he/she/they' ambiguous for more than one sentence; re-name people regularly.
- Put the one visual that matters into words in a few words. Never say 'as you can see' or 'look here'.
- Every beat moves the story forward with something NEW: an action, decision, reveal, consequence or feeling.
- Vary rhythm: mostly short punchy sentences, sometimes one longer. Use vivid but sparing sensory detail and emotion.
- Plant questions and pay them off. The LAST beat of a chapter must leave an open question that pulls the listener into the next chapter.
- Never invent facts, motives or events that the evidence (what_happens / source_dialogue / story bible) does not support."""

STYLES = {
    "fast_recap": {
        "label": "၁။ မြန်ဆန်သော Recap (တတ်တတ်ကျွကျွ)", "kind": "narr", "voice": "narrator_fast_f",
        "desc": "ဖော်ဖော်ရွေရွေ၊ မြန်မြန်ဆန်ဆန်၊ hook နဲ့ စပြီး တင်းမာမှုတက်သွားတဲ့ recap",
        "rules": "PERSONA: a quick, confident friend telling a gripping story out loud. Punchy short sentences, spoken connectors, a hook at the start of each chapter, rising stakes."},
    "pov_first_person": {
        "label": "၂။ ဇာတ်ကောင်ရဲ့ ပါးစပ်က ပြောပြ (ပထမပုဂ္ဂိုလ်)", "kind": "narr", "voice": "narrator_calm", "pov": True,
        "desc": "ဇာတ်ကောင်တစ်ယောက်က 'ငါ' နဲ့ ဇာတ်လမ်းတစ်ခုလုံးကို ပြန်ပြောပြ",
        "rules": "PERSONA: narrate the WHOLE story in FIRST PERSON as {pov} (use ငါ / ကျွန်တော်). Show their fear, doubts, guilt and hope. For events {pov} did not witness, say how {pov} learned of them later (e.g. 'နောက်မှ ငါသိရတာက...'). Never leave the first-person voice."},
    "radio_drama": {
        "label": "၃။ ရေဒီယိုဒရာမာ (Narrator + ဇာတ်ကောင်အသံ)", "kind": "mixed", "voice": "narrator_calm",
        "desc": "Narrator နဲ့ ဇာတ်ကောင်တစ်ယောက်စီ အသံခွဲပြီး စကားပြောတွေ ရောထည့်",
        "rules": dv._RULES[dv.STYLE_RECAP_DIALOGUE]},
    "case_file": {
        "label": "၄။ အမှုဖိုင် / စုံထောက်", "kind": "narr", "voice": "narrator_calm",
        "desc": "ဇာတ်လမ်းကို အမှုတစ်ခု ပြန်ဖွင့်စစ်ဆေးသလို ပြောပြ",
        "rules": "PERSONA: a calm investigator reopening a case file. Lay out the timeline, evidence, suspicions and the first thing that did not add up. Dry, precise, quietly tense. Facts only from the evidence."},
    "campfire_horror": {
        "label": "၅။ မီးပုံဘေး ကြောက်စရာပုံပြင်", "kind": "narr", "voice": "narrator_calm",
        "desc": "တိုးတိုး၊ ဖြည်းဖြည်း၊ ကြောက်စိတ်တက်လာအောင် အချက်တွေကို ထိန်းထားပြီး ပြောပြ",
        "rules": "PERSONA: a hushed storyteller by a campfire. Slow-building dread: short sentences, withheld information, sound/cold/darkness details, silence implied by line breaks. Do not reveal the scariest fact before its moment."},
    "hidden_layer": {
        "label": "၆။ ဇာတ်ကြောင်း + သင်မမြင်လိုက်တဲ့အချက်", "kind": "narr", "voice": "narrator_fast_m", "interlude": True,
        "desc": "ဇာတ်ကြောင်းပြီးတိုင်း ကြိုတင်ညွှန်းထားတဲ့ ပုန်းထားအချက်ကို ဖော်ပြ",
        "rules": "PERSONA: a clear storyteller who also points out what viewers missed. Tell the story normally. In the CLOSING beat of a chapter add one short remark starting with 'သင်မမြင်လိုက်တဲ့အချက်က ...' using ONLY the hidden_details given in the chapter context (foreshadowing, symbols, details worth a rewatch). If no hidden_details are given, skip the remark."},
    "two_hosts": {
        "label": "၇။ နှစ်ယောက်ပြောပြ (ဦးစီး + တုံ့ပြန်သူ)", "kind": "narr", "voice": "narrator_fast_f", "roles": ["HOST_A", "HOST_B"],
        "desc": "သူငယ်ချင်းနှစ်ယောက် ဇာတ်ကားကို တစ်ယောက်ကိုတစ်ယောက် ပြန်ပြောပြသလို",
        "rules": "PERSONA: two friends retelling the movie to each other. HOST_A tells the story; HOST_B reacts, asks the question the listener is thinking, guesses, jokes lightly. Each beat has 1-3 short lines with role HOST_A or HOST_B, alternating naturally. Facts only from the evidence."},
    "reverse_open": {
        "label": "၈။ အဆုံးကနေ စပြီး ပြန်ဖွင့်", "kind": "narr", "voice": "narrator_fast_m", "reverse": True,
        "desc": "အပြင်းထန်ဆုံးအချိန်ကို အရင်ပြပြီး 'ဘယ်လိုရောက်လာတာလဲ' ဆိုပြီး အစကနေ ပြန်ပြောပြ",
        "rules": "PERSONA: a gripping storyteller. The first chapter is a COLD OPEN: the most shocking moment in 2-4 tense lines, ending with a bridge such as 'ဒါပေမယ့် ဒီဟာကို နားလည်ချင်ရင် အစကနေ စရမယ်'. The later chapters then tell the story in order."},
    "bedtime_calm": {
        "label": "၉။ အိပ်ရာဝင်ခါနီး ငြိမ်သက်စွာ", "kind": "narr", "voice": "narrator_calm",
        "desc": "နှေးနှေး၊ နွေးထွေးပြီး စိတ်ငြိမ်စေတဲ့ လေသံ",
        "rules": "PERSONA: a warm, slow, soothing voice for listening in bed. Short calm sentences, gentle wording, no shouting or graphic violence (soften it), reassuring tone. The plot must still move forward; end chapters peacefully."},
    "documentary_analysis": {
        "label": "၁၀။ Documentary + ဝေဖန်သုံးသပ်", "kind": "narr", "voice": "narrator_calm", "interlude": True,
        "desc": "ဇာတ်လမ်းပြောပြပြီး ဘာကြောင့်အရေးကြီးလဲ၊ အဓိကအချက်ကဘာလဲ ကိုပါ ရှင်းပြ",
        "rules": "PERSONA: a documentary narrator who is also a thoughtful critic. Tell what happens, then explain why it matters: motives, themes, cause and effect. Clearly label opinions as opinions (e.g. 'ကျွန်တော့်အမြင်အရတော့...'). Base analysis only on the story bible themes and evidence."},
}


def list_styles():
    return [{"id": k, "label": v["label"], "desc": v["desc"], "kind": v["kind"], "voice": v["voice"],
             "pov": bool(v.get("pov"))} for k, v in STYLES.items()]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _mmss(t):
    t = int(max(0, t))
    return f"{t // 60}:{t % 60:02d}"


def card_lines(cards, limit=900):
    use = cards
    if len(cards) > limit:
        use = [c for c in cards if c["imp"] >= 3]
        if len(use) > limit:
            step = len(use) / float(limit)
            use = [use[int(i * step)] for i in range(limit)]
    return "\n".join(f"{c['idx']}|{_mmss(c['start'])}|{c['imp']}|{c['summary'][:140]}" for c in use)


def _hash(*parts):
    return hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:10]


def _call(deps, prompt, schema, label):
    try:
        return deps.llm_text(prompt, schema, label)
    except deps.fatal:
        raise
    except Exception as e:  # noqa: BLE001
        deps.log(f"⚠️ {label} မအောင်မြင်ပါ: {e}")
        return None


# --------------------------------------------------------------------------
# 2. STORY BIBLE
# --------------------------------------------------------------------------
BIBLE_SCHEMA = {"type": "OBJECT", "properties": {
    "logline": {"type": "STRING"}, "genre": {"type": "STRING"}, "tone": {"type": "STRING"},
    "ending": {"type": "STRING"},
    "themes": {"type": "ARRAY", "items": {"type": "STRING"}},
    "characters": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
        "name": {"type": "STRING"}, "want": {"type": "STRING"}, "arc": {"type": "STRING"}}, "required": ["name"]}},
    "turning_points": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
        "idx": {"type": "INTEGER"}, "why": {"type": "STRING"}}, "required": ["idx"]}},
    "hidden_details": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
        "idx": {"type": "INTEGER"}, "detail": {"type": "STRING"}}, "required": ["idx", "detail"]}},
    "climax_idx": {"type": "ARRAY", "items": {"type": "INTEGER"}},
}, "required": ["logline"]}


def build_bible(deps, cards, cache_path):
    cached = dv._jload(cache_path)
    if cached:
        return cached
    prompt = f"""You are the story editor of a long-form, audio-first movie retelling channel.
Below is the scene list of the movie, in order. Each line: idx|time|importance(1-5)|what happens.
Build the STORY BIBLE as JSON. Use ONLY what the scene list supports; never invent plot.
- logline: one sentence. genre, tone: short English phrases. ending: how the story ends (1-2 sentences).
- themes: 2-4 short themes. characters: up to 8 main characters with what they WANT and how they CHANGE (arc).
- turning_points: 5-12 scenes that change the direction of the story (idx + why).
- hidden_details: up to 10 foreshadowing details / symbols / easily missed facts that pay off later (idx = the scene where it is visible, detail = what to notice).
- climax_idx: 1-4 scene idx of the climax. All idx must come from the list.

SCENES:
{card_lines(cards)}"""
    obj = _call(deps, prompt, BIBLE_SCHEMA, "Story bible")
    bible = obj if isinstance(obj, dict) and obj.get("logline") else None
    if bible is None:
        top = sorted(cards, key=lambda c: -c["imp"])[:3]
        bible = {"logline": cards[0]["summary"] if cards else "", "genre": "", "tone": "", "ending": "",
                 "themes": [], "characters": [], "turning_points": [{"idx": c["idx"], "why": c["summary"]} for c in top],
                 "hidden_details": [], "climax_idx": [top[0]["idx"]] if top else []}
        bible["_fallback"] = True
    else:
        dv._jsave(cache_path, bible)
    return bible


# --------------------------------------------------------------------------
# 3. OUTLINE
# --------------------------------------------------------------------------
OUTLINE_SCHEMA = {"type": "OBJECT", "properties": {
    "chapters": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
        "title": {"type": "STRING"}, "purpose": {"type": "STRING"}, "hook": {"type": "STRING"},
        "end_question": {"type": "STRING"}, "tension": {"type": "INTEGER"}, "share": {"type": "NUMBER"},
        "card_idx": {"type": "ARRAY", "items": {"type": "INTEGER"}}}, "required": ["title", "card_idx"]}},
    "cold_open": {"type": "OBJECT", "properties": {
        "title": {"type": "STRING"}, "hook": {"type": "STRING"},
        "card_idx": {"type": "ARRAY", "items": {"type": "INTEGER"}}}},
}, "required": ["chapters"]}


def _fallback_chapters(cards, n):
    """Deterministic split into n contiguous chapters of equal importance*duration weight."""
    w = [dv.IMP_W[c["imp"]] * c["dur"] for c in cards]
    total = sum(w) or 1.0
    out, acc, k, cur = [], 0.0, 1, []
    for c, x in zip(cards, w):
        cur.append(c["idx"])
        acc += x
        if k < n and acc >= total * k / n:
            out.append({"title": f"အခန်း {k}", "card_idx": cur}); cur = []; k += 1
    if cur:
        out.append({"title": f"အခန်း {k}", "card_idx": cur})
    return out


def build_outline(deps, bible, cards, style_id, n_chapters, pov, cache_path):
    spec = STYLES[style_id]
    cached = dv._jload(cache_path)
    if cached:
        return cached
    raw = None
    if n_chapters > 1 or spec.get("reverse"):
        cold = ""
        if spec.get("reverse"):
            cold = ('\nAlso return "cold_open": {"title","hook","card_idx"} = the 1-4 MOST shocking / climactic scenes to show first '
                    '(they may come from late in the movie).')
        prompt = f"""You are the story editor planning an audio-first retelling of a movie in exactly {n_chapters} chapters.
STYLE: {spec['label']} - {spec['desc']}{(' POV character: ' + pov) if pov else ''}
STORY BIBLE: {json.dumps({k: v for k, v in bible.items() if not k.startswith('_')}, ensure_ascii=False)[:6000]}
Return JSON {{"chapters":[...]}} with EXACTLY {n_chapters} chapters in story order (chronological):
- title: a short, intriguing chapter title in Burmese. purpose: what this chapter achieves (English). hook: the opening hook idea. end_question: the open question that ends the chapter.
- tension: 1-5. share: relative length weight 0.5-2 (bigger = more important chapter).
- card_idx: the 4-12 most important scene idx that chapter should use, increasing. Each chapter must cover a contiguous part of the story; no idx may appear in two chapters. All idx must come from the list.{cold}
SCENES (idx|time|importance|what happens):
{card_lines(cards)}"""
        raw = _call(deps, prompt, OUTLINE_SCHEMA, "Outline")
    return _finish_outline(raw, cards, n_chapters, spec, cache_path, bible)


def _finish_outline(raw, cards, n, spec, cache_path, bible):
    valid = {c["idx"] for c in cards}
    chapters, used = [], set()
    src = (raw or {}).get("chapters") if isinstance(raw, dict) else None
    for ch in src or []:
        if not isinstance(ch, dict):
            continue
        ids = []
        for x in ch.get("card_idx") or []:
            try:
                x = int(x)
            except (TypeError, ValueError):
                continue
            if x in valid and x not in used and x not in ids:
                ids.append(x)
        if ids:
            used.update(ids)
            chapters.append({"title": str(ch.get("title") or "").strip()[:80], "purpose": str(ch.get("purpose") or "")[:200],
                             "hook": str(ch.get("hook") or "")[:200], "end_question": str(ch.get("end_question") or "")[:200],
                             "tension": int(ch.get("tension") or 3) if str(ch.get("tension") or "3").isdigit() else 3,
                             "share": ch.get("share"), "card_idx": sorted(ids)})
    fallback = len(chapters) < max(1, int(n * 0.6))
    if fallback:
        chapters = [dict(title=c["title"], purpose="", hook="", end_question="", tension=3, share=None,
                         card_idx=c["card_idx"]) for c in _fallback_chapters(cards, n)]
    chapters.sort(key=lambda c: c["card_idx"][0])
    for i, ch in enumerate(chapters):
        if not ch["title"]:
            ch["title"] = f"အခန်း {i + 1}"
    # chapters tile the movie: pool of chapter k = [its first idx, next chapter's first idx - 1]
    lo_all = cards[0]["idx"]
    for i, ch in enumerate(chapters):
        lo = lo_all if i == 0 else ch["card_idx"][0]
        hi = (chapters[i + 1]["card_idx"][0] - 1) if i + 1 < len(chapters) else cards[-1]["idx"]
        ch["pool"] = [lo, max(lo, hi)]
    # length shares
    w = []
    for ch in chapters:
        try:
            s = float(ch.get("share"))
            s = dv._clamp(s, 0.5, 2.0)
        except (TypeError, ValueError):
            s = None
        if s is None or fallback:
            pool = [c for c in cards if ch["pool"][0] <= c["idx"] <= ch["pool"][1]]
            s = max(0.3, sum(dv.IMP_W[c["imp"]] * c["dur"] for c in pool))
        w.append(s)
    tot = sum(w) or 1.0
    for ch, x in zip(chapters, w):
        ch["share"] = x / tot
    hidden = {}
    for h in (bible.get("hidden_details") or []):
        try:
            hidden.setdefault(int(h.get("idx")), str(h.get("detail", ""))[:200])
        except (TypeError, ValueError):
            pass
    for ch in chapters:
        ch["hidden"] = [d for i, d in hidden.items() if ch["pool"][0] <= i <= ch["pool"][1]][:3]
    cold = None
    if spec.get("reverse") and isinstance(raw, dict) and isinstance(raw.get("cold_open"), dict):
        ids = []
        for x in raw["cold_open"].get("card_idx") or []:
            try:
                x = int(x)
            except (TypeError, ValueError):
                continue
            if x in valid and x not in ids:
                ids.append(x)
        if ids:
            cold = {"title": str(raw["cold_open"].get("title") or "Cold open")[:80],
                    "hook": str(raw["cold_open"].get("hook") or "")[:200], "card_idx": sorted(ids)}
    if spec.get("reverse") and cold is None:
        clim = [i for i in (bible.get("climax_idx") or []) if i in valid][:3]
        if not clim:
            clim = [max(cards, key=lambda c: c["imp"])["idx"]]
        cold = {"title": "Cold open", "hook": "", "card_idx": sorted(clim)}
    out = {"chapters": chapters, "cold_open": cold, "fallback": fallback and (n > 1 or bool(spec.get("reverse")))}
    if not (bible.get("_fallback") or not cache_path):
        dv._jsave(cache_path, out)
    return out


# --------------------------------------------------------------------------
# 4. CHAPTER PRODUCTION
# --------------------------------------------------------------------------
def _context(bible, ch, pov, interlude, idx, total, is_cold=False):
    parts = [f"STORY: {bible.get('logline', '')} | genre: {bible.get('genre', '')} | tone: {bible.get('tone', '')} | "
             f"themes: {', '.join(bible.get('themes') or [])}"]
    chars = [f"{c.get('name')}: wants {c.get('want', '')}; arc {c.get('arc', '')}" for c in (bible.get("characters") or [])[:6]]
    if chars:
        parts.append("MAIN CHARACTERS: " + " || ".join(chars))
    if is_cold:
        parts.append(f"CHAPTER {idx + 1}/{total} = COLD OPEN. Hook idea: {ch.get('hook', '')}")
    else:
        parts.append(f"CHAPTER {idx + 1}/{total}: \"{ch['title']}\". Purpose: {ch.get('purpose', '')}. Opening hook idea: {ch.get('hook', '')}. "
                     f"Tension level {ch.get('tension', 3)}/5. The chapter's LAST beat must end on this open question: {ch.get('end_question', '')}")
    if interlude and ch.get("hidden"):
        parts.append("hidden_details for this chapter (use only these): " + " | ".join(ch["hidden"]))
    return "\n".join(parts)


def _sig(b):
    return f"{round(b['src_start'], 1)}|{round(b['slot'], 1)}|{b['kind']}"


def produce_chapter(deps, writer, ev, spec, bible, ch, ch_index, n_total, budget_sec, voices, work_dir, counter,
                    pov, lines_cache, is_cold=False, cold_ids=None, store=None, lines_path=None):
    cards, units, transcript, src_dur = ev["cards"], ev["units"], ev["transcript"], ev["src_dur"]
    if is_cold:
        pool = set(cold_ids)
    else:
        pool = {c["idx"] for c in cards if ch["pool"][0] <= c["idx"] <= ch["pool"][1]}
    others = {c["idx"] for c in cards} - pool
    kind = spec["kind"]
    if kind == "mixed":
        beats = dv.plan_mixed(cards, units, budget_sec, src_dur, counter, exclude_extra=others)
    else:
        beats = dv.plan_narration(cards, budget_sec, src_dur, counter, exclude=others)
    if not beats:
        return [], {"warning": f"chapter {ch_index + 1}: no beats planned"}
    beats = dv.order_beats(beats)
    writer.context = _context(bible, ch, pov, spec.get("interlude"), ch_index, n_total, is_cold)
    sigs = [_sig(b) for b in beats]
    key = f"{ch_index}:{'cold' if is_cold else 'ch'}"
    cached = lines_cache.get(key)
    if cached and cached.get("sigs") == sigs:
        for b, ln in zip(beats, cached["lines"]):
            b["lines"] = [tuple(x) for x in ln]
        deps.log(f"♻️ အခန်း {ch_index + 1}: ရေးပြီးသား script ကို cache ကနေ ပြန်သုံးပါသည်")
    else:
        writer.write(beats, transcript, label=f"Ch{ch_index + 1} write")
        writer.polish(beats)
        lines_cache[key] = {"sigs": sigs, "lines": [[list(x) for x in b["lines"]] for b in beats]}
        if lines_path:
            dv._jsave(lines_path, lines_cache)         # written scripts survive a crash in TTS/render
    csig = _hash(json.dumps(sigs), json.dumps([b["lines"] for b in beats], ensure_ascii=False),
                 json.dumps(sorted(voices.items())), round(budget_sec))
    if store is not None:
        done = store.load(ch_index, csig)
        if done:
            deps.log(f"♻️ အခန်း {ch_index + 1}: render ပြီးသား checkpoint ကို ပြန်သုံးပါသည်")
            return done, {"audio": sum(m["_dur"] for m in done), "budget": budget_sec, "resumed": True}
    kept, stats = dv.fit(deps, writer, beats, voices, work_dir)
    allow = pool
    for _rnd in range(2):
        tot = dv.total_dur(kept)
        if not kept or tot >= budget_sec * (1 - dv.TOLERANCE):
            break
        extra = dv.find_topup(kept, cards, src_dur, budget_sec - tot, counter, writer.banned, allow=allow)
        if not extra:
            break
        writer.write(extra, transcript, label=f"Ch{ch_index + 1} topup")
        writer.polish(dv.order_beats(kept + extra), droppable={b["id"] for b in extra})
        add, st2 = dv.fit(deps, writer, extra, voices, work_dir)
        for k in st2:
            stats[k] = stats.get(k, 0) + st2[k]
        kept = dv.order_beats(kept + add)
    if not kept:
        return [], {"warning": f"chapter {ch_index + 1}: nothing could be written"}
    kept, trimmed = dv.trim_to_target(kept, budget_sec)
    scenes, _dropped = dv.build_scenes(kept, src_dur)
    for s in scenes:
        s["_chapter"] = ch_index
    if store is not None:
        scenes = store.save(ch_index, csig, scenes)      # renders + checkpoints this chapter right now
    stats["trimmed"] = trimmed
    stats["audio"] = sum(s["_dur"] for s in scenes)
    stats["budget"] = budget_sec
    return scenes, stats


# --------------------------------------------------------------------------
# 5. METADATA
# --------------------------------------------------------------------------
META_SCHEMA = {"type": "OBJECT", "properties": {
    "titles": {"type": "ARRAY", "items": {"type": "STRING"}},
    "description": {"type": "STRING"},
    "tags": {"type": "ARRAY", "items": {"type": "STRING"}}}, "required": ["titles", "description"]}


def make_meta(deps, bible, chapters, style_label):
    prompt = f"""Write YouTube metadata (in Burmese) for an audio-first movie retelling video.
STYLE: {style_label}
STORY: {json.dumps({k: bible.get(k) for k in ('logline', 'genre', 'tone', 'themes')}, ensure_ascii=False)}
CHAPTERS: {json.dumps([c['title'] for c in chapters], ensure_ascii=False)}
Return JSON: titles = 5 curiosity-driven title options (no clickbait lies, no spoilers of the final twist), description = 3-5 short sentences that say honestly this is a narrated retelling with original commentary (no timestamps), tags = 8-12 short tags."""
    obj = _call(deps, prompt, META_SCHEMA, "YouTube meta")
    if isinstance(obj, dict) and obj.get("titles"):
        return {"titles": [str(t) for t in obj["titles"]][:6], "description": str(obj.get("description", "")),
                "tags": [str(t) for t in (obj.get("tags") or [])][:15]}
    return {"titles": [str(bible.get("logline", ""))[:90]], "description": str(bible.get("logline", "")), "tags": []}


def format_ts(sec):
    sec = int(max(0, sec))
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def chapters_text(chapters, starts):
    """YouTube chapter list: starts at 00:00, every chapter >= 10 s, at least 3 chapters (else '')."""
    keep = []                                   # [(start, title)]
    for ch, t in zip(chapters, starts):
        if not keep:
            keep.append((0.0, ch["title"]))
        elif t - keep[-1][0] < 10.0:
            if len(keep) == 1:                  # a too-short opening chapter: let the next one take 00:00
                keep[0] = (0.0, ch["title"])
            # otherwise this chapter is merged into the previous one
        else:
            keep.append((t, ch["title"]))
    if len(keep) < 3:
        return ""
    return "\n".join(f"{format_ts(t)} {title}" for t, title in keep)


def estimate(src_dur, target_sec, pilot=False):
    """Rough preflight numbers (Gemini calls, chapters) - an estimate, not a measurement."""
    if pilot:
        limit = min(src_dur, max(8 * 60.0, src_dur * 0.12))
        windows = int(math.ceil(limit / dv.WINDOW_SEC))
        return {"windows": windows, "chapters": 1, "calls": windows + 1 + 4 + 1}
    windows = int(math.ceil(src_dur / dv.WINDOW_SEC))
    chapters = int(dv._clamp(round(target_sec / 60.0 / CHAPTER_MINUTES), 2, 20))
    calls = windows + 2 + int(chapters * 3.5) + 1
    return {"windows": windows, "chapters": chapters, "calls": calls}


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
def run_story(video, src_dur, style_id, target_sec, voices, deps, cache_root, work_dir, pilot=False,
              pov="auto", voice_pool=None, store=None):
    if style_id not in STYLES:
        raise dv.PipelineError(f"unknown story style {style_id!r}")
    spec = STYLES[style_id]
    if target_sec <= 30:
        raise dv.PipelineError("target length is too short")
    warnings = []
    limit = None
    if pilot:
        target_sec = min(target_sec, PILOT_SEC)
        limit = min(src_dur, max(8 * 60.0, src_dur * 0.12))
    ev = dv.prepare_evidence(video, src_dur, deps, cache_root, work_dir, limit_sec=limit, want_units=True)
    ev["src_dur"] = src_dur
    warnings.extend(ev["warnings"])
    cards, bible_chars = ev["cards"], ev["bible"]
    cdir = os.path.join(ev["cdir"], "story")
    os.makedirs(cdir, exist_ok=True)
    scope = f"p{int(ev['limit'])}" if pilot else "full"

    kind = spec["kind"]
    if kind == "mixed" and not ev["transcript"]:
        warnings.append("no transcript: radio drama fell back to narrator-only")
        kind = "narr"
    spec = dict(spec, kind=kind)

    bible = build_bible(deps, cards, os.path.join(cdir, f"bible_{scope}.json"))
    pov_name = ""
    if spec.get("pov"):
        pov_name = pov if pov and pov != "auto" else ""
        if not pov_name:
            cs = bible.get("characters") or []
            pov_name = str(cs[0].get("name")) if cs else (bible_chars[0]["name"] if bible_chars else "the main character")

    target_min = target_sec / 60.0
    n = 1 if pilot else int(dv._clamp(round(target_min / CHAPTER_MINUTES), 2, 20))
    okey = _hash(style_id, n, pov_name, scope)
    outline = build_outline(deps, bible, cards, style_id, n, pov_name, os.path.join(cdir, f"outline_{okey}.json"))
    chapters, cold = outline["chapters"], outline.get("cold_open")
    if outline.get("fallback"):
        warnings.append("outline fell back to an even split (Gemini outline was unusable)")

    # voices
    role_names = spec.get("roles")
    roster = []
    if kind == "mixed":
        use_voices, roster = dv.assign_voices(bible_chars, cards, dict(voices), voice_pool)
    else:
        use_voices = {"NARRATOR": voices["NARRATOR"]}
        if role_names:
            use_voices["HOST_A"] = voices["NARRATOR"]
            use_voices["HOST_B"] = voices["FEMALE"] if voices["NARRATOR"] != voices["FEMALE"] else voices["MALE"]
    cps = dv.calibrate(deps, use_voices, work_dir)
    deps.log("🔊 Voice speed: " + ", ".join(f"{k}={v:.1f} chars/s" for k, v in cps.items()))

    rules = AUDIO_FIRST + "\n" + spec["rules"].replace("{pov}", pov_name or "the narrator")
    writer_style = dv.STYLE_RECAP_DIALOGUE if kind == "mixed" else dv.STYLE_RECAP
    writer = dv.Writer(deps, writer_style, cps, "", bible_chars, "Burmese", roster=roster, rules=rules,
                       role_names=role_names)
    counter = [1]
    lines_cache_path = os.path.join(cdir, f"lines_{okey}_{int(target_sec)}.json")
    lines_cache = dv._jload(lines_cache_path) or {}

    plan = []   # (chapter dict, budget, is_cold, cold_ids)
    cold_budget = 0.0
    if cold:
        by_id = {c["idx"]: c for c in cards}
        material = sum(min(dv.S_MAX, by_id[i]["dur"] * 1.10 + 0.5) for i in cold["card_idx"] if i in by_id)
        cold_budget = float(min(dv._clamp(target_sec * 0.05, 30.0, 100.0), 0.8 * material))
        plan.append((dict(cold, hook=cold.get("hook", "")), cold_budget, True, cold["card_idx"]))
    body = target_sec - cold_budget
    for ch in chapters:
        plan.append((ch, body * ch["share"], False, None))

    all_scenes, ch_info, stats_all = [], [], []
    prev_tail, carry = [], 0.0
    for i, (ch, budget, is_cold, cold_ids) in enumerate(plan):
        budget = budget + dv._clamp(carry, -0.3 * budget, 0.5 * budget)     # make up earlier shortfalls
        deps.log(f"📖 အခန်း {i + 1}/{len(plan)}: {ch['title']} (target {budget:.0f}s)")
        writer.tail_seed = prev_tail[-3:]
        scenes, st = produce_chapter(deps, writer, ev, spec, bible, ch, i, len(plan), budget, use_voices, work_dir,
                                     counter, pov_name, lines_cache, is_cold, cold_ids, store, lines_cache_path)
        dv._jsave(lines_cache_path, lines_cache)
        if not scenes:
            warnings.append(st.get("warning", f"chapter {i + 1} skipped"))
            continue
        first = len(all_scenes)
        carry = budget - st.get("audio", 0.0)
        all_scenes.extend(scenes)
        ch_info.append({"title": ch["title"], "idx": i, "first_scene": first, "budget": round(budget, 1),
                        "audio": round(st.get("audio", 0.0), 1), "cold": is_cold})
        stats_all.append(st)
        prev_tail = [re.sub(r"\[[^\]]+\]\s*", "", s["script"]) for s in scenes[-3:]]
        deps.progress(i + 1, len(plan), "chapters")
    if not all_scenes:
        raise dv.PipelineError("no chapter could be produced")

    audio_total = sum(s["_dur"] for s in all_scenes)
    seq = [s for s in all_scenes if s["_kind"] == "narr"]
    left = sum(1 for k in range(1, len(seq))
               if seq[k]["_chapter"] == seq[k - 1]["_chapter"] and dv.similarity(
                   re.sub(r"\[[^\]]+\]", "", seq[k]["script"]), re.sub(r"\[[^\]]+\]", "", seq[k - 1]["script"])) >= dv.DUP_SIM)
    qa = {
        "story_style": style_id, "story_pilot": bool(pilot), "story_chapters": len(ch_info),
        "story_target_sec": round(target_sec, 1), "story_audio_sec": round(audio_total, 1),
        "story_length_ratio": round(audio_total / target_sec, 4),
        "story_beats": len(all_scenes), "story_cards": len(cards), "story_pov": pov_name,
        "story_repeats_left": left, "story_drift_beats": sum(1 for s in all_scenes if s.get("_drift", 0) > 2.0),
        "story_dedupe_dropped": writer.stats["dedupe_dropped"], "story_register_fixed": writer.stats["register_fixed"], "story_local_fixed": writer.stats["local_fixed"],
        "story_chapters_resumed": sum(1 for x in stats_all if x.get("resumed")),
        "story_dropped_unfittable": sum(s.get("dropped_unfittable", 0) for s in stats_all),
        "story_chapter_report": [{"title": c["title"], "budget": c["budget"], "audio": c["audio"]} for c in ch_info],
        "story_cps": {k: round(v, 2) for k, v in cps.items()}, "story_warnings": warnings,
    }
    if left:
        warnings.append(f"{left} narration beat(s) may still repeat their neighbour - review the script")
    if abs(audio_total - target_sec) / target_sec > 0.10:
        warnings.append(f"final audio {audio_total:.0f}s differs from target {target_sec:.0f}s by more than 10%")
    meta = make_meta(deps, bible, [{"title": c["title"]} for c in ch_info], spec["label"])
    return {"scenes": all_scenes, "qa": qa, "chapters": ch_info, "meta": meta, "bible": bible}


# ==========================================================================
# PROJECT MODE: one movie, many parts, continuous story
# ==========================================================================
OVERVIEW_SCHEMA = {"type": "OBJECT", "properties": {
    "logline": {"type": "STRING"}, "genre": {"type": "STRING"}, "tone": {"type": "STRING"}, "ending": {"type": "STRING"},
    "themes": {"type": "ARRAY", "items": {"type": "STRING"}},
    "characters": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
        "name": {"type": "STRING"}, "gender": {"type": "STRING"}, "age": {"type": "STRING"},
        "want": {"type": "STRING"}, "arc": {"type": "STRING"}}, "required": ["name"]}},
    "timeline": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
        "t": {"type": "NUMBER"}, "summary": {"type": "STRING"}}, "required": ["t", "summary"]}},
    "turning_points": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
        "t": {"type": "NUMBER"}, "why": {"type": "STRING"}}, "required": ["t"]}},
    "hidden_details": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
        "t": {"type": "NUMBER"}, "detail": {"type": "STRING"}}, "required": ["t", "detail"]}},
    "climax": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
        "start": {"type": "NUMBER"}, "end": {"type": "NUMBER"}}, "required": ["start", "end"]}},
}, "required": ["logline"]}


def prescan_overview(deps, video, src_dur, cache_root):
    """Look-ahead for project mode: whole-movie transcript (free, local) + ONE text call -> story overview.

    Gives later parts foreknowledge (foreshadowing, hidden details, climax) without analysing the video itself.
    Returns None when the film has no usable speech.
    """
    cdir = os.path.join(cache_root, dv.cache_key(video))
    os.makedirs(cdir, exist_ok=True)
    opath = os.path.join(cdir, "overview.json")
    cached = dv._jload(opath)
    if cached:
        return cached
    tpath = os.path.join(cdir, "transcript.json")
    transcript = dv._jload(tpath)
    if transcript is None:
        deps.log("🔭 Pre-scan: ဇာတ်ကားတစ်ခုလုံး transcript ထုတ်နေပါသည် (တစ်ကြိမ်သာ၊ API မသုံးပါ)...")
        transcript = [r for r in (deps.transcribe(video) or []) if r.get("text")]
        if transcript:
            dv._jsave(tpath, transcript)
    if not transcript:
        return None
    chunks, step = [], 600.0
    n = int(math.ceil(src_dur / step))
    for k in range(n):
        a, b = k * step, (k + 1) * step
        txt = " ".join(r["text"].strip() for r in transcript if a <= r["start"] < b)
        if len(txt) > 1800:                                   # even sample, keeps beginning and end of the chunk
            pieces = [r["text"].strip() for r in transcript if a <= r["start"] < b]
            keep = max(1, int(len(pieces) * 1800 / len(txt)))
            idx = sorted({int(i * len(pieces) / keep) for i in range(keep)})
            txt = " ".join(pieces[i] for i in idx)[:1800]
        if txt.strip():
            chunks.append(f"[{_mmss(a)}-{_mmss(min(b, src_dur))}] {txt}")
    prompt = f"""You are the story editor of a long-form movie retelling channel. Below is the speech transcript of a whole movie in 10-minute chunks ([m:ss-m:ss] text). Speech may contain recognition errors and never shows silent action.
Return JSON with: logline (1 sentence), genre, tone (short English), ending (how it ends, 1-2 sentences), themes (2-4), characters (up to 8: name, gender male/female/unknown, age child/teen/adult/elder, want, arc),
timeline (one entry per ~5 minutes of movie: t = seconds from the start, summary = one factual sentence), turning_points (t seconds + why), hidden_details (foreshadowing / easily-missed facts that pay off later: t seconds where visible + detail),
climax (1-3 time ranges start/end in seconds). Use only what the transcript supports; never invent plot.
TRANSCRIPT:
""" + "\n".join(chunks)
    obj = _call(deps, prompt, OVERVIEW_SCHEMA, "Pre-scan overview")
    if not (isinstance(obj, dict) and obj.get("logline")):
        return None
    dv._jsave(opath, obj)
    return obj


def _in_range(items, key, a, b):
    out = []
    for it in items or []:
        try:
            t = float(it.get(key))
        except (TypeError, ValueError, AttributeError):
            continue
        if a <= t < b:
            out.append(it)
    return out


def overview_for_part(ov, start, end):
    if not ov:
        return {}
    nxt = [str(x.get("summary", ""))[:160] for x in (ov.get("timeline") or [])
           if isinstance(x, dict) and end <= float(x.get("t", -1)) < end + 900][:3]
    return {"logline": ov.get("logline", ""), "genre": ov.get("genre", ""), "tone": ov.get("tone", ""),
            "themes": ov.get("themes") or [], "ending": ov.get("ending", ""),
            "characters": [{"name": c.get("name"), "want": c.get("want", ""), "arc": c.get("arc", "")}
                           for c in (ov.get("characters") or []) if isinstance(c, dict)][:6],
            "hidden": [str(h.get("detail", ""))[:200] for h in _in_range(ov.get("hidden_details"), "t", start, end)][:3],
            "turning": [str(h.get("why", ""))[:160] for h in _in_range(ov.get("turning_points"), "t", start, end)][:3],
            "upcoming": nxt, "climax": ov.get("climax") or []}


PART_SCHEMA = {"type": "OBJECT", "properties": {
    "title": {"type": "STRING"}, "summary": {"type": "STRING"}}, "required": ["title", "summary"]}


def summarize_part(deps, memory, part_no, scripts_text, last):
    prompt = f"""Update the running memory of a story-retelling project.
PREVIOUS SUMMARY: {memory.get('summary', '') or '(none yet - this is part 1)'}
NEW PART {part_no} NARRATION (Burmese): {scripts_text[:7000]}
Return JSON: title = a short intriguing Burmese chapter title for this part (max 6 words); summary = the CUMULATIVE story so far in at most 140 English words (what happened, who wants what, open questions){' - the story is now finished' if last else ''}."""
    obj = _call(deps, prompt, PART_SCHEMA, f"Part {part_no} summary")
    if isinstance(obj, dict) and obj.get("summary"):
        return str(obj.get("title") or f"Part {part_no}")[:80], str(obj["summary"])[:1400]
    first = re.sub(r"\s+", " ", scripts_text)[:200]
    return f"Part {part_no}", (memory.get("summary", "") + f" Part {part_no}: {first}").strip()[:1400]


def run_part(video, src_dur, style_id, voices, deps, cache_root, work_dir, start_sec, end_sec, part_no, total_est,
             target_sec, memory, voice_pool=None, store=None, recap_style_text="", overview=None, pov="auto",
             lines_path=None):
    """Produce ONE part of a project, continuing from `memory` (characters, voices, names, story so far)."""
    if style_id not in STYLES:
        raise dv.PipelineError(f"unknown story style {style_id!r}")
    spec = STYLES[style_id]
    is_last = end_sec >= src_dur - 5.0
    warnings = []
    mem = json.loads(json.dumps(memory or {}))
    ev = dv.prepare_range(video, src_dur, deps, cache_root, work_dir, start_sec, end_sec,
                          memory_chars=mem.get("characters"), recent=mem.get("recent_cards"))
    ev["src_dur"] = src_dur
    warnings.extend(ev["warnings"])
    cards = ev["cards"]
    ov = overview_for_part(overview, start_sec, end_sec)

    # reverse_open: part 1 opens with the movie's climax (needs the pre-scan)
    cold_ids = None
    if spec.get("reverse") and part_no == 1 and ov.get("climax"):
        try:
            c0 = float(ov["climax"][0]["start"]); c1 = float(ov["climax"][0]["end"])
            if c1 >= start_sec and c0 < end_sec:
                raise dv.PipelineError("the climax lies inside this part")
            cev = dv.prepare_range(video, src_dur, deps, cache_root, os.path.join(work_dir, "cold"),
                                   max(0.0, c0 - 20), min(src_dur, max(c1, c0 + 30) + 20))
            merged = sorted(cev["cards"] + cards, key=lambda c: c["start"])
            for i, c in enumerate(merged):
                c["idx"] = i
            cold_ids = sorted(c["idx"] for c in merged if c in cev["cards"] and c["imp"] >= 3)[:4]
            ev["cards"], cards = merged, merged
            ev["transcript"] = sorted(ev["transcript"] + cev["transcript"], key=lambda r: r["start"])
            if not cold_ids:
                cold_ids = None
        except dv.PipelineError as e:
            warnings.append(f"cold open skipped: {e}")
    elif spec.get("reverse") and part_no == 1:
        warnings.append("reverse_open needs the pre-scan overview; started without a cold open")
    part_cards = [c for c in cards if start_sec - 0.5 <= c["start"] < end_sec - 0.5 and (cold_ids is None or c["idx"] not in cold_ids)]
    if not part_cards:
        raise dv.PipelineError("part has no scene cards")

    kind = spec["kind"]
    if kind == "mixed" and not ev["transcript"]:
        warnings.append("no transcript: radio drama fell back to narrator-only")
        kind = "narr"
    spec = dict(spec, kind=kind)
    pov_name = ""
    if spec.get("pov"):
        pov_name = mem.get("pov") or (pov if pov and pov != "auto" else "")
        if not pov_name:
            cs = (overview or {}).get("characters") or ev["bible"]
            pov_name = str((cs[0].get("name") if cs else "") or "the main character")
        mem["pov"] = pov_name

    role_names = spec.get("roles")
    roster = []
    if kind == "mixed":
        use_voices, roster = dv.assign_voices(ev["bible"], part_cards, dict(voices), voice_pool, preset=mem.get("voices"))
        mem["voices"] = {k: v for k, v in use_voices.items() if k in roster}
    else:
        use_voices = {"NARRATOR": voices["NARRATOR"]}
        if role_names:
            use_voices["HOST_A"] = voices["NARRATOR"]
            use_voices["HOST_B"] = voices["FEMALE"] if voices["NARRATOR"] != voices["FEMALE"] else voices["MALE"]
    cps = dv.calibrate(deps, use_voices, work_dir)

    rules = AUDIO_FIRST + "\n" + spec["rules"].replace("{pov}", pov_name or "the narrator")
    rules += ("\nPROJECT RULE: this retelling is released in parts. Continue seamlessly from the previous part (a one-sentence natural bridge, "
              "never a recap of everything). Never reveal the ending or a twist before this part shows it.")
    writer_style = dv.STYLE_RECAP_DIALOGUE if kind == "mixed" else dv.STYLE_RECAP
    writer = dv.Writer(deps, writer_style, cps, recap_style_text or "", ev["bible"], "Burmese", roster=roster, rules=rules,
                       role_names=role_names)
    writer.names = dict(mem.get("names") or {})
    writer.tail_seed = list(mem.get("tail") or [])[-3:]

    bible = {"logline": ov.get("logline", ""), "genre": ov.get("genre", ""), "tone": ov.get("tone", ""),
             "themes": ov.get("themes", []), "characters": ov.get("characters") or [
                 {"name": c.get("name"), "want": c.get("role", ""), "arc": ""} for c in ev["bible"][:6]],
             "hidden_details": []}
    tail_txt = " / ".join(mem.get("tail") or [])[-400:]
    bridge = (f"Continue seamlessly from the previous part, which ended with: {tail_txt}" if part_no > 1
              else "Open with a strong hook.")
    close = ("End with a satisfying, complete closing." if is_last
             else "End on a hook / open question that makes the listener want the next part.")
    purpose = (f"Part {part_no} of about {total_est} (source {start_sec / 60:.0f}-{end_sec / 60:.0f} min). "
               f"STORY SO FAR: {mem.get('summary') or '(this is the beginning)'} "
               f"UPCOMING (foreshadow only, do not reveal): {' | '.join(ov.get('upcoming', []))}")
    n = 1 if target_sec < 9 * 60 else int(dv._clamp(round(target_sec / 60.0 / CHAPTER_MINUTES), 2, 8))
    groups = _fallback_chapters(part_cards, n)
    chapters = []
    for gi, g in enumerate(groups):
        ids = g["card_idx"]
        chapters.append({"title": f"Part {part_no}" + (f" · {gi + 1}" if len(groups) > 1 else ""), "purpose": purpose,
                         "hook": bridge if gi == 0 else "Keep the momentum from the previous chapter.",
                         "end_question": close if gi == len(groups) - 1 else "Hand over to the next chapter with a question.",
                         "tension": 3, "card_idx": ids, "pool": [ids[0], ids[-1]], "hidden": ov.get("hidden", []) if gi == 0 else []})
    total_w = sum(max(0.3, sum(dv.IMP_W[c["imp"]] * c["dur"] for c in part_cards if ch["pool"][0] <= c["idx"] <= ch["pool"][1]))
                  for ch in chapters) or 1.0
    for ch in chapters:
        wch = max(0.3, sum(dv.IMP_W[c["imp"]] * c["dur"] for c in part_cards if ch["pool"][0] <= c["idx"] <= ch["pool"][1]))
        ch["share"] = wch / total_w

    plan = []
    cold_budget = 0.0
    if cold_ids:
        by_id = {c["idx"]: c for c in cards}
        material = sum(min(dv.S_MAX, by_id[i]["dur"] * 1.10 + 0.5) for i in cold_ids)
        cold_budget = float(min(dv._clamp(target_sec * 0.06, 25.0, 90.0), 0.8 * material))
        plan.append(({"title": "Cold open", "hook": "", "purpose": "cold open", "end_question": "bridge back to the start",
                      "tension": 5, "card_idx": cold_ids, "pool": [cold_ids[0], cold_ids[-1]], "hidden": []},
                     cold_budget, True, cold_ids))
    for ch in chapters:
        plan.append((ch, (target_sec - cold_budget) * ch["share"], False, None))

    counter = [1]
    lines_cache = dv._jload(lines_path) if lines_path else None
    lines_cache = lines_cache or {}
    all_scenes, ch_info, stats_all, carry = [], [], [], 0.0
    for i, (ch, budget, is_cold, cids) in enumerate(plan):
        budget = budget + dv._clamp(carry, -0.3 * budget, 0.5 * budget)
        deps.log(f"📖 Part {part_no} · အခန်း {i + 1}/{len(plan)} (target {budget:.0f}s)")
        scenes, st = produce_chapter(deps, writer, ev, spec, bible, ch, i, len(plan), budget, use_voices, work_dir, counter,
                                     pov_name, lines_cache, is_cold, cids, store, lines_path)
        if not scenes:
            warnings.append(st.get("warning", f"chapter {i + 1} skipped"))
            continue
        carry = budget - st.get("audio", 0.0)
        ch_info.append({"title": ch["title"], "idx": i, "first_scene": len(all_scenes), "budget": round(budget, 1),
                        "audio": round(st.get("audio", 0.0), 1), "cold": is_cold})
        all_scenes.extend(scenes)
        stats_all.append(st)
        writer.tail_seed = [re.sub(r"\[[^\]]+\]\s*", "", s["script"]) for s in scenes[-3:]]
    if not all_scenes:
        raise dv.PipelineError("no chapter could be produced for this part")

    plain = [re.sub(r"\[[^\]]+\]\s*", "", s["script"]) for s in all_scenes]
    title, summary = summarize_part(deps, mem, part_no, " ".join(plain), is_last)
    for c in ch_info:
        if not c["cold"] and len(ch_info) <= 2:
            c["title"] = title
    audio_total = sum(s["_dur"] for s in all_scenes)
    mem.update({"summary": summary, "tail": plain[-3:], "names": writer.names,
                "characters": [dict(c) for c in ev["bible"]] or mem.get("characters", []),
                "recent_cards": [c["summary"] for c in part_cards[-6:]]})
    qa = {"story_style": style_id, "story_part": part_no, "story_chapters": len(ch_info), "story_beats": len(all_scenes),
          "story_target_sec": round(target_sec, 1), "story_audio_sec": round(audio_total, 1),
          "story_length_ratio": round(audio_total / target_sec, 4), "story_cards": len(part_cards),
          "story_register_fixed": writer.stats["register_fixed"], "story_local_fixed": writer.stats["local_fixed"],
          "story_dedupe_dropped": writer.stats["dedupe_dropped"], "story_pov": pov_name,
          "story_drift_beats": sum(1 for s in all_scenes if s.get("_drift", 0) > 2.0),
          "story_chapters_resumed": sum(1 for x in stats_all if x.get("resumed")), "story_warnings": warnings}
    if abs(audio_total - target_sec) / target_sec > 0.10:
        warnings.append(f"part audio {audio_total:.0f}s differs from target {target_sec:.0f}s by more than 10%")
    return {"scenes": all_scenes, "qa": qa, "chapters": ch_info, "title": title, "memory": mem, "is_last": is_last}
