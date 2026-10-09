import os
import hashlib
import asyncio
import time
import tempfile
import shutil
import subprocess
import urllib.request
import ssl
import re
import json
import math
import threading
from fastapi import FastAPI, BackgroundTasks, UploadFile, File, Form, Header, HTTPException, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import edge_tts

from google import genai
from google.genai import types as genai_types
from typing import Optional

app = FastAPI(title="Transcript Master AI Engine", docs_url=None, redoc_url=None, openapi_url=None)

# ── Basic per-IP rate limiting ───────────────────────────────────────────
# The internet constantly scans public HTTP servers for exposed secrets
# (.env, config.js, etc.) - this is background noise every public Space
# gets, not a targeted attack, and none of it can succeed here since
# nothing sensitive is served by any real route. This still adds a simple
# request-rate cap per IP so that kind of scanning can't hammer the server
# or exhaust resources, on top of the APP_TOKEN gate on real endpoints.
_rate_buckets = {}  # ip -> [timestamps within the current window]
_RATE_LIMIT = 120     # max requests per window per IP
_RATE_WINDOW = 60     # seconds

@app.middleware("http")
async def _rate_limit_middleware(request: Request, call_next):
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    bucket = _rate_buckets.setdefault(ip, [])
    # Drop timestamps outside the current window
    cutoff = now - _RATE_WINDOW
    while bucket and bucket[0] < cutoff:
        bucket.pop(0)
    if len(bucket) >= _RATE_LIMIT:
        return JSONResponse({"error": "Too many requests"}, status_code=429)
    bucket.append(now)
    # Keep the tracking dict from growing forever across many distinct IPs
    if len(_rate_buckets) > 5000:
        stale = [k for k, v in _rate_buckets.items() if not v or v[-1] < cutoff]
        for k in stale[:2000]:
            _rate_buckets.pop(k, None)
    return await call_next(request)
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DOWNLOAD_DIR  = os.path.join(BASE_DIR, "downloads")
FONTS_DIR     = os.path.join(BASE_DIR, "fonts")
UPLOAD_STAGING_DIR = os.path.join(BASE_DIR, "chunked_uploads")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(FONTS_DIR, exist_ok=True)
os.makedirs(UPLOAD_STAGING_DIR, exist_ok=True)

# P2-1: purge staging chunks older than 24 h so a crashed upload never fills the disk.
def _cleanup_staging():
    cutoff = time.time() - 86400
    for d in os.listdir(UPLOAD_STAGING_DIR):
        dp = os.path.join(UPLOAD_STAGING_DIR, d)
        if os.path.isdir(dp):
            try:
                mtime = max((os.path.getmtime(os.path.join(dp, f)) for f in os.listdir(dp)), default=0)
                if mtime < cutoff:
                    shutil.rmtree(dp, ignore_errors=True)
            except: pass
try: _cleanup_staging()
except: pass

print(f"[STARTUP] FONTS_DIR: {FONTS_DIR}")
print(f"[STARTUP] Fonts: {os.listdir(FONTS_DIR) if os.path.exists(FONTS_DIR) else 'NOT FOUND'}")
app.mount("/downloads", StaticFiles(directory=DOWNLOAD_DIR), name="downloads")
app.mount("/fonts",     StaticFiles(directory=FONTS_DIR),    name="fonts")

# Voice Clone + Thumbnail static dirs
DATA_DIR       = os.path.join(BASE_DIR, "data")
AUDIO_DIR      = os.path.join(DATA_DIR, "generated_audio")
THUMB_DIR      = os.path.join(DATA_DIR, "thumbnails")
PREVIEW_DIR    = os.path.join(DATA_DIR, "previews")
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(THUMB_DIR, exist_ok=True)
os.makedirs(PREVIEW_DIR, exist_ok=True)

# P2-3: sweep thumbnails and previews older than 24 h.
def _cleanup_media_dir(directory, prefix=""):
    if not os.path.isdir(directory): return
    cutoff = time.time() - 86400
    for f in os.listdir(directory):
        fp = os.path.join(directory, f)
        if os.path.isfile(fp):
            try:
                if os.path.getmtime(fp) < cutoff:
                    os.remove(fp)
            except: pass
try: _cleanup_media_dir(THUMB_DIR)
except: pass
try: _cleanup_media_dir(PREVIEW_DIR)
except: pass
app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")
app.mount("/thumbnails", StaticFiles(directory=THUMB_DIR), name="thumbnails")
app.mount("/previews", StaticFiles(directory=PREVIEW_DIR), name="previews")

task_logs = {}
preview_states = {}
# Structured job state so the frontend polls machine-readable fields instead of
# substring-matching Burmese prose.  {video_id: {state, part, total, message}}
job_state = {}
job_qa = {}  # machine-readable sync/quality report per job
def log_status(video_id, msg):
    task_logs[video_id] = msg
    print(f"[{video_id}] {msg}")

def set_job_state(video_id, **kw):
    cur = job_state.setdefault(video_id, {"state": "running", "part": 0, "total": 0, "message": ""})
    cur.update(kw)
    cur["_t"] = time.time()

def update_preview(video_id, state):
    preview_states[video_id] = state

# ── Optional APP_TOKEN gate ────────────────────────────────────────────────
# If the env var APP_TOKEN is set, every destructive / expensive endpoint
# must include  Authorization: Bearer <token>.  If the var is unset, all
# requests are accepted (backward-compatible default).
_APP_TOKEN = os.environ.get("APP_TOKEN", "").strip()

async def require_token(authorization: str = Header(None)):
    if not _APP_TOKEN:                       # gate disabled
        return
    token = ""
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    if token != _APP_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing APP_TOKEN")

def _form_bool(value, default=False):
    """Normalize browser FormData booleans without relying on truthiness.

    FormData sends strings, and bool('false') is True. Keeping this conversion
    at the process boundary prevents silent transform/mode inversions.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on", "checked"}

def _safe_aspect_ratio(value):
    value = str(value or "16:9").strip()
    return value if value in {"16:9", "9:16", "1:1"} else "16:9"

# ==========================================
# 1. DEFAULT FONT
# ==========================================
DEFAULT_FONT_PATH = os.path.join(FONTS_DIR, "Padauk.ttf")

def download_default_font():
    if not os.path.exists(DEFAULT_FONT_PATH):
        print("Downloading Default Unicode Font...")
        font_url = "https://raw.githubusercontent.com/google/fonts/main/ofl/padauk/Padauk-Regular.ttf"
        try:
            subprocess.run(["curl", "-L", "-k", "-o", DEFAULT_FONT_PATH, font_url], check=True)
        except:
            try:
                ctx = ssl.create_default_context()
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
                req = urllib.request.Request(font_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=30, context=ctx) as r, open(DEFAULT_FONT_PATH, 'wb') as f:
                    shutil.copyfileobj(r, f)
            except Exception as e:
                print(f"Font download failed: {e}")

download_default_font()

# ==========================================
# 2. GEMINI AI — multi-key rotation + current Flash fallback
# ==========================================
# Supports:
#   GOOGLE_API_KEY  = single key (backward compatible)
#   GOOGLE_API_KEYS = comma-separated multiple keys (each gets 50 req/day free)
# When all keys hit 429, use the current Flash-Lite model accepted by Google.
_GEMINI_KEYS = []
for _k in (os.environ.get("GOOGLE_API_KEYS") or "").split(","):
    _k = _k.strip().strip('"').strip("'")
    if _k and len(_k) > 10:
        _GEMINI_KEYS.append(_k)
if not _GEMINI_KEYS:
    _single = os.environ.get("GOOGLE_API_KEY", "").strip()
    if _single:
        _GEMINI_KEYS = [_single]

_gemini_idx = 0  # round-robin index
gemini_client = None
if _GEMINI_KEYS:
    gemini_client = genai.Client(api_key=_GEMINI_KEYS[0])
    print(f"[GEMINI] Loaded {len(_GEMINI_KEYS)} API key(s)")

# Model routing, as cascades rather than single names: Google has been
# retiring Gemini models abruptly through 2026 (sometimes with little or no
# warning even on "stable" models) — a hardcoded single model name means the
# whole app breaks the next time Google cuts one. Each purpose gets an
# ordered list; call_gemini_with_retry automatically advances to the next
# model in the list when the current one comes back "no longer available",
# so a Google-side deprecation degrades gracefully instead of failing outright.
# Override any of these via env var (comma-separated) without touching code.
_RETIRED_GEMINI_MODELS = {"gemini-2.5-flash-lite"}
def _model_cascade(env_name, fallback):
    raw = os.environ.get(env_name, fallback)
    models = []
    for item in raw.split(','):
        name = item.strip().removeprefix('models/')
        if name and name not in _RETIRED_GEMINI_MODELS and name not in models:
            models.append(name)
    return models or [fallback.split(',')[0]]

GEMINI_RECAP_MODELS = _model_cascade(
    "GEMINI_RECAP_MODELS", "gemini-2.5-flash,gemini-3.5-flash-lite")
GEMINI_DEFAULT_MODELS = _model_cascade(
    "GEMINI_DEFAULT_MODELS", "gemini-2.5-flash,gemini-3.5-flash-lite")
GEMINI_TTS_MODELS = [m.strip() for m in os.environ.get(
    "GEMINI_TTS_MODELS",
    "gemini-3.1-flash-tts-preview,gemini-2.5-flash-preview-tts"
).split(",") if m.strip()]
_MODEL_CASCADES = {"recap": GEMINI_RECAP_MODELS, "default": GEMINI_DEFAULT_MODELS, "tts": GEMINI_TTS_MODELS}
_cascade_pos = {"recap": 0, "default": 0, "tts": 0}

def _advance_model_cascade(purpose):
    """Move to the next model in this purpose's cascade (Google retired the
    current one). Returns the new model name, or None if already at the end."""
    models = _MODEL_CASCADES.get(purpose)
    if not models:
        return None
    pos = _cascade_pos.get(purpose, 0)
    if pos < len(models) - 1:
        _cascade_pos[purpose] = pos + 1
        return models[_cascade_pos[purpose]]
    return None

# ── Recap script styles (user-selectable per video) ──────────────────────
# Each entry is the narrator-voice + pacing portion of the script prompt.
# The shared JSON/scene-boundary rules are appended after whichever one is
# selected — only the "how it sounds" part differs between styles.
RECAP_STYLES = {
    "fast_storyteller": {
        "label": "Fast Fluent Storyteller",
        "label_mm": "တတ်တတ်ကျွကျွ မြန်မြန်ပြောပြပုံ",
        "voice": """STYLE RULES (fast, fluent, confident movie-recap storyteller):
1. Talk like a lively friend telling a gripping story out loud: quick, punchy, natural SPOKEN {target_lang}.
   Never textbook, never news-reader, never a literal subtitle translation.
2. Short sentences (about 8-18 words). Keep the pace moving: one clear event or reveal per sentence.
   Use spoken connectors a real person would say, e.g. "ဒါနဲ့", "အဲဒီမှာပဲ", "ပြီးတော့", "ဒါပေမယ့်", "တော်တော်ကို", "တစ်ခါတည်း".
3. Open with a hook (a shocking fact or a question). Raise the stakes step by step; save the biggest reaction for real twists.
4. Never invent facts, motives or events the evidence does not show. Keep names and pronouns consistent.
5. Spoken endings only (...တယ်၊ ...ပါတယ်၊ ...ခဲ့တယ်၊ ...တော့). No book-style forms such as ...သည်၊ ၏၊ တွင်၊ ၍.
6. No English words except proper nouns. Output narration text only.""",
    },
    "cinematic": {
        "label": "Natural Cinematic Recap",
        "label_mm": "သဘာဝကျသော ရုပ်ရှင်ဇာတ်လမ်းပြန်ပြောပုံ",
        "voice": """STYLE RULES (natural cinematic movie-recap narration):
1. Write in natural spoken {target_lang}, as a skilled human narrator would
   tell the story aloud. Do not sound like a literal subtitle translation,
   textbook, news report, or AI-generated list.
2. Use one consistent third-person narrator. Do not switch randomly between
   formal literary Burmese, slang, first person, and direct audience address.
   For Burmese, prefer natural spoken endings such as တယ်, ပါတယ်, ရဲ့, တော့.
3. Tell the story, not just the pixels: connect visible actions to clearly
   supported context, motive, consequence, and stakes. Never invent a fact,
   relationship, thought, or event that the supplied evidence does not prove.
4. Use hooks sparingly. Reserve a suspense transition for a real reveal,
   reversal, danger, or unanswered consequence. Do not repeat phrases like
   'ဒါပေမယ့် သူမသိသေးတာက' or 'နောက်တစ်ခဏမှာ' on ordinary shots.
5. Vary sentence length and pacing. Combine related facts into a smooth,
   meaningful narration line; do not make every camera cut sound like a
   disconnected caption. Keep minor connective moments brief and give major
   turning points enough context to land emotionally.
6. Keep names, pronouns, relationships, and tense consistent from beginning
   to end. Do not restart the story or repeat an introduction in later parts.
7. No English words unless they are proper nouns. Output only narration text
   inside the requested JSON fields, with no commentary or writing advice.""",
    },
    "hook": {
        "label": "Cliffhanger Hook (Short-Drama)",
        "label_mm": "ထိပ်တန်း ဆွဲငင်မှု ပုံစံ",
        "voice": """STYLE RULES (write like a real recap YouTuber, not a dry visual description):
1. Write entirely in natural, spoken-style {target_lang} — the way a narrator
   talks, not the way a subtitle-translator writes literally.
2. Tell the STORY (motivations, stakes, tension, twists) — don't just
   describe what is visually on screen shot-by-shot.
3. No English words unless they are proper nouns (character/place names).

NARRATIVE STRUCTURE (the style used by top-performing short-drama recap
channels):
- COLD-OPEN HOOK: the very first entry (or first couple of entries) must
  drop the viewer straight into intrigue, conflict, or a striking moment -
  never a slow "once upon a time" style setup. If the opening footage
  itself is slow, frame the narration around a question or stake that
  makes the viewer need to know what happens ("but he had no idea what
  was coming next...", "what she didn't know yet would change everything").
- PUNCHY, URGENT SENTENCES: keep each entry to roughly one short, punchy
  spoken sentence (under ~25 words is ideal, 40 words max). This is an
  energetic, fast-talking narrator, not a calm documentary voice. Split
  longer beats into multiple short consecutive entries rather than one
  long one.
- ESCALATING TENSION: structure the run of entries so that stakes/tension
  visibly rise - each new revelation, conflict, or turning point should
  feel like a step up from the last, not flat/even pacing throughout.
- FREQUENT DRAMATIC TRANSITIONS: at most scene-cuts where something
  changes (a reveal, a reversal, a new conflict), end or start the
  narration with a hook phrase in {target_lang} equivalent in spirit to
  "but little did they know...", "at that exact moment...", "what
  happened next..." - this is the DEFAULT style for this genre, not an
  occasional flourish. Reserve plain description-only lines for minor
  connective moments.
- EACH ENTRY EARNS ITS PLACE: every single line should make the viewer
  want to see the next second of footage. If a moment is genuinely dull
  connective tissue, keep its narration brief rather than padding it.""",
    },
    "documentary": {
        "label": "Formal Documentary Narrator",
        "label_mm": "တရားဝင် မှတ်တမ်းဇာတ်ကား ပုံစံ",
        "voice": """STYLE RULES (a calm, formal, documentary-style movie-recap voice):
1. Write entirely in natural {target_lang}, in a CALM, FORMAL, documentary
   narration tone — measured and composed, not energetic or hyped-up.
2. STRICT third-person omniscient narrator throughout. NEVER use casual
   slang, colloquial fillers, or a conversational/YouTuber voice — e.g.
   NEVER phrases equivalent to "our hero", "you guys", "this guy here",
   "let's see what happens". The narrator is an unseen, formal storyteller,
   not a personality talking to the camera.
3. Every sentence must end with a formal declarative narrative particle —
   equivalent in spirit to "...ဖြစ်ပါတယ်", "...ခဲ့ပါတယ်",
   "...သိလိုက်ရပါတယ်", "...ပြောဆိုခဲ့ပါတယ်", "...လိုက်ရပါတော့တယ်". Avoid
   abrupt conversational stops — each sentence should flow smoothly into
   the context/action around it, not read like a blunt caption.
4. Be chronological and context-rich: name characters, dates, years,
   locations, and background explicitly whenever relevant (e.g. an
   equivalent of "in May 1980...", "journalist Peter..."). Make cause and
   effect explicit — spell out a character's internal motive alongside the
   external circumstance driving it, using connective phrasing equivalent
   to "because of this...", "having found out that...", "however...".
5. No English words unless they are proper nouns (character/place names).
6. Tell the STORY (motivations, stakes, cause-and-effect) rather than a
   flat shot-by-shot visual description — but always within the calm,
   formal register above, never as a hyped-up hook.

DELIVERY: the narration should read as one continuous, smoothly-flowing
documentary account of the story — clean, sequential text ready to be
read aloud as voiceover, with accurate, natural, grammatically intact
spelling suited to text-to-speech.""",
    },
    "hybrid": {
        "label": "Hybrid (Formal Voice + Hooks)",
        "label_mm": "ရောနှောပုံစံ (တရားဝင် + ဆွဲငင်မှု)",
        "voice": """STYLE RULES (hybrid style: a formal, context-rich documentary narrator
voice, delivered with the hook-driven pacing of a top recap channel):
1. Write entirely in natural {target_lang}, in a composed, formal narrator
   voice — NOT slangy or conversational. Strict third-person omniscient
   throughout; never a casual/YouTuber voice (no equivalents of "our
   hero", "you guys", "this guy here", "let's see what happens").
2. Sentences should end with formal declarative narrative particles —
   equivalent in spirit to "...ဖြစ်ပါတယ်", "...ခဲ့ပါတယ်",
   "...သိလိုက်ရပါတယ်", "...ပြောဆိုခဲ့ပါတယ်", "...လိုက်ရပါတော့တယ်" — not
   abrupt conversational stops.
3. Be context-rich where it matters: name characters, dates, years,
   locations, and background when relevant, and make cause-and-effect
   explicit (motive + circumstance driving it) using connective phrasing
   equivalent to "because of this...", "having found out that...",
   "however...". Sentences can run longer than a punchy caption when
   context requires it — clarity and story weight matter more than a
   strict word cap.
4. Tell the STORY (motivations, stakes, tension, twists) — don't just
   describe what is visually on screen shot-by-shot.
5. No English words unless they are proper nouns (character/place names).

NARRATIVE STRUCTURE (this is the specific pacing/delivery to follow — keep
the formal narrator voice above, but structure it like a top-performing
recap channel, not a flat, even-paced documentary):
- COLD-OPEN HOOK: the very first entry (or first couple of entries) must
  drop the viewer straight into intrigue, conflict, or a striking moment -
  never a slow "once upon a time" style setup. If the opening footage
  itself is slow, frame the narration around a question or stake that
  makes the viewer need to know what happens, still in the formal voice
  (e.g. "...ဒါပေမယ့် သူ မသိသေးတာက...", "...ဒါကို သူ တစ်ခါမှ မမျှော်လင့်ခဲ့ပါ").
- ESCALATING TENSION: structure the run of entries so that stakes/tension
  visibly rise - each new revelation, conflict, or turning point should
  feel like a step up from the last, not flat/even pacing throughout.
- FREQUENT DRAMATIC TRANSITIONS: at most scene-cuts where something
  changes (a reveal, a reversal, a new conflict), end or start the
  narration with a hook phrase equivalent in spirit to "but little did
  they know...", "at that exact moment...", "what happened next..." —
  phrased in the formal register (rule 1-2 above), not a hyped-up
  conversational one. This is the DEFAULT style for this genre, not an
  occasional flourish. Reserve plain description-only lines for minor
  connective moments.
- EACH ENTRY EARNS ITS PLACE: every entry should make the viewer want to
  see what happens next. If a moment is genuinely dull connective tissue,
  keep its narration brief rather than padding it — but don't sacrifice
  needed context (names/dates/cause-effect) just to be short.""",
    },
    "cdrama": {
        "label": "Chinese AI Short-Drama Recap (TikTok Viral)",
        "label_mm": "တရုတ် Short-Drama Viral ပုံစံ",
        "voice": """STYLE RULES (the style used by viral Chinese short-drama ("AI drama" /
micro-drama) recap accounts trending on TikTok — maximum-hook, trope-driven
delivery):
1. Write entirely in natural, spoken-style {target_lang} — punchy and
   dramatic, the way a viral short-form narrator talks, never a flat
   description.
2. Lean into genre TROPES explicitly and name them as the story reveals
   them - hidden identity, secret heir, revenge-after-humiliation, sudden
   rebirth/second chance, rags-to-riches reversal, a character who is
   secretly far more powerful/wealthy than everyone around them believes.
   When the footage supports it, frame character intros around their
   hidden status/role rather than just a name (e.g. "the young woman
   everyone underestimated", "the man they all thought was powerless").
3. EXTREMELY SHORT, punchy sentences - shorter than normal recap pacing
   (aim for well under 20 words per entry). Split every beat into as many
   short consecutive entries as needed rather than one longer one.
4. MAXIMUM HOOK DENSITY: nearly every entry should end on a hook, a
   reveal-tease, or a rhetorical beat equivalent in spirit to "but she had
   a secret nobody knew...", "what he said next shocked everyone...",
   "she was about to make him regret every word...". This style has almost
   no purely flat/neutral connective lines - keep tension high virtually
   continuously.
5. Favor sharp reversals and payoff moments — when a character gets
   humiliated, betrayed, or underestimated early on, make the eventual
   comeback/reveal moment land as a clear, satisfying turn the narration
   visibly sets up for.
6. No English words unless they are proper nouns (character/place names).
7. Tell the STORY (motivations, stakes, twists) — don't just describe what
   is visually on screen shot-by-shot.""",
    },
    "investigator": {
        "label": "Analytical Investigator (Case-File Style)",
        "label_mm": "စုံထောက် အကဲဖြတ် ပုံစံ",
        "voice": """STYLE RULES (an analytical "case-file" narrator style — framing the
recap as investigating and breaking down what really happened, good for
mystery/thriller/crime stories but usable generally):
1. Write entirely in natural {target_lang}, in a measured, analytical
   narrator voice — composed and inquisitive, not hyped-up or slangy.
2. Structure the recap around uncovering the truth behind what happened.
   You may occasionally frame a genuine mystery as a question, but do not
   repeat this as a device on every beat — most of the time, state the
   puzzling fact plainly and let the SUBSEQUENT reveal do the work,
   rather than asking the audience to wonder.
3. Call out foreshadowing and clues explicitly when the plot reveals their
   significance — equivalent in spirit to "this detail seemed unimportant
   at the time... but it would matter later", "what looked like a
   coincidence was, in fact, deliberate."
4. Make cause-and-effect and motive explicit and precise — this style is
   about UNDERSTANDING the story's mechanics, not just feeling its drama.
   Use connective phrasing equivalent to "because of this...", "this is
   what explains...", "however, what nobody realized was...".
5. Sentences can run a bit longer than a punchy hook style when needed for
   clarity, but stay tight — avoid rambling. Prefer precision over flourish.
6. No English words unless they are proper nouns (character/place names).
7. Tell the STORY (motivations, stakes, twists) — don't just describe what
   is visually on screen shot-by-shot.""",
    },
    "immersive": {
        "label": "Immersive Cinematic Storyteller",
        "label_mm": "ဇာတ်ကားထဲ ဝင်ရောက်ခံစားရမယ့် ပုံစံ",
        "voice": """STYLE RULES (the style of top professional recap channels that make
someone who has NEVER seen the movie feel like they just watched the
whole thing — engagement built through immersion and momentum, NOT
through repeated rhetorical questions):
1. Write entirely in natural, vivid {target_lang}. Put the listener INSIDE
   each scene: name what a character sees, hears, and feels in the
   moment, not just what happens next in the plot. A few concrete
   sensory/atmospheric details (a look on someone's face, the tension in
   a room, a sound that changes everything) do more work than a
   generic play-by-play.
2. DO NOT lean on rhetorical questions as a hook device (avoid repeatedly
   asking things like "but what happens next?" or "could this be true?").
   Build curiosity instead through what you reveal and withhold: state
   something intriguing as FACT, let the implication create the tension,
   and let the next scene pay it off. One occasional question is fine;
   it must never become a repeated crutch.
3. Give every major character a consistent, recognizable narrative voice
   across the whole recap — describe them the same way each time they
   matter so the listener tracks who's who and comes to care what
   happens to them, the same way an actual viewer would.
4. Maintain a clear emotional throughline: know what the protagonist
   wants and fears, and narrate events in terms of how they move that
   forward, not just as a list of things that occurred.
5. Callbacks pay off: when something set up earlier becomes relevant
   again, say so explicitly and briefly, so the reveal LANDS instead of
   passing by unnoticed.
6. Vary pacing deliberately: quicker, punchier beats for
   action/confrontation; a touch more space for a genuinely pivotal
   emotional moment. Uniform pacing throughout reads as flat and
   disengaging even if individual lines are good.
7. No English words unless they are proper nouns (character/place names).
8. Tell the STORY (motivations, stakes, twists) — don't just describe what
   is visually on screen shot-by-shot.""",
    },
    "emotional": {
        "label": "Character-Driven Emotional Journey",
        "label_mm": "ဇာတ်ကောင် စိတ်ခံစားမှု ဗဟိုပြု ပုံစံ",
        "voice": """STYLE RULES (anchors the whole recap in the protagonist's inner
experience, so the audience becomes emotionally invested in outcomes
rather than just informed of plot facts — a technique used by
recap channels whose audience stays through the full runtime):
1. Write entirely in natural {target_lang}. For every major scene, narrate
   not just WHAT happens but what it COSTS or MEANS to the character
   experiencing it — their fear, hope, humiliation, relief, resolve.
   Plot facts should arrive already filtered through what they mean for
   someone the audience is rooting for (or against).
2. Establish the protagonist's core want and core fear early, and keep
   returning to them: frame later events as movement toward or away from
   that want/fear so the audience always knows what's emotionally at
   stake, not just what's plot-mechanically happening.
3. Avoid rhetorical questions as the primary engagement tool. Build
   investment through empathy and stakes instead — make the listener
   feel WITH the character, not quizzed about the plot.
4. Let quieter emotional beats breathe with a touch more narrative space
   than pure plot mechanics need - a betrayal, a reunion, a loss should
   register as a moment, not a bullet point equal in weight to a scene
   transition.
5. Villains and obstacles should be framed through their effect on the
   protagonist's emotional stakes, not merely described as external plot
   devices.
6. No English words unless they are proper nouns (character/place names).
7. Tell the STORY (motivations, stakes, twists) — don't just describe what
   is visually on screen shot-by-shot.""",
    },
    "banter": {
        "label": "Character Voice Reenactment (Back-and-Forth Dialogue)",
        "label_mm": "ဇာတ်ကောင် အပြန်အလှန် စကားပြော ပုံစံ",
        "voice": """STYLE RULES (the narrator steps in and out of the characters'
own voices at key moments, the way viral "acted-out recap" creators do —
distinct from a pure third-person narrator style):
1. Write entirely in natural, spoken-style {target_lang}. The BASE layer
   is still a narrator connecting scenes together, but at pivotal
   moments (a confrontation, a reveal, a joke, an emotional beat) drop
   INTO short quoted lines voiced as if the character is speaking
   directly - equivalent in spirit to: Narrator sets the scene, then:
   character-voice line in quotes, then narrator reacts/continues.
2. SMOOTH TRANSITIONS ARE MANDATORY - this is the single most important
   rule of this style. The narrator line going INTO a quote and the
   narrator line coming OUT of one must read as ONE continuous voice
   telling a story, never as two disconnected pieces stitched together.
   Lead into a quote with a natural spoken connector that sets up why the
   character is about to speak ("so he turns to her and goes...", "she
   just looks at him and says...", "and that's when he finally admits
   it..."), and come out of a quote by reacting to or continuing directly
   from what was just quoted - never restart the sentence structure or
   re-explain what the quote already said. If a smooth lead-in or
   follow-up doesn't come naturally for a given moment, skip the quote
   there entirely and stay in narrator voice - a clean narrated beat
   always beats an awkwardly bolted-on quote.
3. Keep in-character quoted lines SHORT and punchy (one line, rarely
   two) - this is a dramatized beat, not a full script transcription.
   Alternate between at least two characters' "voices" when a scene is a
   back-and-forth exchange, so it reads like a mini reenactment, not one
   person's monologue.
4. Give each recurring character a consistent verbal flavor (a favorite
   phrase, a tone - confident, nervous, sly) so the audience recognizes
   who's "speaking" without needing a name-tag every time.
5. Use the narrator layer for connective tissue, reactions, and stakes
   ("but she wasn't buying it for a second...") - the character-voice
   lines are seasoning throughout the recap, not a replacement for
   narration; most of the runtime should still be narrated normally.
6. No English words unless they are proper nouns (character/place names).
7. Tell the STORY (motivations, stakes, twists) — don't just describe what
   is visually on screen shot-by-shot.""",
    },
    "reactor": {
        "label": "Reactor Commentary (Personal Reactions Woven In)",
        "label_mm": "တုံ့ပြန်ချက်ပါ ပြောပြတဲ့ ပုံစံ (Reaction Style)",
        "voice": """STYLE RULES (the narrator is a visible personality reacting to the
story as they tell it, the way popular "reaction commentary" recap
creators do — distinct from an invisible/neutral narrator):
1. Write entirely in natural, spoken-style {target_lang}, as if a real
   person is watching along with the audience and reacting out loud in
   the moment - equivalent in spirit to "okay, THIS is where it gets
   good...", "I did not expect that twist, honestly...", "you have to
   feel for her here...".
2. Weave short first-person reaction asides into the narration at
   genuinely surprising, funny, or emotional beats - not on every single
   line (that gets exhausting), but often enough that the personality
   comes through as a throughline, roughly one aside every few beats.
3. Still tell the actual STORY accurately and completely underneath the
   reactions - the commentary is seasoning on top of real plot
   narration, never a replacement for it or a distraction from it.
4. Keep the reactions PG and good-natured - genuine enthusiasm,
   surprise, secondhand embarrassment, admiration - not mean-spirited
   mockery of the material or the characters.
5. No English words unless they are proper nouns (character/place names).
6. Tell the STORY (motivations, stakes, twists) — don't just describe what
   is visually on screen shot-by-shot.""",
    },
    "pov": {
        "label": "Second-Person POV (\"Imagine You're There\")",
        "label_mm": "မင်းပဲ ဖြစ်နေရင် ပုံစံ (Second-Person)",
        "voice": """STYLE RULES (the recap addresses the audience directly as "you",
putting them inside the protagonist's shoes — a technique used by viral
short-form storytellers to maximize personal investment):
1. Write entirely in natural, spoken-style {target_lang}, addressing the
   audience directly in the second person at key moments - equivalent in
   spirit to "imagine finding out the person you trusted most had been
   lying the whole time...", "now you're standing there, and you have
   seconds to decide...". Use this framing most heavily at emotionally
   loaded turning points, not necessarily every single line.
2. Where the second-person framing would feel forced (pure connective
   plot mechanics), fall back to normal third-person narration - don't
   contort every sentence into "you" just to keep the gimmick constant.
3. Make the stakes personal and immediate: frame consequences in terms of
   what "you" would feel, risk, or lose, not just what happens to a
   character with a name.
4. Keep sentences punchy and immediate - short, present-feeling phrasing
   suits this style much better than long, clause-heavy sentences.
5. No English words unless they are proper nouns (character/place names).
6. Tell the STORY (motivations, stakes, twists) — don't just describe what
   is visually on screen shot-by-shot.""",
    },
    "comedic": {
        "label": "Comedic / Sarcastic Commentary",
        "label_mm": "ဟာသ / ချောင်းချောင်းထိုးပုံစံ",
        "voice": """STYLE RULES (a funny, lightly sarcastic recap voice — popular for
comedy-recap and "roast" style channels, while still respecting the
actual story):
1. Write entirely in natural, spoken-style {target_lang}, with a wry,
   funny, slightly teasing tone throughout - equivalent in spirit to
   pointing out how dramatic/absurd/predictable a movie moment is while
   still narrating what actually happens.
2. Land a genuine joke or playful aside at least every few beats - a
   funny comparison, an exaggerated reaction, calling out an obvious
   plot contrivance - without ever losing track of the real plot
   underneath the jokes.
3. Punch at the SITUATION and plot tropes, not at real people, protected
   groups, or anything genuinely dark (abuse, tragedy, etc.) - keep the
   humor about the storytelling and character choices, and drop the
   comedic framing entirely for moments that are meant to land seriously.
4. Still be accurate: exaggeration for comic effect is fine (obvious
   hyperbole the audience reads as a joke), but never actually misstate
   what happens in the plot.
5. No English words unless they are proper nouns (character/place names).
6. Tell the STORY (motivations, stakes, twists) — don't just describe what
   is visually on screen shot-by-shot.""",
    },
    "shorts_rapidfire": {
        "label": "Rapid-Fire Ultra-Condensed (built for Shorts/Reels)",
        "label_mm": "အမြန်ဆုံး ချုံ့ငယ်ပြောပြပုံစံ (Shorts/Reels အတွက်)",
	    "voice": """STYLE RULES (built for very short vertical-video runtimes where every
single second must justify itself - the tightest, highest-density style
available):
1. Write entirely in natural, spoken-style {target_lang}. This is the
   MOST condensed style - assume the viewer will abandon within seconds
   if a line doesn't pull its weight, so cut harder than any other style
   here would.
2. Open on the single most striking hook the story has - the biggest
   twist, the most shocking moment, or the highest-stakes question -
   within the very first entry. Do not build up to it; start there.
3. Extremely short entries - aim for well under 15 words each. One idea
   per entry, no throat-clearing, no scene-setting that isn't strictly
   necessary to follow the plot.
4. Cut ruthlessly: skip every scene, subplot, and character that is not
   essential to the single throughline you're telling. This style would
   rather leave out a real plot detail than pad runtime with it.
5. End on the strongest possible payoff or cliffhanger the material
   supports, so the very last moment still makes the viewer want more.
6. No English words unless they are proper nouns (character/place names).
7. Tell the STORY (motivations, stakes, twists) — don't just describe what
	   is visually on screen shot-by-shot.""",
	    },
	"thriller_suspense": {
	    "label": "Thriller Suspense Build",
	    "label_mm": "သည်းထိတ်ရင်ဖို အဆင့်ဆင့်တက်ပုံ",
	    "voice": """STYLE RULES (a controlled thriller narrator that builds dread without inventing facts):
1. Use natural spoken {target_lang}; make every beat increase uncertainty, danger, or suspicion.
2. Reveal information in the same order as the footage. Do not reveal a twist before its source scene.
3. Emphasize concrete clues, threats, ticking clocks, reversals, and consequences; never use empty hype.
4. Keep sentences tight, with occasional restrained cliffhanger transitions at genuine turning points.
5. Make the audience understand what the character knows versus what remains hidden.
6. Preserve the truth of the footage; do not label a character guilty, supernatural, or dangerous unless supported.
7. Tell the story, not a shot list, and use natural spoken Myanmar endings.""",
	},
	"romance": {
	    "label": "Romance Chemistry & Misunderstanding",
	    "label_mm": "အချစ်ဇာတ်လမ်း ဆက်ဆံရေးဗဟိုပြု",
	    "voice": """STYLE RULES (a warm relationship-centered recap):
1. Write natural spoken {target_lang}, tracking attraction, trust, hesitation, misunderstanding, and emotional payoff.
2. Explain what each person wants and why their choices bring them closer or push them apart.
3. Let looks, silences, promises, betrayals, and small gestures matter when visible in the evidence.
4. Keep conflict and consent clear; never invent feelings or relationships not supported by the story.
5. Use gentle pacing for intimate moments and sharper pacing for separation or betrayal.
6. Avoid cheesy generic romance language; every emotional line must belong to this story.""",
	},
	"action": {
	    "label": "Action Momentum & Tactical Beats",
	    "label_mm": "အက်ရှင် အရှိန်မြန် ပြကွက်ဗဟိုပြု",
	    "voice": """STYLE RULES (a high-energy but clear action recap):
1. Use natural spoken {target_lang} and make geography, objective, obstacle, and consequence easy to follow.
2. Describe the tactical reason behind major actions, not every punch or explosion.
3. Keep momentum high with short sentences during fights and give pivotal reversals enough explanation.
4. Track injuries, weapons, escapes, allies, and changing advantages only when shown or spoken.
5. Avoid meaningless superlatives; the actual stakes and choices should create excitement.
6. Connect every action beat to the next story consequence.""",
	},
	"noir": {
	    "label": "Noir Mystery & Moral Ambiguity",
	    "label_mm": "နွိုက်စတိုင် လျှို့ဝှက်မှုနဲ့ မီးခိုးရောင်ကျင့်ဝတ်",
	    "voice": """STYLE RULES (a restrained noir case narration):
1. Write natural spoken {target_lang} with a dark, observant, atmospheric voice.
2. Highlight debts, secrets, corruption, conflicting motives, and choices with moral cost.
3. Use visual atmosphere only when it supports the mood or story; never replace plot facts with purple prose.
4. Reveal clues chronologically and let ambiguity remain when the movie does not resolve it.
5. Use dry understatement sparingly, not constant jokes or dramatic filler.
6. Keep names, relationships, and causality precise so the mystery remains understandable.""",
	},
	"survival": {
	    "label": "Survival Pressure & Escalation",
	    "label_mm": "ရှင်သန်ရေး ဖိအားအဆင့်ဆင့်တက်ပုံ",
	    "voice": """STYLE RULES (a survival-focused recap):
1. Use natural spoken {target_lang}; frame each scene around resources, danger, decisions, and consequences.
2. Make the changing survival problem clear: what is running out, who is at risk, and what must happen next.
3. Show teamwork, betrayal, sacrifice, and adaptation through source-grounded actions.
4. Escalate pressure across the recap without pretending every moment is a crisis.
5. Keep the physical geography and timeline clear when characters are separated or lost.
6. Do not invent survival skills, injuries, or threats absent from the footage.""",
	},
	"psychological": {
	    "label": "Psychological Descent & Unreliable Reality",
	    "label_mm": "စိတ်ပိုင်းဆိုင်ရာ ကျဆင်းမှုနဲ့ မသေချာတဲ့အမှန်တရား",
	    "voice": """STYLE RULES (a psychological character-and-reality recap):
1. Write natural spoken {target_lang} and distinguish visible fact from a character's belief, fear, memory, or suspicion.
2. Track the protagonist's mental and emotional state as it changes through concrete story events.
3. Signal uncertainty honestly; do not declare hallucination, dream, or deception until the story supports it.
4. Use recurring images, contradictions, and callbacks only when they are present in the evidence.
5. Keep the audience oriented even while explaining confusion: who knows what, when, and why.
6. Prefer precise unease over generic claims that something is mysterious.""",
	},
	"plot_explainer": {
	    "label": "Clear Plot Explainer (Beginner-Friendly)",
	    "label_mm": "ဇာတ်လမ်းကို ရှင်းလင်းလွယ်ကူစွာ နားလည်အောင်ပြောပုံ",
	    "voice": """STYLE RULES (the clearest possible story explanation):
1. Use natural spoken {target_lang}; assume the listener has never seen the movie.
2. Introduce characters by role and relationship before using names repeatedly.
3. Explain cause and effect plainly: what happened, why it happened, and what changed afterward.
4. Use short connective summaries between scenes so time jumps and location changes never confuse the listener.
5. Do not add interpretation where the movie gives no evidence. Clarity is more important than flourish.
6. Preserve major setup, conflict, turning points, consequences, and resolution.""",
	},
	"dialogue_cinematic": {
	    "label": "★ Dialogue-Centered Cinematic",
	    "label_mm": "★ စကားပြောခန်းအသားပေး ရုပ်ရှင်ဆန်သောပုံစံ",
	    "voice": """STYLE RULES (dialogue-centered cinematic recap):
1. Write natural spoken {target_lang}; the narrator explains the plot while preserving source-grounded character exchanges at moments that change the story.
2. For every clear conversation, use a smooth [NARRATOR] setup, then short [MALE] / [FEMALE] lines only when the video or transcript supports them, followed by the consequence.
3. Keep each exchange chronological and visually anchored. Never put dialogue from a later scene over an earlier shot, and never invent a quote to fill runtime.
4. Prioritize meaningful turns: threats, decisions, confessions, questions, refusals, jokes, and emotional reactions. Compress greetings and repeated small talk.
5. Narration remains the connective backbone; dialogue should sharpen character and conflict, not become a raw transcript.
6. Use the exact voice tags [NARRATOR], [MALE], and [FEMALE] when Dialogue mode is active. Do not speak the tags aloud.
7. Keep names, motives, relationships, and cause/effect precise. Use natural spoken Myanmar endings, not formal textbook endings.""",
	},
	"dialogue_banter": {
	    "label": "★ Character Banter & Back-and-Forth",
	    "label_mm": "★ ဇာတ်ကောင်အပြန်အလှန် စကားပြောအသားပေး",
	    "voice": """STYLE RULES (fast character banter recap):
1. Tell the story in natural spoken {target_lang} with a lively narrator and short source-grounded back-and-forth exchanges.
2. At a genuine exchange, alternate [MALE] and [FEMALE] only when the source supports the speaker distinction; otherwise keep the line [NARRATOR].
3. Make every line answer, challenge, reveal, or escalate the line before it. Do not output isolated subtitle fragments or a transcript dump.
4. Keep character lines short enough for clear multi-voice delivery, then use [NARRATOR] to explain what the exchange changes in the plot.
5. Preserve the original conversation order and attach each exchange to the exact scene timestamp. Never foreshadow with dialogue from a later scene.
6. Ban invented quotes, invented speaker identities, repeated filler, and generic descriptions of visible objects.
7. Use [NARRATOR], [MALE], and [FEMALE] tags only as control metadata; do not speak them aloud.""",
	},
	"visual_hook": {
	    "label": "★ Visual Hook Cinematic — TikTok Recap",
	    "label_mm": "★ TikTok Visual Hook ရုပ်ရှင်ပြန်ပြောပုံ",
	    "voice": """STYLE RULES (short-form visual-hook recap for Auto mode):
1. Open with the strongest source-grounded hook in the first beat; do not invent a twist or reveal information before its source scene.
2. Use short natural spoken {target_lang} sentences. Every line must explain what the visual means, what the character wants, what blocks them, or what changes next.
3. Select meaningful visual beats—reaction, clue, decision, threat, action, reveal—not generic object labels or every five-second cut.
4. Use narrator voice for most of the runtime. Insert a brief [MALE] or [FEMALE] quote only when the source clearly supports an important threat, question, confession, refusal, or reaction.
5. Keep the exact chronological scene mapping. Hook wording may create curiosity, but the footage must pay it off in the correct order.
6. End each short episode with a real consequence or supported cliffhanger, never empty hype or invented facts.
7. Use natural spoken Myanmar endings such as တယ်, ပါတယ်, ရဲ့, တော့; avoid formal textbook narration and subtitle-like fragments.""",
	},
	"short_drama_translation": {
	    "label": "★ Chinese Short-Drama Translation / Dialogue Dubbing",
	    "label_mm": "★ တရုတ် Short-Drama ဘာသာပြန် + Dialogue အသံခွဲ",
	    "voice": """STYLE RULES (source-grounded short-drama localization for Dialogue mode):
1. Preserve important dialogue turns in chronological order and translate their meaning into natural spoken {target_lang}; do not replace the scene with a generic summary.
2. Separate connective explanation as [NARRATOR] and character lines as [MALE] or [FEMALE]. When the transcript contains a real exchange and the visual shows two distinguishable speakers, use the two character tags consistently instead of converting the whole exchange into narration. Use [NARRATOR] only for connective explanation or speech with no grounded speaker.
3. For every source speech cluster with two or more turns, include a compact back-and-forth of at least two tagged character blocks when supported. Preserve questions, answers, refusals, threats, confessions, emotional reactions, and consequences. Compress greetings and repeated filler.
4. Keep every spoken line tied to its source timestamp and visual exchange. Never place later dialogue over earlier footage.
5. Write localization, not literal machine translation: preserve intent, relationship, tension, and emotional force using natural Burmese speech.
6. Narrator lines must bridge the exchange and explain what changed; character lines must sound like a real conversation rather than isolated subtitles.
7. Do not speak the control tags [NARRATOR], [MALE], or [FEMALE].""",
	},
}
RECAP_STYLE_DEFAULT = "cinematic"
RECAP_STYLE_FEATURED = {
    "cinematic", "hook", "hybrid", "cdrama", "investigator", "immersive",
    "emotional", "thriller_suspense", "action", "plot_explainer",
    "dialogue_cinematic", "dialogue_banter", "visual_hook", "short_drama_translation"
}

def _next_gemini_client():
    """Rotate to the next API key. Returns the client."""
    global _gemini_idx, gemini_client
    if len(_GEMINI_KEYS) <= 1:
        return gemini_client
    _gemini_idx = (_gemini_idx + 1) % len(_GEMINI_KEYS)
    gemini_client = genai.Client(api_key=_GEMINI_KEYS[_gemini_idx])
    print(f"[GEMINI] Rotated to key #{_gemini_idx + 1}/{len(_GEMINI_KEYS)}")
    return gemini_client

def _all_keys_exhausted():
    """True when every loaded API key has hit 429 today."""
    return len(_exhausted_keys) >= len(_GEMINI_KEYS) and len(_GEMINI_KEYS) > 0

def get_gemini_model(purpose="default"):
    """Return (client, model_name) for the given purpose ('recap', 'default',
    or 'tts'). Walks that purpose's model cascade — call_gemini_with_retry
    advances the cascade position when Google retires the current model."""
    models = _MODEL_CASCADES.get(purpose, _MODEL_CASCADES["default"])
    if _all_keys_exhausted():
        # The last entry in each cascade is the cheapest/highest-quota model —
        # best bet when every key's main quota is used up for the day.
        print(f"[GEMINI] All keys exhausted — falling back to {models[-1]}")
        return gemini_client, models[-1]
    idx = min(_cascade_pos.get(purpose, 0), len(models) - 1)
    return gemini_client, models[idx]

_exhausted_keys = set()  # keys that hit 429 today


class GeminiBillingError(RuntimeError):
    """A permanent account/project billing or prepaid-credit failure.

    Retrying, rotating models, or changing analysis windows cannot repair a
    402 billing state. Keeping this distinct prevents misleading high-demand
    retry messages and lets the UI give the user the real next action.
    """


def _is_gemini_billing_error(message):
    text = str(message or '').lower()
    return (
        ('402' in text and ('resource_exhausted' in text or 'payment' in text or 'billing' in text))
        or 'prepayment credits are depleted' in text
        or 'prepaid credits are depleted' in text
        or 'prepayment credit' in text and ('depleted' in text or 'exhausted' in text)
    )

def call_gemini_with_retry(fn, video_id=None, part_tag="", label="AI", max_attempts=5, purpose=None, file_recovery_fn=None):
    """
    Gemini returns 503 UNAVAILABLE / 429 RESOURCE_EXHAUSTED when Google's
    servers are under high demand.  On 429 we rotate to the next API key
    (each Google account gets 50 req/day free).  When all keys are exhausted
    the caller should fall back to flash-lite (separate, larger quota).

    When Google retires the current model entirely (404 "no longer
    available"/NOT_FOUND — this has happened abruptly to several Gemini
    models through 2026), and `purpose` is given, automatically advances to
    the next model in that purpose's cascade and retries right away instead
    of failing the whole job.

    Gemini's File API scopes an uploaded file to the API key/project that
    uploaded it - if `fn` references an uploaded video file and we rotate
    to a different key mid-job, that file becomes inaccessible to the new
    key ("403 PERMISSION_DENIED ... may not exist"), even though the file
    is fine. When `file_recovery_fn` is given, this re-uploads under the
    now-active key and retries, instead of failing outright.
    """
    delays   = [5, 15, 30, 60, 90, 180, 300, 300, 300]
    last_err = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:
            msg          = str(e)
            low_msg = msg.lower()
            if _is_gemini_billing_error(msg):
                clear = (
                    "Gemini API billing error: prepaid credits are depleted "
                    "(HTTP 402 RESOURCE_EXHAUSTED). Add/restore billing credits "
                    "for the configured Google project, or use Manual mode; "
                    "retrying or changing the model cannot fix this."
                )
                if video_id:
                    log_status(video_id, f"❌ {part_tag}{label}: {clear}")
                raise GeminiBillingError(clear) from e
            empty_response = any(k in low_msg for k in ["response.text is none", "empty structured response", "empty json text", "no usable"])
            malformed_json = any(k in low_msg for k in ["unterminated string", "expecting value", "expecting ',' delimiter", "malformed or truncated json"])
            transient    = empty_response or malformed_json or any(k in msg for k in ["503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "500", "overloaded", "INTERNAL"])
            retired      = "404" in msg and ("NOT_FOUND" in msg or "no longer available" in msg)
            file_missing = "PERMISSION_DENIED" in msg and ("may not exist" in msg or "403" in msg)
            last_err     = e
            # On 429: rotate key so next attempt uses a fresh quota bucket.
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                _exhausted_keys.add(_GEMINI_KEYS[_gemini_idx] if _GEMINI_KEYS else "")
                if len(_GEMINI_KEYS) > 1:
                    _next_gemini_client()
                    print(f"[Gemini] Key exhausted, rotated to next (attempt {attempt+1})")
            if file_missing and file_recovery_fn:
                if video_id:
                    log_status(video_id, f"{part_tag}{label}: video file ကို key အသစ်နဲ့ ပြန် upload လုပ်နေပါသည်...")
                try:
                    file_recovery_fn()
                    continue  # retry right away with the freshly uploaded file
                except Exception as reup_err:
                    print(f"[Gemini] File re-upload failed: {reup_err}")
                    # fall through to normal error handling below
            if retired and purpose:
                new_model = _advance_model_cascade(purpose)
                if new_model:
                    if video_id:
                        log_status(video_id, f"{part_tag}{label}: Google က model ကို ရုတ်သိမ်းလိုက်လို့ {new_model} ကို auto ပြောင်းသုံးနေပါသည်...")
                    else:
                        print(f"[Gemini] {label}: model retired, switching to {new_model}")
                    continue  # retry right away with the new model, no backoff needed
                # A 404 is permanent for this model; retrying it only adds
                # delay and produces the same error. Let the caller decide
                # whether a local fallback is safe.
                raise
            # 503/500 high-demand outages are not fixed by retrying the same
            # model forever. Advance the purpose cascade immediately, then use
            # bounded backoff only after the cascade has been tried.
            if transient and purpose and (empty_response or malformed_json or any(k in msg for k in ["503", "UNAVAILABLE", "500", "overloaded", "INTERNAL"])):
                new_model = _advance_model_cascade(purpose)
                if new_model:
                    if video_id:
                        log_status(video_id, f"{part_tag}{label}: 503/high-demand ဖြစ်လို့ {new_model} သို့ model ပြောင်းပြီး ပြန်ကြိုးစားနေပါသည်...")
                    else:
                        print(f"[Gemini] {label}: transient outage, switching to {new_model}")
                    continue
            if (not transient and not retired and not file_missing) or attempt == max_attempts - 1:
                raise
            wait = delays[min(attempt, len(delays) - 1)]
            if video_id:
                log_status(video_id, f"⏳ {part_tag}{label} — Google AI server အလုပ်များနေပါသည် (high demand)။ {wait} စက္ကန့်စောင့်ပြီး ပြန်ကြိုးစားနေပါသည် ({attempt+1}/{max_attempts})...")
            else:
                print(f"[Gemini Retry] {label} transient error, retrying in {wait}s (attempt {attempt+1}/{max_attempts}): {msg[:150]}")
            time.sleep(wait)
    raise last_err


def _safe_response_json(response):
    """Read JSON from a GenAI response, tolerating a truncated JSON array.

    The current SDK may expose structured output through ``parsed`` while
    some blocked/empty responses have ``text is None``. Both cases must be
    handled explicitly so local-fast reports a useful error instead of the
    low-level ``json.loads(None)`` TypeError.
    """
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, (list, dict)):
        return parsed
    text = parsed if isinstance(parsed, str) else getattr(response, "text", None)
    if text is None:
        raise ValueError("Gemini returned no JSON text (response.text is None; possible safety block, empty response, or quota issue)")
    text = str(text).strip().replace("```json", "").replace("```", "").strip()
    if not text:
        raise ValueError("Gemini returned empty JSON text (possible safety block, empty response, or quota issue)")
    try:
        return json.loads(text)
    except json.JSONDecodeError as original:
        # Free-tier responses can be cut off after several complete objects.
        # Recover only complete objects; never invent or close a partial string.
        start = text.find('[')
        if start >= 0:
            decoder = json.JSONDecoder()
            pos = start + 1
            recovered = []
            while pos < len(text):
                while pos < len(text) and text[pos].isspace():
                    pos += 1
                if pos < len(text) and text[pos] == ',':
                    pos += 1
                    continue
                if pos < len(text) and text[pos] == ']':
                    break
                try:
                    item, end = decoder.raw_decode(text, pos)
                except json.JSONDecodeError:
                    break
                if isinstance(item, dict):
                    recovered.append(item)
                pos = end
            if recovered:
                return recovered
        raise ValueError(f"Gemini returned malformed or truncated JSON: {original.msg}") from original

# ==========================================
# 3. HELPERS
# ==========================================
def format_timestamp(s: float):
    h = int(s // 3600); m = int((s % 3600) // 60)
    sec = int(s % 60);  ms = int((s - int(s)) * 1000)
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"

def parse_time_to_sec(time_str):
    if time_str is None or str(time_str).strip() == "":
        raise ValueError("timestamp is empty")
    raw = str(time_str).strip().replace(',', '.')
    try:
        parts = raw.split(':')
        if len(parts) == 3:
            value = int(parts[0])*3600 + int(parts[1])*60 + float(parts[2])
        elif len(parts) == 2:
            value = int(parts[0])*60 + float(parts[1])
        elif len(parts) == 1:
            value = float(raw)
        else:
            raise ValueError("expected SS, MM:SS, or HH:MM:SS")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid timestamp {time_str!r}") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"invalid timestamp {time_str!r}")
    return value


# TTS pacing helpers.  Keep the value used for Gemini's word budget and the
# value sent to Edge-TTS in one place; otherwise values such as 1.4 and 140
# can silently produce different script/audio targets.
def normalize_tts_rate(raw_rate, default=1.0):
    try:
        rate = float(raw_rate)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(rate):
        return float(default)
    # Accept both UI multiplier form (1.4) and percentage form (140).
    if rate > 10.0:
        rate /= 100.0
    return max(0.5, min(2.0, rate))


def edge_tts_rate_string(raw_rate):
    rate = normalize_tts_rate(raw_rate)
    percent = int(round((rate - 1.0) * 100.0))
    return f"{percent:+d}%"


# Edge TTS exposes two Burmese neural voices, so these are delivery profiles
# rather than claims of twenty different trained speakers.  Each profile keeps
# a real Edge voice ID and applies a small, bounded rate/pitch offset.  This
# gives the Dialogue tool consistent character identities without pretending
# that pitch shifting is a new human voice model.
EDGE_BURMESE_PROFILES = {
    "narrator_calm": {"label": "Narrator · Calm", "voice": "my-MM-NilarNeural", "rate": 0, "pitch": 0},
    "narrator_story": {"label": "Narrator · Storyteller", "voice": "my-MM-NilarNeural", "rate": -3, "pitch": 1},
    "female_soft": {"label": "မသိမ့် · Soft Female", "voice": "my-MM-NilarNeural", "rate": -4, "pitch": 3},
    "female_young": {"label": "မိန်းကလေး · Young Female", "voice": "my-MM-NilarNeural", "rate": 2, "pitch": 6},
    "female_strong": {"label": "မင်းသမီး · Strong Female", "voice": "my-MM-NilarNeural", "rate": -1, "pitch": -1},
    "female_sad": {"label": "မိခင် · Emotional Female", "voice": "my-MM-NilarNeural", "rate": -9, "pitch": -2},
    "male_calm": {"label": "အောင် · Calm Male", "voice": "my-MM-ThihaNeural", "rate": -3, "pitch": -3},
    "male_deep": {"label": "မင်းသား · Deep Male", "voice": "my-MM-ThihaNeural", "rate": -6, "pitch": -7},
    "male_hero": {"label": "သူရဲကောင်း · Hero Male", "voice": "my-MM-ThihaNeural", "rate": 1, "pitch": -2},
    "male_urgent": {"label": "အရေးပေါ် · Urgent Male", "voice": "my-MM-ThihaNeural", "rate": 8, "pitch": 0},
    "child_girl": {"label": "ကလေးမ · Child Girl", "voice": "my-MM-NilarNeural", "rate": 8, "pitch": 12},
    "child_boy": {"label": "ကလေးသား · Child Boy", "voice": "my-MM-ThihaNeural", "rate": 10, "pitch": 9},
    "teen_girl": {"label": "ဆယ်ကျော်သက်မ · Teen Girl", "voice": "my-MM-NilarNeural", "rate": 7, "pitch": 8},
    "teen_boy": {"label": "ဆယ်ကျော်သက်သား · Teen Boy", "voice": "my-MM-ThihaNeural", "rate": 6, "pitch": 5},
    "elder_female": {"label": "အဘွား · Elder Female", "voice": "my-MM-NilarNeural", "rate": -10, "pitch": -7},
    "elder_male": {"label": "အဘိုး · Elder Male", "voice": "my-MM-ThihaNeural", "rate": -12, "pitch": -10},
    "villain": {"label": "လူကြမ်း · Villain", "voice": "my-MM-ThihaNeural", "rate": -5, "pitch": -12},
    "comic": {"label": "ဟာသ · Comic", "voice": "my-MM-ThihaNeural", "rate": 8, "pitch": 6},
    "whisper": {"label": "လျှို့ဝှက် · Whisper-like", "voice": "my-MM-NilarNeural", "rate": -8, "pitch": 2},
    "energetic": {"label": "တက်ကြွ · Energetic", "voice": "my-MM-NilarNeural", "rate": 10, "pitch": 4},
    # Fast, fluent storyteller delivery for recap narration (rate/pitch variants of the two real Edge voices)
    "narrator_fast_f": {"label": "Narrator · Fast Storyteller (မ)", "voice": "my-MM-NilarNeural", "rate": 16, "pitch": 3},
    "narrator_fast_m": {"label": "Narrator · Fast Storyteller (ကျား)", "voice": "my-MM-ThihaNeural", "rate": 16, "pitch": -2},
    "narrator_hype_f": {"label": "Narrator · Hype Recap (မ)", "voice": "my-MM-NilarNeural", "rate": 24, "pitch": 5},
    "narrator_hype_m": {"label": "Narrator · Hype Recap (ကျား)", "voice": "my-MM-ThihaNeural", "rate": 24, "pitch": 1},
    "narrator_gossip": {"label": "Narrator · တတ်တတ်ကျွကျွ ပြောပြ (မ)", "voice": "my-MM-NilarNeural", "rate": 20, "pitch": 7},
}


def resolve_edge_voice_profile(voice_id, raw_rate=1.0, raw_pitch=0):
    """Resolve a profile ID to a real Edge voice plus bounded delivery values."""
    profile_id = str(voice_id or "").strip()
    profile_id = profile_id[7:] if profile_id.startswith("profile:") else profile_id
    profile = EDGE_BURMESE_PROFILES.get(profile_id)
    base_rate = normalize_tts_rate(raw_rate)
    try:
        base_pitch = int(float(raw_pitch or 0))
    except (TypeError, ValueError):
        base_pitch = 0
    if not profile:
        return str(voice_id or "my-MM-NilarNeural"), base_rate, base_pitch
    return (
        profile["voice"],
        max(0.5, min(2.0, base_rate + (float(profile.get("rate", 0)) / 100.0))),
        max(-20, min(20, base_pitch + int(profile.get("pitch", 0)))),
    )


def estimate_tts_wpm(raw_rate, base_wpm=130.0):
    """Estimate spoken WPM conservatively for the Gemini word budget.

    Edge-TTS rate settings are not perfectly linear for Burmese because
    punctuation and pause timing remain.  A 0.80 linearity factor avoids
    over-requesting words on the first pass; actual audio duration remains the
    final authority after synthesis.
    """
    rate = normalize_tts_rate(raw_rate)
    effective_factor = 1.0 + 0.80 * (rate - 1.0)
    return max(60.0, float(base_wpm) * effective_factor)


_tts_calibration_cache = {}
_tts_calibration_lock = threading.Lock()


def calibrate_tts_wpm(voice_type, voice_id, raw_rate, pitch=0):
    """Measure the selected local TTS voice once, then reuse its real WPM.

    Word-count estimates are especially unreliable for Burmese because pauses,
    punctuation, and Edge-TTS rate percentages are not perfectly linear. This
    calibration never calls Gemini; it synthesizes a short local sample and
    measures the resulting file with ffprobe. If calibration fails, the
    conservative formula remains available so production never depends on a
    calibration request succeeding.
    """
    kind = str(voice_type or 'edge').strip().lower()
    if not tts_rate_affects_duration(kind, voice_id):
        return 130.0, 'backend_rate_not_applied'
    if kind not in ('edge', 'old', ''):
        return estimate_tts_wpm(raw_rate), 'formula_non_edge'
    rate = normalize_tts_rate(raw_rate)
    try:
        pitch_value = int(float(pitch or 0))
    except (TypeError, ValueError):
        pitch_value = 0
    cache_key = (kind, str(voice_id or ''), round(rate, 3), pitch_value)
    with _tts_calibration_lock:
        cached = _tts_calibration_cache.get(cache_key)
    if cached:
        return cached, 'measured_cache'

    # 100 whitespace-delimited Burmese spoken units: the same counting rule
    # used by the script validator, without sending any movie content.
    sample = (
        'ဒီ ဇာတ်လမ်းမှာ အဓိက ဇာတ်ကောင်က အရေးကြီးတဲ့ ဆုံးဖြတ်ချက်တစ်ခုကို ချရပါတယ်။ '
        'သူ့ရဲ့ ရည်ရွယ်ချက်ကို အကောင်အထည်ဖော်ဖို့ အခက်အခဲတွေကို ရင်ဆိုင်ရပြီး '
        'ဖြစ်ရပ်တစ်ခုချင်းစီက နောက်ထပ် ပြဿနာအသစ်တွေကို ဖြစ်စေပါတယ်။ '
    ) * 5
    word_count = len(re.findall(r'\S+', sample))
    fd, sample_path = tempfile.mkstemp(prefix='tts_calibration_', suffix='.mp3')
    os.close(fd)
    try:
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(generate_audio_only(sample, kind or 'edge', voice_id, sample_path, rate, pitch_value))
        finally:
            loop.close()
        measured = float(probe_media(sample_path).get('audio_duration') or 0.0)
        if measured <= 0.1 or word_count <= 0:
            raise RuntimeError('calibration audio duration is empty')
        measured_wpm = (word_count / measured) * 60.0
        # Reject pathological provider responses; use formula instead.
        if not math.isfinite(measured_wpm) or not 40.0 <= measured_wpm <= 320.0:
            raise RuntimeError(f'calibration WPM out of range: {measured_wpm!r}')
        with _tts_calibration_lock:
            _tts_calibration_cache[cache_key] = measured_wpm
        return measured_wpm, 'measured'
    except Exception as exc:
        print(f'[TTS CALIBRATION] fallback to formula: {exc}')
        return estimate_tts_wpm(rate), 'formula_fallback'
    finally:
        try:
            if os.path.exists(sample_path):
                os.remove(sample_path)
        except OSError:
            pass


def tts_rate_affects_duration(voice_type, voice_id=None):
    """Whether the selected backend actually applies the speed setting."""
    kind = str(voice_type or "edge").strip().lower()
    if kind == "new" or str(voice_id or "").startswith("gemini:"):
        return False  # Gemini native audio branch has no rate parameter here.
    if kind == "clone":
        return False  # voice_clone.generate_speech_long currently ignores rate.
    return True

def probe_media(path):
    """Return duration, stream types, FPS, and audio sample rate for QA."""
    data = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", path
    ]).decode("utf-8", errors="replace"))
    streams = data.get("streams") or []
    fmt = data.get("format") or {}
    def _num(v, default=0.0):
        try: return float(v)
        except (TypeError, ValueError): return default
    video = next((x for x in streams if x.get("codec_type") == "video"), {})
    audio = next((x for x in streams if x.get("codec_type") == "audio"), {})
    format_duration = _num(fmt.get("duration"), 0.0)
    video_duration = _num(video.get("duration"), format_duration)
    audio_duration = _num(audio.get("duration"), format_duration)
    return {
        "duration": format_duration or max(video_duration, audio_duration),
        "video_duration": video_duration,
        "audio_duration": audio_duration,
        "has_video": bool(video), "has_audio": bool(audio),
        "width": int(video.get("width") or 0), "height": int(video.get("height") or 0),
        "fps": _num(str(video.get("r_frame_rate", "0/1")).split("/")[0], 0.0) /
               max(1.0, _num(str(video.get("r_frame_rate", "0/1")).split("/")[-1], 1.0)),
        "sample_rate": int(audio.get("sample_rate") or 0),
    }

def validate_scene_timeline(scenes, source_duration):
    """Validate Gemini/local scene ranges before TTS and rendering."""
    errors, warnings, ranges = [], [], []
    prev_end = -0.05
    for i, scene in enumerate(scenes or []):
        try:
            start_raw, end_raw = scene.get("start_time"), scene.get("end_time")
            if start_raw is None or end_raw is None:
                raise ValueError("start_time/end_time missing")
            start, end = parse_time_to_sec(start_raw), parse_time_to_sec(end_raw)
            if start < -0.05 or end > source_duration + 0.05:
                errors.append(f"scene {i+1}: range {start:.3f}-{end:.3f}s outside source {source_duration:.3f}s")
            if end <= start + 0.05:
                errors.append(f"scene {i+1}: end_time must be after start_time")
            if start < prev_end - 0.05:
                errors.append(f"scene {i+1}: overlaps the previous source range")
            if end - start < 0.20:
                warnings.append(f"scene {i+1}: very short source range ({end-start:.2f}s)")
            ranges.append((max(0.0, start), min(source_duration, end)))
            prev_end = end
        except Exception as e:
            errors.append(f"scene {i+1}: invalid timestamp ({e})")
    return {"errors": errors, "warnings": warnings, "ranges": ranges}

def run_cmd(cmd, step_name, video_id, env=None):
    run_env = env if env else os.environ.copy()
    res = subprocess.run(cmd, capture_output=True, env=run_env)
    if res.returncode != 0:
        err = res.stderr.decode('utf-8', errors='replace')
        log_status(video_id, f"❌ Error in {step_name}")
        print(f"--- FFmpeg ERROR [{step_name}] ---\n{err[-2000:]}")
        raise Exception(f"FFmpeg Error: {step_name}")
    else:
        for line in res.stderr.decode('utf-8', errors='replace').split('\n'):
            if any(k in line.lower() for k in ['fontselect','loading font','fallback','cannot find font']):
                print(f"  [LIBASS] {line.strip()}")

async def _run_edge_tts(text, v_id, rate_str, pitch_str, audio_path):
    """
    Runs edge-tts and captures WordBoundary events, which give real
    per-word start/end timestamps (in seconds, local to this audio clip).
    This lets subtitles be built from exact word timing instead of an
    estimated character-proportional split.
    """
    import asyncio as _aio
    # Edge can occasionally close a concurrent request without returning an
    # audio chunk. Treat that as a retryable provider failure, not a valid file.
    attempts = [(v_id, rate_str, pitch_str),
                (v_id, "+0%", "+0Hz"),
                ("my-MM-NilarNeural", "+0%", "+0Hz")]
    last_error = None
    for attempt_no, (try_voice, try_rate, try_pitch) in enumerate(attempts, 1):
        word_timings, audio_bytes = [], 0
        tmp_path = f"{audio_path}.attempt{attempt_no}.tmp"
        try:
            communicate = edge_tts.Communicate(text, try_voice, rate=try_rate, pitch=try_pitch)
            async def _collect():
                nonlocal audio_bytes
                with open(tmp_path, "wb") as f:
                    async for chunk in communicate.stream():
                        if chunk.get("type") == "audio" and chunk.get("data"):
                            f.write(chunk["data"])
                            audio_bytes += len(chunk["data"])
                        elif chunk.get("type") == "WordBoundary":
                            word_timings.append({
                                "text": chunk.get("text", ""),
                                "start": chunk.get("offset", 0) / 10_000_000,
                                "end": (chunk.get("offset", 0) + chunk.get("duration", 0)) / 10_000_000,
                            })
            await _aio.wait_for(_collect(), timeout=60)
            if audio_bytes < 256 or not os.path.exists(tmp_path) or os.path.getsize(tmp_path) < 256:
                raise RuntimeError("No audio was received")
            os.replace(tmp_path, audio_path)
            if attempt_no > 1:
                print(f"[EDGE-TTS] recovered on attempt {attempt_no} with voice={try_voice}, rate={try_rate}")
            return word_timings
        except _aio.TimeoutError as exc:
            last_error = RuntimeError(f"Edge-TTS timeout on attempt {attempt_no}")
            print(f"[EDGE-TTS] Timeout after 60s for voice={try_voice}, attempt={attempt_no}")
        except Exception as exc:
            last_error = exc
            print(f"[EDGE-TTS] attempt {attempt_no} failed for voice={try_voice}: {exc}")
        finally:
            try:
                if os.path.exists(tmp_path): os.remove(tmp_path)
            except OSError:
                pass
        if attempt_no < len(attempts):
            await _aio.sleep(1.0 * attempt_no)
    raise RuntimeError(f"Edge-TTS produced no usable audio after {len(attempts)} attempts: {last_error}")

async def generate_audio_only(text, voice_type, voice_id, audio_path, rate, pitch):
    text = text.strip()
    if not text:
        raise ValueError("Text is empty.")

    # Global timeout: if ANY voice generation hangs for >90s, raise error
    async def _do_generate():
        # Valid Gemini TTS prebuilt voice names (native audio-out models only)
        GEMINI_VOICES = [
            'Zephyr','Puck','Charon','Kore','Fenrir','Leda','Orus','Aoede',
            'Callirrhoe','Autonoe','Enceladus','Iapetus','Umbriel','Algieba',
            'Despina','Erinome','Algenib','Rasalgethi','Laomedeia','Achernar',
            'Alnilam','Schedar','Gacrux','Pulcherrima','Achird','Zubenelgenubi',
            'Vindemiatrix','Sadachbia','Sadaltager','Sulafat',
        ]
        word_timings = None

        if voice_type == 'new' or (voice_id and voice_id.startswith('gemini:')):
            voice_name = voice_id.replace('gemini:', '') if voice_id else 'Aoede'
            if voice_name not in GEMINI_VOICES:
                voice_name = 'Aoede'

            def _run_gemini():
                if not gemini_client:
                    raise Exception("GOOGLE_API_KEY not configured")
                def _call_tts():
                    _tc, _tm = get_gemini_model("tts")
                    return _tc.models.generate_content(
                        model=_tm,
                        contents=text,
                        config=genai_types.GenerateContentConfig(
                            response_modalities=["AUDIO"],
                            speech_config=genai_types.SpeechConfig(
                                voice_config=genai_types.VoiceConfig(
                                    prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(voice_name=voice_name)
                                )
                            ),
                        )
                    )
                response = call_gemini_with_retry(
                    _call_tts,
                    label="Voice Generation", purpose="tts"
                )
                raw = audio_path.replace('.mp3', '.raw')
                with open(raw, 'wb') as f:
                    f.write(response.candidates[0].content.parts[0].inline_data.data)
                subprocess.run(["ffmpeg","-y","-f","s16le","-ar","24000","-ac","1","-i",raw,"-c:a","libmp3lame",audio_path],
                               check=True, capture_output=True)

            await asyncio.to_thread(_run_gemini)

        elif voice_type == 'edge':
            v_id, resolved_rate, resolved_pitch = resolve_edge_voice_profile(
                voice_id or "my-MM-NilarNeural", rate, pitch
            )
            rate_str  = edge_tts_rate_string(resolved_rate)
            pitch_str = f"+{int(resolved_pitch)}Hz" if int(resolved_pitch) >= 0 else f"{int(resolved_pitch)}Hz"
            word_timings = await _run_edge_tts(text, v_id, rate_str, pitch_str, audio_path)

        elif voice_type == 'clone':
            import voice_clone
            profile_id = int(voice_id) if voice_id and voice_id.isdigit() else 0
            if not profile_id:
                raise Exception("Voice clone profile not found")
            profile = voice_clone.get_profile(profile_id)
            if not profile:
                raise Exception(f"Voice profile {profile_id} not found")
            def _run_clone():
                audio_bytes, ct = voice_clone.generate_speech_long(text, profile_id)
                with open(audio_path, "wb") as f:
                    f.write(audio_bytes)
            await asyncio.to_thread(_run_clone)

        elif voice_type == 'old':
            # Legacy option → redirect to edge-tts
            lang = voice_id if voice_id else 'my'
            lang_voice_map = {'my': 'my-MM-NilarNeural', 'en': 'en-US-AriaNeural',
                              'th': 'th-TH-PremwadeeNeural', 'zh': 'zh-CN-XiaoxiaoNeural',
                              'ja': 'ja-JP-NanamiNeural', 'ko': 'ko-KR-SunHiNeural',
                              'hi': 'hi-IN-SwaraNeural', 'es': 'es-ES-ElviraNeural',
                              'id': 'id-ID-GadisNeural', 'vi': 'vi-VN-HoaiMyNeural'}
            v_id = lang_voice_map.get(lang, 'my-MM-NilarNeural')
            v_id, resolved_rate, resolved_pitch = resolve_edge_voice_profile(v_id, rate, pitch)
            rate_str = edge_tts_rate_string(resolved_rate)
            pitch_str = f"+{int(resolved_pitch)}Hz" if int(resolved_pitch) >= 0 else f"{int(resolved_pitch)}Hz"
            word_timings = await _run_edge_tts(text, v_id, rate_str, pitch_str, audio_path)

        else:
            v_id      = "my-MM-NilarNeural"
            rate_str  = edge_tts_rate_string(rate)
            pitch_str = f"+{int(pitch)}Hz" if int(pitch) >= 0 else f"{int(pitch)}Hz"
            word_timings = await _run_edge_tts(text, v_id, rate_str, pitch_str, audio_path)

        if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 256:
            raise RuntimeError("TTS provider returned no usable audio file")
        return word_timings

    try:
        return await asyncio.wait_for(_do_generate(), timeout=90)
    except asyncio.TimeoutError:
        print(f"[VOICE] Generation timed out after 90s (type={voice_type})")
        raise Exception(f"Voice generation timed out after 90 seconds")
    except Exception as e:
        print(f"[VOICE] Generation failed (type={voice_type}): {e}")
        raise

# ==========================================
# 3b. FONT HELPERS
# ==========================================
def get_real_font_name(font_path: str) -> str:
    try:
        import struct
        with open(font_path, 'rb') as f:
            data = f.read()
        num_tables  = struct.unpack('>H', data[4:6])[0]
        name_offset = None
        for i in range(num_tables):
            s = 12 + i * 16
            if data[s:s+4] == b'name':
                name_offset = struct.unpack('>I', data[s+8:s+12])[0]
                break
        if name_offset is None:
            raise ValueError("no name table")
        count        = struct.unpack('>H', data[name_offset+2:name_offset+4])[0]
        str_off_base = struct.unpack('>H', data[name_offset+4:name_offset+6])[0]
        storage      = name_offset + str_off_base
        names = {}
        for i in range(count):
            r   = name_offset + 6 + i * 12
            pid = struct.unpack('>H', data[r:r+2])[0]
            nid = struct.unpack('>H', data[r+6:r+8])[0]
            lng = struct.unpack('>H', data[r+8:r+10])[0]
            off = struct.unpack('>H', data[r+10:r+12])[0]
            if nid not in [1, 4]:
                continue
            raw = data[storage+off : storage+off+lng]
            try:
                d = raw.decode('utf-16-be') if pid == 3 else raw.decode('latin-1')
                d = d.strip()
                if d and nid not in names:
                    names[nid] = d
            except:
                pass
        for nid in [1, 4]:
            if names.get(nid):
                return names[nid]
    except Exception as e:
        pass
    return os.path.splitext(os.path.basename(font_path))[0]

def srt_to_ass(srt_path, ass_path, font_path, font_size, color_hex, margin_v):
    font_name = get_real_font_name(font_path)
    with open(srt_path, encoding='utf-8') as f:
        raw = f.read()
    blocks = re.findall(
        r'\d+\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n([\s\S]*?)(?=\n\n|$)',
        raw.strip()
    )
    def ts(t):
        h,m,sms = t.split(':'); s,ms = sms.split(',')
        return f"{int(h)}:{m}:{s}.{int(ms)//10:02d}"
    header = (
        "[Script Info]\nScriptType: v4.00+\n\n"
        "[V4+ Styles]\n"
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,"
        "BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,"
        "BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding\n"
        f"Style: Default,{font_name},{font_size},{color_hex},&H000000FF,"
        f"&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,2,1,2,10,10,{margin_v},1\n\n"
        "[Events]\nFormat: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n"
    )
    lines = [
        "Dialogue: 0,{},{},Default,,0,0,0,,{}".format(ts(s), ts(e), t.strip().replace('\n', '\\N'))
        for s, e, t in blocks
    ]
    with open(ass_path, 'w', encoding='utf-8') as f:
        f.write(header + '\n'.join(lines) + '\n')

# ==========================================
# 3c. SUBTITLE LANGUAGE TRANSLATION
#     Lets subtitles be in a different language than the narration
#     (e.g. Burmese voice + English subtitles). Translates the built
#     SRT text lines with Gemini, in chunks, keeping timing intact.
# ==========================================
def translate_srt_blocks(blocks, target_language, video_id=None, chunk_size=50):
    if not gemini_client:
        raise Exception("GOOGLE_API_KEY not configured")
    from google.genai import types as genai_types
    parsed = [b.split('\n', 2) for b in blocks]
    texts  = [p[2] if len(p) > 2 else '' for p in parsed]
    out    = list(texts)
    for start in range(0, len(texts), chunk_size):
        chunk = texts[start:start + chunk_size]
        prompt = (
            f"Translate these {len(chunk)} movie-recap subtitle lines into {target_language}. "
            "Keep each translation short, natural and spoken-style. Do NOT merge or split "
            "lines, do NOT number them. Return ONLY a JSON array of translated strings in "
            "the same order with the same length.\n\n" +
            "\n".join(chunk)
        )
        def _call_translate():
            _tc, _tm = get_gemini_model("default")
            return _tc.models.generate_content(
                model=_tm,
                contents=prompt + "\n\nReturn ONLY valid JSON array. No markdown, no explanation.",
                config=genai_types.GenerateContentConfig(
                    response_mime_type="application/json",
                    max_output_tokens=16384,
                ),
            )
        response = call_gemini_with_retry(
            _call_translate,
            video_id=video_id, label="Subtitle Translation", purpose="default"
        )
        translated = _safe_response_json(response)
        if not isinstance(translated, list) or len(translated) != len(chunk):
            raise Exception("subtitle translation length mismatch")
        for j, t in enumerate(translated):
            out[start + j] = str(t)
    rebuilt = []
    for i, p in enumerate(parsed):
        idx_line = p[0] if len(p) > 0 else str(i + 1)
        timing   = p[1] if len(p) > 1 else ''
        rebuilt.append(f"{idx_line}\n{timing}\n{out[i]}\n")
    return rebuilt

# ==========================================
# 4. CORE PIPELINE
# ==========================================
def run_advanced_pipeline(video_id, local_path, logo_path, font_path, settings):
    """
    Orchestrator: Auto Gemini uses 30-minute output parts and bounded
    5-minute analysis windows; each part is TTS/rendered once and can be
    merged at the user's request. Local/legacy modes retain compatibility.
    """
    # The UI's "split_count" slider (5-50) previously did nothing on the
    # backend. It's now used as the target chunk length in minutes — how
    # long each auto-split part should be. Clamped to the slider's own
    # range as a safety net against bad/missing input.
    try:
        _auto_chunk = float(settings.get('auto_analysis_chunk_min', 0) or 0)
    except (TypeError, ValueError):
        _auto_chunk = 0.0
    try:
        chunk_minutes = float(settings.get('split_count', 25))
    except (TypeError, ValueError):
        chunk_minutes = 25.0
    if str(settings.get('analysis_engine', '')).strip().lower() == 'gemini_auto':
        # Auto output parts are fixed at 30 minutes. The actual Gemini
        # analysis window is 5 minutes and is handled inside the Auto
        # orchestrator, never as a published output part.
        chunk_minutes = 30.0
    else:
        chunk_minutes = max(5.0, min(50.0, chunk_minutes))
    CHUNK_LIMIT_SEC = chunk_minutes * 60

    try:
        _run_advanced_pipeline_inner(video_id, local_path, logo_path, font_path, settings, CHUNK_LIMIT_SEC)
    finally:
        # The full original upload (hundreds of MB to a few GB) was never
        # being deleted after the job finished - every recap left its
        # source video sitting on disk forever, which is exactly what
        # causes later uploads to start failing once disk fills up.
        try:
            if local_path and os.path.exists(local_path):
                os.remove(local_path)
        except Exception as e:
            print(f"[CLEANUP] Could not remove source video {local_path}: {e}")


def _maybe_generate_teaser(video_id, settings):
    """Best-effort: if the user enabled the Trailer/Teaser toggle, prepend
    a cold-open flash-montage onto the just-finished recap's own video, in
    place. MUST run after _maybe_generate_shorts (Shorts should always cut
    from the original, untouched recap timeline, not a teaser-shifted
    one). Never lets a failure here affect the main recap job's already-
    successful status."""
    if str(settings.get('make_teaser', '')).strip().lower() not in ('true', '1', 'on', 'yes'):
        return
    try:
        import shorts_reels
        num_flashes = int(settings.get('num_flashes', 6))
        target_lang = settings.get('target_lang', 'Burmese')
        log_status(video_id, f"🎞️ ရုပ်ရှင် trailer ပုံစံ Teaser Intro ဖန်တီးနေပါသည် (flash {num_flashes} ခု)...")
        result = shorts_reels.build_teaser_intro(video_id, num_flashes, target_lang)
        log_status(video_id, f"✅ Teaser Intro ({result['teaser_duration']}s) ကို Recap ရဲ့ အစမှာ ထည့်ပြီးပါပြီ။")
    except Exception as e:
        log_status(video_id, f"⚠️ Teaser Intro ဖန်တီးမှု မအောင်မြင်ပါ: {e} — Recap video ကတော့ ပုံမှန် ရရှိပါသည်။")


def _maybe_generate_shorts(video_id, settings):
    """Best-effort: if the user enabled the Shorts/Reels toggle, generate
    them from the just-finished recap's own output files (see
    shorts_reels.py's module docstring for why this needs no extra
    Gemini-video/Whisper cost). Never lets a Shorts failure affect the
    main recap job's already-successful status."""
    if str(settings.get('make_shorts', '')).strip().lower() not in ('true', '1', 'on', 'yes'):
        return
    try:
        import shorts_reels
        num_shorts  = int(settings.get('num_shorts', 3))
        target_lang = settings.get('target_lang', 'Burmese')
        log_status(video_id, f"🎬 Reels/Shorts {num_shorts} ခု ဖန်တီးနေပါသည်...")
        results = shorts_reels.run_shorts_generation(video_id, num_shorts, target_lang)
        log_status(video_id, f"✅ Shorts {len(results)} ခု အသင့်ဖြစ်ပါပြီ (Download Files ထဲတွင် ရယူနိုင်ပါသည်)။")
    except Exception as e:
        log_status(video_id, f"⚠️ Shorts/Reels ဖန်တီးမှု မအောင်မြင်ပါ: {e} — Recap video ကတော့ ပုံမှန် ရရှိပါသည်။")


def _load_continuation_context(settings):
    """If the user picked a past completed job to continue from
    (settings['continue_from_id']), read that job's saved _script.txt and
    return the same kind of narration tail normally carried between
    auto-split parts of the SAME job - so a manually separate 'Part 2'
    upload can still continue Part 1's story instead of starting cold.
    Handles a Part 1 that was itself auto-split into sub-parts by picking
    its LAST sub-part's script (highest _partNN, or _combined, or plain)."""
    cont_id = str(settings.get('continue_from_id', '')).strip()
    if not cont_id:
        return ""
    safe_id = re.sub(r'[^a-zA-Z0-9_-]', '', cont_id)[:64]
    if not os.path.isdir(DOWNLOAD_DIR):
        return ""
    matches = [f for f in os.listdir(DOWNLOAD_DIR)
               if f.startswith(safe_id) and f.endswith("_script.txt")]
    if not matches:
        print(f"[continuity] no saved script found for continue_from_id={cont_id}")
        return ""

    def _rank(fname):
        m = re.search(r'_part(\d+)_script\.txt$', fname)
        if m:
            return (0, int(m.group(1)))       # a numbered sub-part - higher number = later
        if fname.endswith("_combined_script.txt"):
            return (1, 0)                      # combined file, if one ever exists
        return (0, 0)                          # plain single-job script
    fname = sorted(matches, key=_rank)[-1]

    try:
        with open(os.path.join(DOWNLOAD_DIR, fname), "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        # Each line is "[start - end] script" - pull just the script
        # portion from the last couple of lines.
        scripts = []
        for line in lines[-2:]:
            m = re.match(r'^\[.*?\]\s*(.*)$', line)
            scripts.append(m.group(1) if m else line)
        return " ".join(scripts)[-600:]
    except Exception as e:
        print(f"[continuity] failed to read {fname}: {e}")
        return ""


def polish_recap_script(scenes, target_lang, recap_style, video_id, part_tag=""):
    """Editorial pass: improve narration only; never alter source mapping.

    This deliberately receives text/timestamps, not the video, so it cannot
    invent new visual evidence or change the renderer's timeline. On failure
    the original source-grounded script is returned unchanged.
    """
    if not gemini_client or not scenes:
        return scenes
    style = RECAP_STYLES.get(recap_style, RECAP_STYLES[RECAP_STYLE_DEFAULT])
    payload = [{"index": i, "start_time": s.get("start_time"), "end_time": s.get("end_time"), "script": s.get("script", "")} for i, s in enumerate(scenes)]
    prompt = f"""You are the final editor for a {target_lang} movie-recap narration.
Rewrite the supplied lines into a smooth, natural, cinematic story while preserving the exact chronology and every index.
STYLE: {style.get('label', recap_style)}
Rules:
- Return ONLY a JSON array with exactly one object for every input index: {{"index": integer, "script": string}}.
- Do not add, remove, reorder, merge, or split entries. Do not change timestamps.
- Preserve every supported fact and action. Do not invent motives, dialogue, names, relationships, twists, or events.
- Remove generic filler, repeated hooks, contradictory pronouns, literal-translation phrasing, and textbook/formal Burmese endings.
- Use natural spoken {target_lang}; connect adjacent lines with short transitions only when the supplied lines support that connection.
- Use a hook only for a real turning point. Ordinary lines should state the story clearly without hype.
- Keep the recap chronological and make the protagonist's goal, obstacle, consequence, and rising stakes clear when the input supports them.

INPUT:
{json.dumps(payload, ensure_ascii=False)}"""
    def _call_polish():
        c, m = get_gemini_model("recap")
        return c.models.generate_content(
            model=m, contents=prompt,
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json", max_output_tokens=16384,
                response_schema={"type":"ARRAY","items":{"type":"OBJECT","properties":{"index":{"type":"INTEGER"},"script":{"type":"STRING"}},"required":["index","script"]}}))
    try:
        resp = call_gemini_with_retry(_call_polish, video_id=video_id, part_tag=part_tag, label="Script Editorial Polish", max_attempts=3, purpose="recap")
        data = _safe_response_json(resp)
        by_index = {int(x.get("index")): str(x.get("script", "")).strip() for x in data if str(x.get("script", "")).strip()}
        if set(by_index) != set(range(len(scenes))):
            raise ValueError("editor returned incomplete index set")
        return [{**scene, "script": by_index[i], "narration": by_index[i]} for i, scene in enumerate(scenes)]
    except Exception as err:
        print(f"[SCRIPT POLISH] skipped: {err}")
        return scenes


_DIALOGUE_TAG_RE = re.compile(r"\[(?:NARRATOR|MALE|FEMALE)\]\s*", re.IGNORECASE)

def _spoken_script_text(text):
    """Remove optional dialogue voice tags before counting/subtitling speech."""
    return _DIALOGUE_TAG_RE.sub('', str(text or '')).strip()

def _dialogue_voice_lines(text):
    """Return (role, spoken_text) pairs; untagged text remains narrator speech."""
    raw = str(text or '').strip()
    matches = list(re.finditer(r"\[(NARRATOR|MALE|FEMALE)\]\s*", raw, re.IGNORECASE))
    if not matches:
        return [('NARRATOR', raw)] if raw else []
    rows = []
    if matches[0].start() > 0 and raw[:matches[0].start()].strip():
        rows.append(('NARRATOR', raw[:matches[0].start()].strip()))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        text_part = raw[m.end():end].strip()
        if text_part:
            rows.append((m.group(1).upper(), text_part))
    return rows

def _dialogue_transcript_evidence(local_path, video_id="", part_tag=""):
    """Return compact local-Whisper evidence for one Dialogue analysis clip.

    This is evidence for Gemini's writing step only. It is never used as a
    narration fallback and never sent directly to TTS. A missing local ASR
    dependency is reported as a QA warning by the caller, not as fabricated
    dialogue.
    """
    try:
        rows = _manual_transcribe(local_path, f"{part_tag}dialogue")
        clean = []
        for row in rows or []:
            text = re.sub(r"\s+", " ", str(row.get("text", "") or "")).strip()
            try:
                start, end = float(row.get("start", 0)), float(row.get("end", 0))
            except (TypeError, ValueError):
                continue
            if text and math.isfinite(start) and math.isfinite(end) and end > start:
                clean.append({"start": round(max(0.0, start), 3), "end": round(max(0.0, end), 3), "text": text})
        return clean[:240]
    except Exception as exc:
        if video_id:
            log_status(video_id, f"⚠️ Dialogue local Whisper မရနိုင်သေးပါ — Gemini က visual evidence သာသုံးပါမည် ({exc})")
        return []

def _dialogue_transcript_clusters(rows, max_gap=1.25, max_chars=900):
    """Group adjacent Whisper segments into bounded conversation intervals."""
    clusters = []
    current = None
    for row in rows or []:
        try:
            start, end = float(row.get('start', 0)), float(row.get('end', 0))
        except (TypeError, ValueError):
            continue
        text = re.sub(r'\s+', ' ', str(row.get('text', '') or '')).strip()
        if not text or end <= start:
            continue
        if current and start - current['end'] <= max_gap and len(current['text']) + len(text) + 1 <= max_chars:
            current['end'] = round(max(current['end'], end), 3)
            current['text'] = f"{current['text']} {text}".strip()
        else:
            current = {'start': round(max(0.0, start), 3), 'end': round(end, 3), 'text': text}
            clusters.append(current)
    return clusters

def _script_word_count(scenes):
    """Count only spoken narration/dialogue text that will reach TTS."""
    return sum(len(re.findall(r"\S+", _spoken_script_text(s.get('script', '')))) for s in (scenes or []))


def _targeted_window_completion(video_id, part_tag, scenes, target_words, actual_wpm,
                                target_lang="Burmese", evidence_video_path=None,
                                recap_style="cinematic", max_attempts=2):
    """Expand a short analysis window without changing its evidence mapping.

    This is deliberately text-only and bounded.  The model receives the already
    validated scene ranges, so it cannot silently invent new timestamps or make
    a second broad video-analysis call.  If it cannot reach the budget, the
    caller keeps the best source-grounded version and reports the shortfall.
    """
    if not gemini_client or not scenes or target_words <= 0:
        return None
    current_words = _script_word_count(scenes)
    minimum_words = max(1, int(round(target_words * 0.85)))
    maximum_words = max(minimum_words, int(round(target_words * 1.15)))
    style_block = RECAP_STYLES.get(recap_style, RECAP_STYLES[RECAP_STYLE_DEFAULT])["voice"].format(target_lang=target_lang)
    compact = []
    for i, sc in enumerate(scenes):
        compact.append({
            "index": i,
            "start_time": sc.get("start_time"),
            "end_time": sc.get("end_time"),
            "script": str(sc.get("script", "") or ""),
        })
    prompt = f"""You are completing a source-grounded Burmese movie-recap narration window.

The input scenes and timestamps have already been validated against the actual video.
Keep every input index, start_time, and end_time exactly unchanged. Do not add,
delete, reorder, or merge scenes. Rewrite/expand only the existing script fields,
using details already supported by each scene's evidence. Do not invent names,
events, motives, dialogue, or off-screen facts. Do not use generic filler such as
'ဒီအပိုင်းမှာတော့' or object labels. Preserve chronological cause-and-effect and
natural spoken Myanmar endings (တယ်/ပါတယ်/ရဲ့, not formal သည် endings).

This window needs a measured narration budget:
- Current narration: {current_words} words
- Target: {target_words} words
- Acceptable range: {minimum_words}-{maximum_words} words
- Measured planning rate: {actual_wpm:.1f} words/minute
SELECTED RECAP STYLE — {RECAP_STYLES.get(recap_style, RECAP_STYLES[RECAP_STYLE_DEFAULT]).get('label', recap_style)}:
{style_block}
Return ONLY a JSON array with exactly one object for every input index:
[{{"index": 0, "script": "..."}}]

INPUT SCENES:
{json.dumps(compact, ensure_ascii=False)}"""

    completion_file = None
    try:
        # The first analysis call already saw the clip, but a completion call
        # must receive visual/audio evidence again. Sending only timestamps and
        # short scripts leaves Gemini with nothing factual to expand, which is
        # exactly how a 75-word result became only 133 words after repair.
        if evidence_video_path and os.path.exists(evidence_video_path):
            completion_file = gemini_client.files.upload(file=evidence_video_path)
            def _state_name(f):
                state = getattr(f, 'state', None)
                return getattr(state, 'name', state)
            while _state_name(completion_file) == 'PROCESSING':
                time.sleep(3)
                completion_file = gemini_client.files.get(name=completion_file.name)
            if _state_name(completion_file) != 'ACTIVE':
                completion_file = None

        def _reupload_completion_file():
            nonlocal completion_file
            if not evidence_video_path or not os.path.exists(evidence_video_path):
                return
            completion_file = gemini_client.files.upload(file=evidence_video_path)
            def _state_name_reupload(f):
                state = getattr(f, 'state', None)
                return getattr(state, 'name', state)
            while _state_name_reupload(completion_file) == 'PROCESSING':
                time.sleep(3)
                completion_file = gemini_client.files.get(name=completion_file.name)

        def _call_completion():
            client, model = get_gemini_model("recap")
            contents = [prompt, completion_file] if completion_file is not None else [prompt]
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=genai_types.GenerateContentConfig(
                    response_mime_type="application/json",
                    max_output_tokens=12000,
                    response_schema={
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "index": {"type": "INTEGER"},
                                "script": {"type": "STRING"},
                            },
                            "required": ["index", "script"],
                        },
                    },
                ),
            )

        response = call_gemini_with_retry(
            _call_completion, video_id=video_id, part_tag=part_tag,
            label="Window Targeted Completion", max_attempts=max_attempts,
            purpose="recap", file_recovery_fn=_reupload_completion_file,
        )
        data = _safe_response_json(response)
        if not isinstance(data, list):
            return None
        by_index = {}
        for item in data:
            if not isinstance(item, dict):
                continue
            try:
                idx = int(item.get("index"))
            except (TypeError, ValueError):
                continue
            text = str(item.get("script", "") or "").strip()
            if 0 <= idx < len(scenes) and text:
                by_index[idx] = text
        if len(by_index) != len(scenes):
            return None
        repaired = []
        for i, sc in enumerate(scenes):
            item = dict(sc)
            item["script"] = by_index[i]
            item["narration"] = by_index[i]
            repaired.append(item)
        repaired_words = _script_word_count(repaired)
        # Never replace a stronger script with a result that moved farther away
        # from the budget.  A below-target result can still be useful when it is
        # materially closer, and the global gate will make the remaining gap
        # explicit before TTS.
        if abs(repaired_words - target_words) >= abs(current_words - target_words):
            return None
        return repaired
    except Exception as exc:
        log_status(video_id, f"⚠️ {part_tag}Window completion မအောင်မြင်ပါ: {exc}")
        return None
    finally:
        if completion_file is not None:
            try:
                gemini_client.files.delete(name=completion_file.name)
            except Exception:
                pass


def _global_story_budget_completion(video_id, part_tag, scenes, target_words,
                                    dynamic_allocations, target_lang="Burmese",
                                    recap_style="cinematic", max_attempts=2):
    """Make one bounded, whole-part script pass after window analysis.

    Five-minute windows are evidence boundaries, so their first-pass scripts
    may be uneven. This pass receives the merged, timestamped evidence and the
    story-density allocation, then expands/compresses narration globally while
    preserving every source range and beat order.
    """
    if not gemini_client or not scenes or target_words <= 0:
        return None
    current_words = _script_word_count(scenes)
    if current_words >= int(round(target_words * 0.85)):
        return None
    min_words = max(1, int(round(target_words * 0.85)))
    max_words = max(min_words, int(round(target_words * 1.15)))
    style_block = RECAP_STYLES.get(recap_style, RECAP_STYLES[RECAP_STYLE_DEFAULT])["voice"].format(target_lang=target_lang)
    alloc_map = {int(x.get('window') or 0): int(x.get('dynamic_target_words') or 0)
                 for x in (dynamic_allocations or [])}
    compact = []
    for i, sc in enumerate(scenes):
        win = int(sc.get('analysis_window') or 0)
        compact.append({
            'index': i,
            'analysis_window': win,
            'window_target_words': alloc_map.get(win, 0),
            'start_time': sc.get('start_time'),
            'end_time': sc.get('end_time'),
            'script': str(sc.get('script', '') or ''),
            'raw_evidence': str(sc.get('raw_script', '') or ''),
        })
    prompt = f"""You are the final global editor for a source-grounded spoken {target_lang} movie recap.
The input beats are already chronologically mapped to real source timestamps.
Rewrite ONLY the script fields. Keep every index, analysis_window, start_time,
end_time, and beat order exactly unchanged. Do not add, delete, merge, or split beats.

GLOBAL HARD BUDGET:
- Current narration: {current_words} words
- Target: {target_words} words
- Accepted range: {min_words}-{max_words} words
- Do not stop early at a short summary. Use the full evidence supplied in every beat.
- Spend more detail on windows with larger window_target_words and compress only
  repetitive transitions. Cover setup, action, reaction, conflict, consequence,
  turning points, and ending state whenever those facts are present in the evidence.
- Expand only from script/raw_evidence and the exact mapped beat. Do not invent
  names, dialogue, motives, events, or off-screen facts. Do not use generic filler,
  repeated sentences, object labels, or formal Myanmar book-style endings.
- Write natural spoken Myanmar using တယ်/ပါတယ်/ရဲ့ style.

SELECTED RECAP STYLE — {RECAP_STYLES.get(recap_style, RECAP_STYLES[RECAP_STYLE_DEFAULT]).get('label', recap_style)}:
{style_block}

Return ONLY a JSON array with exactly one object per input index:
[{{"index": 0, "script": "..."}}]
Before returning, count only the script fields and make the total as close as
possible to {target_words} words without unsupported invention.

MERGED EVIDENCE:
{json.dumps(compact, ensure_ascii=False)}"""
    def _call_global():
        client, model = get_gemini_model("recap")
        return client.models.generate_content(
            model=model,
            contents=[prompt],
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                max_output_tokens=24000,
                response_schema={
                    'type': 'ARRAY',
                    'items': {
                        'type': 'OBJECT',
                        'properties': {
                            'index': {'type': 'INTEGER'},
                            'script': {'type': 'STRING'},
                        },
                        'required': ['index', 'script'],
                    },
                },
            ),
        )
    try:
        response = call_gemini_with_retry(
            _call_global, video_id=video_id, part_tag=part_tag,
            label='Global Story Budget Completion', max_attempts=max_attempts,
            purpose='recap',
        )
        data = _safe_response_json(response)
        if not isinstance(data, list):
            return None
        by_index = {}
        for item in data:
            if not isinstance(item, dict):
                continue
            try:
                idx = int(item.get('index'))
            except (TypeError, ValueError):
                continue
            text = str(item.get('script', '') or '').strip()
            if 0 <= idx < len(scenes) and text:
                by_index[idx] = text
        if len(by_index) != len(scenes):
            return None
        repaired = []
        for i, sc in enumerate(scenes):
            item = dict(sc)
            item['script'] = by_index[i]
            item['narration'] = by_index[i]
            repaired.append(item)
        repaired_words = _script_word_count(repaired)
        if abs(repaired_words - target_words) >= abs(current_words - target_words):
            return None
        return repaired
    except Exception as exc:
        log_status(video_id, f"⚠️ {part_tag}Global story-budget completion မအောင်မြင်ပါ: {exc}")
        return None


def _run_gemini_auto_parts_workflow(video_id, local_path, logo_path, font_path, settings, v_dur):
    """Auto workflow: publish at most one 30-minute source part at a time.

    A source <=30 minutes stays as one part. Longer sources are split into
    30-minute parts. Each part is analyzed in 5-minute Gemini windows, then
    its merged local timeline is sent through one TTS/render pass. Parts can
    remain separate or be joined with the explicit merge_parts toggle.
    """
    part_sec = 30.0 * 60.0
    analysis_sec = 5.0 * 60.0
    n_parts = max(1, int(math.ceil(v_dur / part_sec)))
    merge_enabled = str(settings.get('merge_parts', settings.get('clean_output', 'false'))).strip().lower() in ('true', '1', 'on', 'yes')
    root_temp = tempfile.mkdtemp(prefix=f"auto_parts_{video_id}_")
    part_videos = []
    dialogue_totals = {
        'segments': 0, 'clusters': 0, 'covered': 0,
        'tagged_scenes': 0, 'scenes': 0,
    }
    try:
        log_status(video_id, f"🤖 Auto: {v_dur/60:.1f} မိနစ် source ကို {n_parts} Part ({'30 မိနစ်စီ' if n_parts > 1 else 'Part မခွဲ'}) ခွဲပြီး Part တစ်ခုချင်းစီကို 5 မိနစ်စီ analysis လုပ်ပါမည်။")
        set_job_state(video_id, state="running", part=0, total=n_parts)
        for part_idx in range(n_parts):
            part_start = part_idx * part_sec
            part_len = min(part_sec, v_dur - part_start)
            if part_len <= 0.2:
                continue
            part_tag = f"[Part {part_idx + 1}/{n_parts}] " if n_parts > 1 else ""
            part_suffix = f"_part{part_idx + 1:02d}" if n_parts > 1 else ""
            part_path = local_path if n_parts == 1 else os.path.join(root_temp, f"source_part_{part_idx + 1:02d}.mp4")
            try:
                if n_parts > 1:
                    log_status(video_id, f"{part_tag}မူရင်း video မှ 30 မိနစ် Part ကို ပြင်ဆင်နေပါသည်...")
                    run_cmd(["ffmpeg", "-y", "-ss", str(part_start), "-i", local_path, "-t", str(part_len),
                             "-c:v", "libx264", "-preset", "veryfast", "-r", "30", "-vsync", "cfr",
                             "-c:a", "aac", "-ar", "44100", "-avoid_negative_ts", "make_zero", part_path],
                            f"Prepare Auto {part_tag}", video_id)

                analysis_temp = tempfile.mkdtemp(prefix=f"auto_analysis_{video_id}_{part_idx + 1:02d}_", dir=root_temp)
                merged = []
                window_budgets = []
                n_windows = max(1, int(math.ceil(part_len / analysis_sec)))
                for win_idx in range(n_windows):
                    win_start = win_idx * analysis_sec
                    win_len = min(analysis_sec, part_len - win_start)
                    if win_len <= 0.2:
                        continue
                    clip = os.path.join(analysis_temp, f"analysis_{win_idx + 1:03d}.mp4")
                    log_status(video_id, f"{part_tag}🔎 Analysis {win_idx + 1}/{n_windows} — 5 မိနစ်စာ visual/story evidence စုဆောင်းနေပါသည်...")
                    run_cmd(["ffmpeg", "-y", "-ss", str(win_start), "-i", part_path, "-t", str(win_len),
                             "-vf", "scale=-2:480,fps=15", "-c:v", "libx264", "-preset", "veryfast",
                             "-crf", "30", "-c:a", "aac", "-b:a", "64k", clip],
                            f"Auto analysis {part_tag}{win_idx + 1}/{n_windows}", video_id)
                    chunk_settings = dict(settings)
                    chunk_settings['_analysis_only'] = True
                    # The Auto orchestrator owns the Gemini window workflow.
                    # Do not inherit a stale local-fast value from an older UI
                    # request, otherwise the per-window Gemini word budget and
                    # targeted-completion gates are bypassed entirely.
                    chunk_settings['analysis_engine'] = 'gemini_auto'
                    chunk_settings.pop('_precomputed_scenes', None)
                    result = process_video_segment(
                        video_id, clip, logo_path, font_path, chunk_settings,
                        file_suffix=f"_auto_p{part_idx + 1:02d}_a{win_idx + 1:03d}",
                        prev_context=(" ".join(x.get('script', '') for x in merged[-2:]))[-600:],
                        chunk_label=f"Part {part_idx + 1}/{n_parts} analysis {win_idx + 1}/{n_windows}",
                    )
                    if not result.get('success'):
                        detail = result.get('error') or result.get('errors') or result.get('qa', {}).get('errors') or 'no detailed error returned'
                        if isinstance(detail, (list, tuple)):
                            detail = '; '.join(str(x) for x in detail)
                        if result.get('qa', {}).get('failure_category') == 'gemini_billing_credits_depleted':
                            raise GeminiBillingError(str(detail))
                        raise RuntimeError(f"Auto Part {part_idx + 1} analysis {win_idx + 1}/{n_windows} failed: {detail}")
                    window_scenes = [dict(x) for x in result.get('analysis_scenes', [])]
                    window_qa = result.get('qa') or {}
                    if str(settings.get('dialogue_mode', 'false')).strip().lower() in ('true', '1', 'yes', 'on'):
                        dialogue_totals['segments'] += int(window_qa.get('dialogue_transcript_segments') or 0)
                        dialogue_totals['clusters'] += int(window_qa.get('dialogue_transcript_clusters') or 0)
                        dialogue_totals['covered'] += int(window_qa.get('dialogue_clusters_covered') or 0)
                        dialogue_totals['tagged_scenes'] += int(window_qa.get('dialogue_tagged_scene_count') or 0)
                        dialogue_totals['scenes'] += int(window_qa.get('dialogue_scene_count') or 0)
                    window_target_words = int(window_qa.get('target_words') or 0)
                    window_expected_wpm = float(window_qa.get('expected_wpm') or 0.0)
                    window_actual_words = _script_word_count(window_scenes)
                    window_min_words = int(round(window_target_words * 0.85)) if window_target_words else 0
                    if window_target_words and window_actual_words < window_min_words:
                        log_status(
                            video_id,
                            f"{part_tag}⚠️ Window {win_idx + 1}/{n_windows} narration တိုနေပါသည် "
                            f"({window_actual_words}/{window_target_words} words) — targeted completion တစ်ကြိမ်လုပ်ပါမည်..."
                        )
                        repaired_window = _targeted_window_completion(
                            video_id, part_tag, window_scenes, window_target_words,
                            window_expected_wpm or 130.0,
                            target_lang=settings.get('target_lang', 'Burmese'),
                            evidence_video_path=clip,
                            recap_style=settings.get('recap_style', RECAP_STYLE_DEFAULT),
                            max_attempts=2,
                        )
                        if repaired_window:
                            window_scenes = repaired_window
                            window_actual_words = _script_word_count(window_scenes)
                            log_status(
                                video_id,
                                f"{part_tag}Window {win_idx + 1}/{n_windows} completion result: "
                                f"{window_actual_words}/{window_target_words} words "
                                f"({window_actual_words / max(window_target_words, 1) * 100:.1f}%)"
                            )
                        if window_actual_words < window_min_words:
                            # A 5-minute analysis window is an evidence
                            # boundary, not a story-length boundary. Opening
                            # or transition windows can legitimately contain
                            # fewer words than a climax window. Keep the best
                            # grounded result and enforce the budget globally
                            # after all windows have been seen.
                            log_status(
                                video_id,
                                f"⚠️ {part_tag}Window {win_idx + 1}/{n_windows} သည် "
                                f"{window_actual_words}/{window_target_words} words ပဲရပါသည် — "
                                f"window-level warning အဖြစ်ထားပြီး story-density အလိုက် global budget ကို ပြန်တွက်ပါမည်။"
                            )
                    window_budgets.append({
                        'window': win_idx + 1,
                        'target_words': window_target_words,
                        'actual_words': window_actual_words,
                        'ratio': round(window_actual_words / window_target_words, 4) if window_target_words else 1.0,
                        'scene_count': len(window_scenes),
                    })
                    for scene in window_scenes:
                        item = dict(scene)
                        a = parse_time_to_sec(item.get('start_time')) + win_start
                        b = parse_time_to_sec(item.get('end_time')) + win_start
                        a = max(0.0, min(part_len, a)); b = max(a, min(part_len, b))
                        item['start_time'] = format_timestamp(a)
                        item['end_time'] = format_timestamp(b)
                        item['analysis_window'] = win_idx + 1
                        merged.append(item)
                    try:
                        os.remove(clip)
                    except OSError:
                        pass
                shutil.rmtree(analysis_temp, ignore_errors=True)
                if not merged:
                    raise RuntimeError(f"Auto Part {part_idx + 1}: Gemini produced no usable scenes")
                merged.sort(key=lambda x: parse_time_to_sec(x.get('start_time')))
                clean = []
                last_end = 0.0
                for item in merged:
                    a = parse_time_to_sec(item.get('start_time')); b = parse_time_to_sec(item.get('end_time'))
                    if b <= a + 0.05:
                        continue
                    if clean and a < last_end:
                        a = last_end; item['start_time'] = format_timestamp(a)
                    if b > a + 0.05:
                        clean.append(item); last_end = b
                if not clean:
                    raise RuntimeError(f"Auto Part {part_idx + 1}: merged timeline has no valid scenes")
                merged_words = _script_word_count(clean)
                merged_target_words = sum(int(x.get('target_words') or 0) for x in window_budgets)
                # Allocate the global budget by story density rather than
                # assuming every five-minute window deserves the same number
                # of words. Scene count and narrated detail are useful local
                # signals; the allocation is advisory for repair/QA, while
                # the original part target remains the honest denominator.
                _density_scores = []
                for wb in window_budgets:
                    _scene_signal = min(1.0, float(wb.get('scene_count') or 0) / 10.0)
                    _detail_signal = min(1.0, float(wb.get('actual_words') or 0) / max(1.0, merged_target_words / max(1, len(window_budgets))))
                    _density_scores.append(max(0.20, 0.55 * _scene_signal + 0.45 * _detail_signal))
                _density_total = sum(_density_scores) or 1.0
                for wb, score in zip(window_budgets, _density_scores):
                    wb['story_density_score'] = round(score, 4)
                    wb['dynamic_target_words'] = int(round(merged_target_words * score / _density_total)) if merged_target_words else 0
                _dynamic_target_words = sum(int(w.get('dynamic_target_words') or 0) for w in window_budgets)
                if merged_target_words and merged_words < int(round(merged_target_words * 0.85)):
                    log_status(
                        video_id,
                        f"{part_tag}Global story-budget completion — "
                        f"{merged_words}/{merged_target_words} words ကို dynamic allocation အတိုင်း တစ်ကြိမ်ပြန်ညှိပါမည်..."
                    )
                    globally_repaired = _global_story_budget_completion(
                        video_id, part_tag, clean, merged_target_words,
                        window_budgets, target_lang=settings.get('target_lang', 'Burmese'),
                        recap_style=settings.get('recap_style', RECAP_STYLE_DEFAULT),
                        max_attempts=2,
                    )
                    if globally_repaired:
                        clean = globally_repaired
                        merged_words = _script_word_count(clean)
                        log_status(
                            video_id,
                            f"{part_tag}Global completion result: {merged_words}/{merged_target_words} words "
                            f"({merged_words / max(merged_target_words, 1) * 100:.1f}%)"
                        )
                merged_min_words = int(round(merged_target_words * 0.85)) if merged_target_words else 0
                # "About 50%" allows a small WPM/rounding difference around
                # the boundary, so a measured 48-49% result is not rejected
                # after Gemini has exhausted its bounded attempts.
                # A word budget is an editorial target, not evidence that may
                # be safely fabricated. Gemini can legitimately return fewer
                # words for a quiet/visual window. Keep the old ~50% value as
                # a useful QA marker, but never abort a non-empty, validated
                # timeline solely because the writer could not pad it.
                merged_accept_words = int(round(merged_target_words * 0.30)) if merged_target_words else 0
                merged_ratio = (merged_words / merged_target_words) if merged_target_words else 1.0
                if merged_target_words and merged_words < merged_min_words:
                    if merged_words >= merged_accept_words:
                        job_qa.setdefault(video_id, {}).setdefault('warnings', []).append(
                            f"Part {part_idx + 1} global narration is below target but accepted at "
                            f"{merged_ratio * 100:.1f}% after bounded Gemini attempts"
                        )
                        log_status(
                            video_id,
                            f"⚠️ {part_tag}Global narration target မပြည့်သေးပါ — "
                            f"{merged_words}/{merged_target_words} words ({merged_ratio * 100:.1f}%)၊ "
                            f"Gemini ထပ်မတိုးနိုင်တော့သဖြင့် grounded-script fallback နဲ့ warning ဆက်လုပ်ပါမည်။"
                        )
                    else:
                        # Do not turn a valid, source-grounded recap into a
                        # fatal job error. The final TTS gate will measure the
                        # actual duration; this branch only records that the
                        # editorial target was not reachable without adding
                        # unsupported content.
                        job_qa.setdefault(video_id, {}).setdefault('warnings', []).append(
                            f"Part {part_idx + 1} narration is materially below target "
                            f"({merged_ratio * 100:.1f}%) but the grounded timeline is usable"
                        )
                        log_status(
                            video_id,
                            f"⚠️ {part_tag}Global word target မပြည့်သေးပါ — "
                            f"{merged_words}/{merged_target_words} words ({merged_ratio * 100:.1f}%)။ "
                            f"Unsupported filler မထည့်ဘဲ grounded script နဲ့ TTS/render ဆက်လုပ်ပါမည်။"
                        )
                job_qa.setdefault(video_id, {}).setdefault('auto_window_budgets', []).append({
                    'part': part_idx + 1,
                    'windows': window_budgets,
                    'merged_target_words': merged_target_words,
                    'merged_words': merged_words,
                    'dynamic_target_words': _dynamic_target_words,
                    'merged_ratio': round(merged_ratio, 4),
                    'minimum_acceptance_ratio': 0.30,
                })
                log_status(
                    video_id,
                    f"{part_tag}✅ Global story-density budget စစ်ပြီးပါပြီ — "
                    f"{merged_words}/{merged_target_words} words "
                    f"({merged_ratio * 100:.1f}%; dynamic allocation={_dynamic_target_words})"
                )
                log_status(video_id, f"{part_tag}✅ 5-minute analyses ပြီးပါပြီ — {len(clean)} scenes ကို TTS/render တစ်ကြိမ်လုပ်ပါမည်။")
                final_settings = dict(settings)
                final_settings['_precomputed_scenes'] = clean
                final_settings.pop('_analysis_only', None)
                final_settings['analysis_engine'] = 'gemini_auto'
                final = process_video_segment(video_id, part_path, logo_path, font_path, final_settings,
                                              file_suffix=part_suffix, prev_context="", chunk_label=part_tag)
                if not final.get('success'):
                    detail = final.get('error') or final.get('errors') or final.get('qa', {}).get('errors') or 'no detailed render error returned'
                    if isinstance(detail, (list, tuple)):
                        detail = '; '.join(str(x) for x in detail)
                    raise RuntimeError(f"Auto {part_tag} final TTS/render failed: {detail}")
                if final.get('video_path'):
                    part_videos.append(final['video_path'])
                if str(settings.get('dialogue_mode', 'false')).strip().lower() in ('true', '1', 'yes', 'on'):
                    job_qa.setdefault(video_id, {}).update({
                        'dialogue_transcript_segments': dialogue_totals['segments'],
                        'dialogue_transcript_clusters': dialogue_totals['clusters'],
                        'dialogue_clusters_covered': dialogue_totals['covered'],
                        'dialogue_cluster_coverage_ratio': round(
                            dialogue_totals['covered'] / dialogue_totals['clusters'], 4
                        ) if dialogue_totals['clusters'] else None,
                        'dialogue_tagged_scene_count': dialogue_totals['tagged_scenes'],
                        'dialogue_scene_count': dialogue_totals['scenes'],
                    })
                set_job_state(video_id, state="running", part=part_idx + 1, total=n_parts)
            except Exception:
                raise

        job_qa.setdefault(video_id, {}).update({
            'auto_mode': '30_min_parts_5_min_analysis',
            'source_duration_sec': round(v_dur, 3),
            'auto_parts': n_parts,
            'auto_part_duration_sec': 1800,
            'analysis_window_sec': 300,
            'merge_parts_requested': merge_enabled,
            'published_output_parts': n_parts if not merge_enabled else 1,
        })
        if merge_enabled and len(part_videos) == n_parts and n_parts > 1:
            combined_path = os.path.join(DOWNLOAD_DIR, f"{video_id}_combined_final.mp4")
            list_path = os.path.join(root_temp, "auto_concat.txt")
            with open(list_path, "w", encoding="utf-8") as f:
                for p in part_videos:
                    f.write("file '" + p.replace("'", "'\\''") + "'\n")
            log_status(video_id, f"🔗 Auto Part {n_parts} ခုကို video တစ်ဖိုင်တည်း ပေါင်းစည်းနေပါသည်...")
            run_cmd(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path,
                     "-c:v", "libx264", "-preset", "veryfast", "-r", "30", "-vsync", "cfr",
                     "-c:a", "aac", "-b:a", "128k", "-ar", "44100", combined_path],
                    "Combine Auto Parts", video_id)
            if not os.path.exists(combined_path) or os.path.getsize(combined_path) <= 1000:
                raise RuntimeError("Auto Part merge produced no usable combined MP4")
            for p in part_videos:
                try: os.remove(p)
                except OSError: pass
            log_status(video_id, "✅ Auto Part merge ပြီးပါပြီ — combined final MP4 အသင့်ဖြစ်ပါပြီ။")
        else:
            log_status(video_id, f"✅ Auto recap ပြီးပါပြီ — {n_parts} Part output ကို သီးခြားသိမ်းထားပါသည်။")
        set_job_state(video_id, state="done", part=n_parts, total=n_parts)
        return {"success": True, "video_path": (combined_path if merge_enabled and n_parts > 1 else (part_videos[0] if n_parts == 1 and part_videos else None))}
    except Exception as exc:
        log_status(video_id, f"❌ Auto root error: {type(exc).__name__}: {exc}")
        job_qa.setdefault(video_id, {}).setdefault('errors', []).append(f"{type(exc).__name__}: {exc}")
        set_job_state(video_id, state="error", part=0, total=n_parts)
        return {"success": False, "error": f"{type(exc).__name__}: {exc}", "video_path": None}
    finally:
        shutil.rmtree(root_temp, ignore_errors=True)


def _run_gemini_full_movie_workflow(video_id, local_path, logo_path, font_path, settings, v_dur):
    """Analyze a complete movie in bounded Gemini evidence windows, then
    synthesize/render one continuous recap from the merged global script.

    The three-minute windows are an analysis/context limit only. They are not
    output parts, are never published, and are never TTS-rendered separately.
    """
    analysis_temp = tempfile.mkdtemp(prefix=f"gemini_analysis_{video_id}_")
    try:
        window_sec = 180.0
        n_windows = max(1, int(math.ceil(v_dur / window_sec)))
        merged = []
        log_status(video_id, f"🤖 Auto Gemini: ရုပ်ရှင်တစ်ကားလုံးကို {n_windows} ခုသော 3-minute analysis windows နဲ့ အပြည့်အဝလေ့လာနေပါသည် — output ကို Part မခွဲပါ။")
        set_job_state(video_id, state="running", part=0, total=n_windows + 1)
        for idx in range(n_windows):
            start = idx * window_sec
            length = min(window_sec, v_dur - start)
            if length <= 0.2:
                continue
            clip = os.path.join(analysis_temp, f"analysis_{idx+1:03d}.mp4")
            label = f"analysis {idx+1}/{n_windows}"
            log_status(video_id, f"🔎 {label}: ဇာတ်လမ်းအချက်အလက်နဲ့ visual continuity ကို စုဆောင်းနေပါသည်...")
            run_cmd(["ffmpeg", "-y", "-ss", str(start), "-i", local_path, "-t", str(length),
                     "-vf", "scale=-2:480,fps=15", "-c:v", "libx264", "-preset", "veryfast",
                     "-crf", "30", "-c:a", "aac", "-b:a", "64k", clip], label, video_id)
            chunk_settings = dict(settings)
            chunk_settings['_analysis_only'] = True
            chunk_settings.pop('_precomputed_scenes', None)
            result = process_video_segment(
                video_id, clip, logo_path, font_path, chunk_settings,
                file_suffix=f"_analysis{idx+1:03d}",
                prev_context=(" ".join(x.get('script', '') for x in merged[-2:]))[-600:],
                chunk_label=f"{idx+1}/{n_windows}",
            )
            if not result.get('success'):
                # Keep the first real failure visible. Previously Auto only
                # reported "analysis window N failed", which hid whether the
                # cause was Gemini quota/model, empty JSON, TTS, ffmpeg, or
                # source-media probing and made a 50-minute job impossible
                # to diagnose from the UI log.
                _window_errors = result.get('error') or result.get('errors') or result.get('qa', {}).get('errors')
                if isinstance(_window_errors, (list, tuple)):
                    _window_errors = '; '.join(str(x) for x in _window_errors)
                _window_errors = str(_window_errors or 'no detailed error returned')
                log_status(video_id, f"❌ Auto Gemini analysis window {idx+1}/{n_windows} မအောင်မြင်ပါ: {_window_errors}")
                raise RuntimeError(f"Gemini analysis window {idx+1}/{n_windows} failed: {_window_errors}")
            for scene in result.get('analysis_scenes', []):
                item = dict(scene)
                a = parse_time_to_sec(item.get('start_time')) + start
                b = parse_time_to_sec(item.get('end_time')) + start
                a = max(0.0, min(v_dur, a)); b = max(a, min(v_dur, b))
                item['start_time'] = format_timestamp(a)
                item['end_time'] = format_timestamp(b)
                item['analysis_window'] = idx + 1
                merged.append(item)
            set_job_state(video_id, state="running", part=idx + 1, total=n_windows + 1)
            try:
                os.remove(clip)
            except OSError:
                pass

        if not merged:
            raise RuntimeError("Gemini produced no usable full-movie scenes")
        merged.sort(key=lambda x: parse_time_to_sec(x.get('start_time')))
        # Remove boundary duplicates and clamp accidental overlaps while
        # preserving chronological evidence ownership.
        clean = []
        last_end = 0.0
        for item in merged:
            a = parse_time_to_sec(item.get('start_time')); b = parse_time_to_sec(item.get('end_time'))
            if b <= a + 0.05:
                continue
            if clean and a < last_end:
                a = last_end
                item['start_time'] = format_timestamp(a)
            if b > a + 0.05:
                clean.append(item); last_end = b
        if not clean:
            raise RuntimeError("Merged Gemini analysis contained no valid chronological scenes")
        log_status(video_id, f"✅ Full-movie analysis ပြီးပါပြီ — {len(clean)} scenes ကို script တစ်ခုတည်းအဖြစ် ပေါင်းပြီး narration တစ်ကြိမ်တည်း သွင်းပါမည်။")
        final_settings = dict(settings)
        final_settings['_precomputed_scenes'] = clean
        final_settings.pop('_analysis_only', None)
        final_settings['analysis_engine'] = 'gemini_auto'
        job_qa.setdefault(video_id, {}).update({
            'auto_analysis_mode': 'full_movie_merged',
            'analysis_window_sec': 180,
            'analysis_windows': n_windows,
            'published_output_parts': 1,
            'script_merge_gate': 'pass',
        })
        final = process_video_segment(video_id, local_path, logo_path, font_path, final_settings,
                                      file_suffix="", prev_context="", chunk_label="")
        if final.get('success'):
            set_job_state(video_id, state="done", part=n_windows + 1, total=n_windows + 1)
            # Auto output is intentionally a four-file contract: narration
            # audio, final script, SRT, and final MP4. Shorts/teaser assets
            # would violate that contract, so they remain Manual/legacy-only.
        else:
            set_job_state(video_id, state="error", part=n_windows, total=n_windows + 1)
        return final
    finally:
        shutil.rmtree(analysis_temp, ignore_errors=True)


def _run_advanced_pipeline_inner(video_id, local_path, logo_path, font_path, settings, CHUNK_LIMIT_SEC):
    try:
        v_dur = float(subprocess.check_output(
            ["ffprobe","-v","error","-show_entries","format=duration",
             "-of","default=noprint_wrappers=1:nokey=1", local_path]
        ).decode().strip())
        if not math.isfinite(v_dur) or v_dur <= 0:
            raise ValueError(f"invalid source duration {v_dur!r}")
    except Exception as e:
        log_status(video_id, f"❌ ဗီဒီယို Duration ကို ဖတ်၍မရပါ: {e}")
        return

    # Container-level duration metadata can be wrong/corrupted (some
    # re-encoded or badly-remuxed files report a bogus total duration),
    # which silently explodes n_chunks below into dozens of tiny, mostly
    # bogus "parts" - a 20-minute file reporting 63 parts is exactly this
    # failure mode. Cross-check against the video stream's OWN duration
    # (from its packet timestamps, a separate metadata field) and prefer
    # whichever is smaller/more plausible when they disagree substantially.
    try:
        v_dur_stream = float(subprocess.check_output(
            ["ffprobe","-v","error","-select_streams","v:0","-show_entries","stream=duration",
             "-of","default=noprint_wrappers=1:nokey=1", local_path]
        ).decode().strip())
        if math.isfinite(v_dur_stream) and v_dur_stream > 0 and abs(v_dur - v_dur_stream) / max(v_dur, v_dur_stream) > 0.15:
            log_status(video_id, f"⚠️ Video duration metadata မကိုက်ညီပါ (container={v_dur/60:.1f} မိနစ်, video-stream={v_dur_stream/60:.1f} မိနစ်) — ပိုမှန်ကန်မယ့်ဟာကို သုံးပါမည်...")
            # The recap timeline is visual, so prefer the valid video-stream
            # duration rather than blindly taking the smaller container value
            # (which can be a truncated audio/container timestamp).
            v_dur = v_dur_stream
    except Exception:
        pass  # best-effort cross-check only; keep the format=duration value if this fails

    # Auto Gemini uses 30-minute output parts and 5-minute analysis windows.
    # Analysis windows are private working steps, never published as parts.
    if str(settings.get('analysis_engine', '')).strip().lower() == 'gemini_auto':
        _run_gemini_auto_parts_workflow(video_id, local_path, logo_path, font_path, settings, v_dur)
        return

    if v_dur <= CHUNK_LIMIT_SEC:
        set_job_state(video_id, state="running", part=1, total=1)
        single_result = process_video_segment(
            video_id, local_path, logo_path, font_path, settings,
            file_suffix="", prev_context=_load_continuation_context(settings), chunk_label=""
        )
        if single_result.get("success"):
            set_job_state(video_id, state="done", part=1, total=1)
            _maybe_generate_shorts(video_id, settings)
            _maybe_generate_teaser(video_id, settings)
        else:
            set_job_state(video_id, state="error", part=0, total=1)
        return

    n_chunks = math.ceil(v_dur / CHUNK_LIMIT_SEC)
    log_status(video_id, f"🎬 ဇာတ်ကားက {int(v_dur//60)} မိနစ်ရှိသဖြင့် Part {n_chunks} ပိုင်းခွဲပြီး အဆက်စပ် Recap လုပ်ပါမည်...")
    set_job_state(video_id, state="running", part=0, total=n_chunks)

    split_temp   = tempfile.mkdtemp(prefix=f"split_{video_id}_")
    prev_context = _load_continuation_context(settings)
    ok_parts     = 0
    part_videos  = []
    try:
        for idx in range(n_chunks):
            start      = idx * CHUNK_LIMIT_SEC
            length     = min(CHUNK_LIMIT_SEC, v_dur - start)
            chunk_path = os.path.join(split_temp, f"chunk_{idx+1}.mp4")
            label      = f"{idx+1}/{n_chunks}"

            log_status(video_id, f"✂️ [Part {label}] မူရင်းဇာတ်ကားမှ အပိုင်းလေးကို ဖြတ်ထုတ်နေပါသည်...")
            set_job_state(video_id, state="running", part=idx + 1, total=n_chunks)
            try:
                run_cmd([
                    "ffmpeg","-y","-ss",str(start),"-i",local_path,"-t",str(length),
                    "-c:v","libx264","-preset","ultrafast","-c:a","aac",
                    "-avoid_negative_ts","make_zero", chunk_path
                ], f"Split Part {label}", video_id)
            except Exception as e:
                log_status(video_id, f"❌ [Part {label}] ဖြတ်ခြင်း မအောင်မြင်ပါ: {e}. ဒီအပိုင်းကို ကျော်ပါမည်။")
                continue

            result = process_video_segment(
                video_id, chunk_path, logo_path, font_path, settings,
                file_suffix=f"_part{idx+1:02d}", prev_context=prev_context, chunk_label=label
            )
            if result.get("success"):
                ok_parts += 1
                if result.get("video_path"):
                    part_videos.append(result["video_path"])
            if result.get("last_script"):
                prev_context = result["last_script"]

            try: os.remove(chunk_path)
            except: pass

        if ok_parts == n_chunks:
            log_status(video_id, f"✅ ဇာတ်ကားလုံးကို Part {n_chunks} ခု ပြည့်စုံစွာ Recap ပြီးစီးပါပြီ။ Download စာမျက်နှာတွင် Part အားလုံး ရယူနိုင်ပါသည်။")
            if str(settings.get('clean_output', '')).strip().lower() in ('true', '1', 'on', 'yes') and len(part_videos) == n_chunks:
                log_status(video_id, f"🔗 Part {n_chunks} ခုကို ဗီဒီယိုတစ်ခုတည်း ပေါင်းစည်းနေပါသည်...")
                combined_path = os.path.join(DOWNLOAD_DIR, f"{video_id}_combined_final.mp4")
                list_path = os.path.join(split_temp, "concat_list.txt")
                try:
                    expected_combined_duration = sum(probe_media(p)['duration'] for p in part_videos)
                    with open(list_path, "w", encoding="utf-8") as f:
                        for p in part_videos:
                            f.write(f"file '{p}'\n")
                    # Re-encode on combine rather than -c copy: stream-copy
                    # concat can leave tiny gaps/overlaps at each seam
                    # (AAC audio frame boundaries in particular don't always
                    # line up cleanly when copied), and those small seam
                    # errors compound across every part boundary into
                    # visible drift by the later parts of a long combined
                    # video. Re-encoding forces every frame/sample to be
                    # laid out on a single continuous, exact timeline.
                    run_cmd(["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                              "-i", list_path,
                              "-c:v", "libx264", "-preset", "veryfast", "-r", "30", "-vsync", "cfr",
                              "-c:a", "aac", "-b:a", "128k", "-ar", "44100",
                              combined_path],
                            "Combine Parts", video_id)
                    if os.path.exists(combined_path) and os.path.getsize(combined_path) > 1000:
                        combined_media = probe_media(combined_path)
                        combined_error_ms = abs(combined_media['duration'] - expected_combined_duration) * 1000
                        combined_qa = job_qa.setdefault(video_id, {})
                        combined_qa.update({
                            'combined_duration_sec': round(combined_media['duration'], 3),
                            'combined_expected_duration_sec': round(expected_combined_duration, 3),
                            'combined_duration_error_ms': round(combined_error_ms, 2),
                            'combined_media': combined_media,
                        })
                        combined_ok = combined_media['has_video'] and combined_media['has_audio'] and combined_error_ms <= 250
                        if combined_ok:
                            log_status(video_id, "✅ ပေါင်းစည်းပြီးသား ဗီဒီယို QA အောင်မြင်ပါသည် (Download Files ထဲတွင် ရယူနိုင်ပါသည်)။")
                            # The per-part files were only useful as pieces to
                            # combine - once the combined video exists, remove them.
                            for p in part_videos:
                                try: os.remove(p)
                                except Exception: pass
                            set_job_state(video_id, state="done", part=n_chunks, total=n_chunks)
                            _maybe_generate_shorts(video_id, settings)
                            _maybe_generate_teaser(video_id, settings)
                        else:
                            combined_qa['needs_review'] = True
                            combined_qa.setdefault('errors', []).append('Combined final MP4 failed stream or duration QA')
                            set_job_state(video_id, state="error", part=n_chunks, total=n_chunks)
                            log_status(video_id, "❌ Combined final MP4 QA မအောင်မြင်ပါ — Part files များကို မဖျက်သေးပါ။")
                    else:
                        log_status(video_id, "⚠️ ပေါင်းစည်းမှု မအောင်မြင်ပါ — Part တစ်ခုချင်းစီကိုပဲ Download ရယူပါ။")
                        set_job_state(video_id, state="error", part=n_chunks, total=n_chunks)
                except Exception as combine_err:
                    log_status(video_id, f"⚠️ ပေါင်းစည်းမှု Error: {combine_err} — Part တစ်ခုချင်းစီကိုပဲ Download ရယူပါ။")
                    job_qa.setdefault(video_id, {}).setdefault('errors', []).append(str(combine_err))
                    set_job_state(video_id, state="error", part=n_chunks, total=n_chunks)
            elif len(part_videos) == n_chunks:
                # Non-clean mode intentionally keeps the validated part MP4s.
                set_job_state(video_id, state="done", part=n_chunks, total=n_chunks)
            else:
                set_job_state(video_id, state="error", part=ok_parts, total=n_chunks)
        elif ok_parts > 0:
            log_status(video_id, f"⚠️ Part {ok_parts}/{n_chunks} ကို အောင်မြင်စွာ ပြီးစီးပါသည်။ ကျန်အပိုင်းများတွင် Error ဖြစ်ပေါ်ခဲ့ပါသည် — Log အထက်ပိုင်းကို စစ်ဆေးပါ။")
            set_job_state(video_id, state="error", part=ok_parts, total=n_chunks)
        else:
            log_status(video_id, "❌ Part အားလုံး Fail ဖြစ်သွားပါသည်။")
            set_job_state(video_id, state="error", part=0, total=n_chunks)
    finally:
        try: shutil.rmtree(split_temp)
        except: pass


# ==========================================
# Local-Fast analysis engine (alternative to Gemini-video STEP 1 below)
# ==========================================
# Instead of uploading the whole source video to Gemini (which costs
# ~250-300 tokens PER SECOND of video and forces chunking to stay under
# context/rate limits), this detects scene cuts and transcribes dialogue
# LOCALLY (free, no API, no rate limit) and sends Gemini only TEXT (the
# transcript) + a handful of small keyframe IMAGES - 1-2 orders of
# magnitude cheaper/faster, and typically needs only ONE Gemini call for
# an entire clip instead of one call per chunk.
#
# Returns scenes_data in the EXACT shape the rest of the pipeline already
# expects ([{"start_time","end_time","script"}, ...] with "HH:MM:SS.mmm"
# strings), so TTS/frame-exact-assembly code below needs zero changes.
_whisper_model_cache = {}

def _local_fast_get_whisper():
    from faster_whisper import WhisperModel
    key = "cpu_int8"
    if key not in _whisper_model_cache:
        model_name = os.environ.get("WHISPER_MODEL", "base")
        _whisper_model_cache[key] = WhisperModel(model_name, device="cpu", compute_type="int8")
    return _whisper_model_cache[key]


def _five_second_evidence_units(duration_sec, quantum=5.0):
    """Build a deterministic source index: contiguous ~5-second windows.

    The full index is cheap metadata (not 5-second encoded files). Rendering
    later seeks directly into the original source, so this avoids hundreds of
    temporary MP4s while giving every narration decision a stable source
    address.
    """
    try:
        duration = max(0.1, float(duration_sec))
        q = max(2.0, float(quantum))
    except (TypeError, ValueError):
        duration, q = 0.1, 5.0
    units = []
    start = 0.0
    while start < duration - 0.01:
        end = min(duration, start + q)
        units.append((round(start, 3), round(end, 3)))
        start = end
    return units


def _local_fast_detect_beats(local_path, min_beat_sec=12.0, min_cut_sec=1.5, target_duration_sec=0.0):
    """Detect scene cuts. Returns the RAW cuts (only merging truly
    degenerate sub-min_cut_sec fragments into a neighbor) - NOT merged
    into min_beat_sec-sized spans. Each raw cut becomes its own
    scenes_data entry later, so its footage always matches whatever
    narration is written for it exactly; min_beat_sec instead becomes
    pacing GUIDANCE given to Gemini (see local_scene_analysis) for how
    many seconds of cuts to treat as one narrative beat when deciding
    how to distribute a beat's narration across its cuts. Merging cuts
    into one big scenes_data span was the earlier design here, and it
    caused audio/video to only match at a scene's start/end while
    drifting out of content-sync in the middle - the video for a merged
    span is one linear clip, but the narration for it describes several
    distinct moments spread across that span, so what's on screen at any
    given second doesn't track what's being said at that second."""
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import ContentDetector
    video = open_video(local_path)
    sm = SceneManager()
    # PySceneDetect expects min_scene_len as an integer frame count.
    # Passing a value such as "35.00s" causes its internal `>` comparison
    # to fail with: TypeError: '>' not supported between str and int.
    try:
        fps = float(video.frame_rate)
    except (AttributeError, TypeError, ValueError):
        fps = 24.0
    min_scene_len_frames = max(1, int(round(max(1.0, float(min_cut_sec)) * fps)))
    sm.add_detector(ContentDetector(threshold=27.0, min_scene_len=min_scene_len_frames))
    sm.detect_scenes(video=video)
    raw = sm.get_scene_list()
    raw_cuts = [(s.get_seconds(), e.get_seconds()) for s, e in raw]
    if not raw_cuts:
        units = _five_second_evidence_units(video.duration.get_seconds(), quantum=5.0)
        print(f"[local_fast] raw_cuts=0 -> {len(units)} selected five-second source units")
        return units
    # First merge truly tiny fragments (< min_cut_sec) into a neighbor.
    cuts = []
    cur_s, cur_e = raw_cuts[0]
    for s, e in raw_cuts[1:]:
        if cur_e - cur_s < min_cut_sec:
            cur_e = e
        else:
            cuts.append((cur_s, cur_e)); cur_s, cur_e = s, e
    cuts.append((cur_s, cur_e))
    # Build the complete 5-second source index. The camera-cut detector is
    # retained for diagnostics, but fixed contiguous windows are the stable
    # address space used by script/audio mapping and final rendering.
    source_duration = float(video.duration.get_seconds())
    all_units = _five_second_evidence_units(source_duration, quantum=5.0)
    # Keep the complete contiguous address space. Gemini is chunked later for
    # request size, but sampling units here would make a beat range jump over
    # unseen footage and break exact source-to-narration mapping.
    print(f"[local_fast] raw_cuts={len(raw_cuts)} -> {len(cuts)} clean cuts -> {len(all_units)} contiguous five-second source units")
    return all_units


def _local_fast_sec_to_ts(s):
    h = int(s // 3600); m = int((s % 3600) // 60); sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:06.3f}"


def _validate_story_beats(data, unit_count, min_words=0, min_beats=0,
                          soft_min_words=0, soft_min_beats=0):
    """Validate beats without losing source-grounded content.

    The strict budget is a generation target, not a reason to discard a
    valid response. Free-tier models often stop early. If the response still
    contains enough non-overlapping evidence, accept it as a short batch and
    let the global duration gate report the real final mismatch.
    """
    if not isinstance(data, list):
        raise ValueError("Gemini story-beat response must be a JSON array")
    generic_markers = (
        "ဒီအပိုင်းမှာတော့ ဇာတ်လမ်းရဲ့ အခြေအနေကို ဆက်ပြီးဖော်ပြနေပါတယ်",
        "ဒီအပိုင်းမှာ ဇာတ်လမ်းရဲ့ အခြေအနေတစ်ခုကို ဆက်လက်ဖော်ပြထားပြီး",
        "ဇာတ်လမ်းကို ဆက်လက်ဖော်ပြထားပါတယ်",
        "အခြေအနေတစ်ခုကို ပြသထားပါတယ်",
    )
    accepted, used = [], set()
    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            a, b = int(item["start_unit"]), int(item["end_unit"])
        except (KeyError, TypeError, ValueError):
            continue
        raw = str(item.get("raw_script", "") or "").strip()
        narration = str(item.get("narration", "") or "").strip()
        if a < 0 or b < a or b >= unit_count:
            continue
        if used.intersection(range(a, b + 1)):
            continue
        if len(narration) < 25 or len(re.findall(r"\S+", narration)) <= 5:
            continue
        if any(marker in narration for marker in generic_markers):
            continue
        accepted.append({"start_unit": a, "end_unit": b,
                         "raw_script": raw, "narration": narration})
        used.update(range(a, b + 1))
    accepted.sort(key=lambda x: x["start_unit"])
    if not accepted:
        raise ValueError("Gemini returned no meaningful story beats")
    actual_words = sum(len(re.findall(r"\S+", x["narration"])) for x in accepted)
    beat_short = bool(min_beats and len(accepted) < int(min_beats))
    word_short = bool(min_words and actual_words < int(min_words))
    if beat_short or word_short:
        # Evidence-first fallback: do not invent padding, but do not throw
        # away a usable response either. The caller logs this condition and
        # the final TTS duration gate remains authoritative.
        enough_soft_beats = not soft_min_beats or len(accepted) >= int(soft_min_beats)
        enough_soft_words = not soft_min_words or actual_words >= int(soft_min_words)
        if not (enough_soft_beats and enough_soft_words):
            if beat_short:
                raise ValueError(f"Gemini story-beat response too short: {len(accepted)} beats; minimum {int(min_beats)}")
            raise ValueError(f"Gemini story-beat narration too short: {actual_words} words; minimum {int(min_words)}")
    return accepted


def local_scene_analysis(video_id, part_tag, local_path, temp_dir, target_minutes, target_words,
                         target_lang, recap_style, recap_ratio, chunk_label, prev_context,
                         min_beat_sec=20.0):
    """Transcript-first story-beat analysis.

    Five-second windows are source addresses only. Gemini groups consecutive
    windows into meaningful beats and returns separate raw evidence and final
    spoken narration. No generic narration fallback is ever synthesized.
    """
    log_status(video_id, f"{part_tag}Local scene timeline ကို 5-second source units အဖြစ် တည်ဆောက်နေပါသည်...")
    beats = _local_fast_detect_beats(local_path, min_beat_sec,
                                     target_duration_sec=target_minutes * 60.0)
    if not beats:
        raise RuntimeError("No source timeline units were detected")

    log_status(video_id, f"{part_tag}Dialogue transcript ကို local Whisper ဖြင့် ထုတ်နေပါသည်...")
    wav_path = os.path.join(temp_dir, "lf_audio.wav")
    subprocess.run(["ffmpeg", "-y", "-i", local_path, "-ac", "1", "-ar", "16000", wav_path],
                   check=True, capture_output=True)
    whisper_model = _local_fast_get_whisper()
    whisper_segments, _info = whisper_model.transcribe(wav_path, beam_size=1)
    transcript = [{"start": float(x.start), "end": float(x.end), "text": x.text} for x in whisper_segments]

    def dialogue_for(a, b):
        return " ".join(x["text"].strip() for x in transcript
                         if x["end"] > a and x["start"] < b).strip()

    source_units = [
        {"unit_index": i, "source_start": round(float(a), 3),
         "source_end": round(float(b), 3),
         "dialogue": dialogue_for(a, b)[:1200]}
        for i, (a, b) in enumerate(beats)
    ]
    style_block = RECAP_STYLES.get(recap_style, RECAP_STYLES[RECAP_STYLE_DEFAULT])["voice"].format(target_lang=target_lang)
    continuity = ""
    if chunk_label:
        continuity = f"""
CONTINUITY: This is part {chunk_label} of a longer movie. Continue naturally;
do not restart the introduction or conclude the entire movie. Previous ending:
{(prev_context or '')[-700:]}
"""
    # Smaller requests are more reliable on free-tier models and make a
    # truncated response affect only a short source window, not the whole job.
    chunk_size = max(20, min(60, int(os.environ.get("STORY_BEAT_CHUNK_UNITS", "45"))))
    all_beats = []
    failed_chunks = []
    total_chunks = (len(source_units) + chunk_size - 1) // chunk_size

    for chunk_no, offset in enumerate(range(0, len(source_units), chunk_size), 1):
        chunk = source_units[offset:offset + chunk_size]
        chunk_target_words = int(target_words * len(chunk) / max(1, len(source_units))) if target_words else 0
        # Ask Gemini for a narrow first-pass band, but keep the validator floor
        # slightly softer so a valid evidence response is not discarded solely
        # because Burmese tokenization differs from our whitespace count.
        chunk_min_words = int(chunk_target_words * 0.82) if chunk_target_words else 0
        chunk_target_low = int(chunk_target_words * 0.90) if chunk_target_words else 0
        chunk_target_high = int(chunk_target_words * 1.10) if chunk_target_words else 0
        # A word floor by itself is ambiguous: Gemini may return only a few
        # 30-word beats and stop.  Require enough distinct beats to make the
        # floor achievable without generic padding.
        required_beats = max(1, int(math.ceil(chunk_min_words / 30.0))) if chunk_min_words else 1
        # Non-overlapping validation means a chunk cannot contain more beats
        # than source units, especially for the final short chunk.
        required_beats = min(required_beats, len(chunk))
        # Free-tier responses can be shorter even when the evidence is valid.
        # Accept a conservative evidence floor (50% of the strict budget) so
        # one short response cannot erase an entire source chunk. This is not
        # a duration pass: final actual TTS duration is still gated later.
        soft_min_words = max(40, int(chunk_min_words * 0.50)) if chunk_min_words else 40
        soft_min_beats = max(1, int(math.ceil(required_beats * 0.50)))
        prompt = f"""You are an expert movie-recap scriptwriter and story editor.

Convert the timestamped source evidence into meaningful chronological story beats.
The source is divided into fixed five-second units ONLY for exact video mapping.
DO NOT write one narration line per five-second unit.

GROUPING:
- Group consecutive units belonging to the same action, conversation, setting, or story beat.
- A beat may contain several consecutive units.
- Never combine unrelated or non-contiguous units.
- Skip silent, repetitive, and story-irrelevant units.
- Keep the smallest source range that fully supports the narration.

{style_block}
{continuity}
LANGUAGE AND QUALITY:
- Write natural spoken {target_lang}, not textbook, subtitle-like, or literal translation.
- Tell the story: action, consequence, conflict, goal, and stakes when supported.
- Do not write object labels or transcript fragments.
- Never use generic filler such as ဒီအပိုင်းမှာတော့ ဇာတ်လမ်းရဲ့ အခြေအနေကို ဆက်ပြီးဖော်ပြနေပါတယ်.
- Do not invent names, motives, relationships, locations, twists, emotions, or events.
- If evidence is insufficient, omit that unit; never fill it with generic narration.
        - Aim for 18-35 spoken words per meaningful beat.
- Requested total for this chunk is approximately {chunk_target_words} words.
- FIRST-PASS TARGET BAND: finish between {chunk_target_low} and {chunk_target_high} narration words whenever the evidence supports it.
- HARD LENGTH FLOOR: produce at least {chunk_min_words} spoken words for this chunk. Cover more distinct supported actions, consequences, dialogue, and cause/effect from the supplied units when needed. Do not stop after the opening minutes.
- HARD BEAT FLOOR: return at least {required_beats} non-overlapping story beats for this chunk. A response with only 3-5 beats is incomplete even if its JSON is valid.
- Use the full supplied range from local unit 0 through local unit {len(chunk)-1}; do not stop after the first few units. Each beat should normally contain 20-35 spoken words, so the beat count and word floor are both achievable.

FIELDS:
- raw_script: internal factual source note; NOT sent to TTS.
- narration: final spoken recap text; this is the ONLY field sent to TTS.

RETURN ONLY JSON ARRAY:
[
  {{"start_unit": 0, "end_unit": 3, "raw_script": "source-grounded note", "narration": "natural spoken recap"}}
]
The indexes above are LOCAL to this chunk, from 0 through {len(chunk)-1}.

SOURCE EVIDENCE:
{json.dumps(chunk, ensure_ascii=False, indent=2)}"""

        _chunk_attempt = [0]
        def call_chunk():
            _chunk_attempt[0] += 1
            compact_mode = _chunk_attempt[0] > 1
            request_prompt = prompt
            if compact_mode:
                request_prompt += f"\nRETRY: The previous response was malformed, had too few beats, or was below the required length. Return a complete JSON array with at least {required_beats} non-overlapping beats and {chunk_target_low}-{chunk_target_high} narration words (minimum floor {chunk_min_words}). Use 20-35 spoken words per beat, cover the full local unit range, and do not stop after the opening."
            client, model_name = get_gemini_model(purpose="recap")
            response = client.models.generate_content(
                model=model_name,
                contents=request_prompt,
                config=genai_types.GenerateContentConfig(
                    response_mime_type="application/json",
                    max_output_tokens=(8192 if compact_mode else min(8192, max(2048, chunk_target_words * 4 if chunk_target_words else 4096))),
                    response_schema={"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
                        "start_unit": {"type": "INTEGER"}, "end_unit": {"type": "INTEGER"},
                        "raw_script": {"type": "STRING"}, "narration": {"type": "STRING"}},
                        "required": ["start_unit", "end_unit", "raw_script", "narration"]}}
                ))
            data = _safe_response_json(response)
            return _validate_story_beats(
                data, len(chunk), min_words=chunk_min_words, min_beats=required_beats,
                soft_min_words=soft_min_words, soft_min_beats=soft_min_beats
            )

        try:
            local_result = call_gemini_with_retry(
                call_chunk, video_id=video_id, part_tag=part_tag,
                label=f"Story Beat {chunk_no}/{total_chunks}", max_attempts=3, purpose="recap")
            actual_chunk_words = sum(len(re.findall(r"\S+", x.get("narration", ""))) for x in local_result)
            # Spend at most one extra call on a valid but under-length chunk.
            # This completion keeps source ranges/evidence fixed and is only
            # adopted when it produces more grounded narration.
            if chunk_target_words and actual_chunk_words < int(chunk_target_words * 0.85):
                completion_prompt = (
                    f"Complete this source-grounded movie recap chunk. Current narration has {actual_chunk_words} words; "
                    f"target is about {chunk_target_words}. Return ONLY a complete JSON array with the same fields and "
                    f"local indexes 0 through {len(chunk)-1}. Preserve existing supported facts and chronology, expand "
                    f"with supported action, consequence, dialogue context and cause/effect, and add beats only for "
                    f"meaningful uncovered units. Do not invent facts, use generic filler, or write technical labels. "
                    f"Aim for {int(chunk_target_words*0.85)}-{int(chunk_target_words*1.10)} narration words.\n\n"
                    f"CURRENT VALID BEATS:\n{json.dumps(local_result, ensure_ascii=False)}\n\n"
                    f"SOURCE EVIDENCE:\n{json.dumps(chunk, ensure_ascii=False)}"
                )
                try:
                    def _call_completion():
                        client2, model2 = get_gemini_model(purpose="recap")
                        return client2.models.generate_content(
                            model=model2, contents=completion_prompt,
                            config=genai_types.GenerateContentConfig(
                                response_mime_type="application/json",
                                max_output_tokens=min(8192, max(3072, chunk_target_words * 4)),
                                response_schema={"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
                                    "start_unit": {"type": "INTEGER"}, "end_unit": {"type": "INTEGER"},
                                    "raw_script": {"type": "STRING"}, "narration": {"type": "STRING"}},
                                    "required": ["start_unit", "end_unit", "raw_script", "narration"]}}
                            )
                        )
                    completion_response = call_gemini_with_retry(
                        _call_completion, video_id=video_id, part_tag=part_tag,
                        label=f"Story Beat Completion {chunk_no}/{total_chunks}", max_attempts=1, purpose="recap")
                    completion_result = _validate_story_beats(
                        _safe_response_json(completion_response), len(chunk),
                        soft_min_words=soft_min_words, soft_min_beats=soft_min_beats)
                    completion_words = sum(len(re.findall(r"\S+", x.get("narration", ""))) for x in completion_result)
                    if completion_words > actual_chunk_words:
                        local_result, actual_chunk_words = completion_result, completion_words
                        log_status(video_id, f"{part_tag}Story beat {chunk_no}/{total_chunks} targeted completion ဖြင့် {completion_words} words အထိ ဖြည့်ပြီးပါပြီ။")
                except Exception:
                    log_status(video_id, f"{part_tag}Story beat {chunk_no}/{total_chunks} completion မအောင်မြင်သော်လည်း short beats ကို ဆက်သုံးပါမည်။")
            for beat in local_result:
                beat["start_unit"] += offset
                beat["end_unit"] += offset
                all_beats.append(beat)
            if actual_chunk_words < chunk_min_words or len(local_result) < required_beats:
                log_status(video_id, f"{part_tag}Story beat {chunk_no}/{total_chunks} ကို source evidence မပျောက်စေရန် short-but-valid အဖြစ် လက်ခံပါသည် ({actual_chunk_words} words / {chunk_min_words} floor, {len(local_result)} beats)")
            else:
                log_status(video_id, f"{part_tag}Story beat {chunk_no}/{total_chunks} ပြီးပါပြီ ({len(local_result)} beats)")
        except Exception as err:
            failed_chunks.append((chunk_no, str(err)))
            log_status(video_id, f"{part_tag}Story beat {chunk_no}/{total_chunks} မအောင်မြင်ပါ — generic fallback မသုံးဘဲ skip လုပ်ပါမည်။")

    all_beats.sort(key=lambda x: x["start_unit"])
    # Global overlap check across chunks.
    prev_end = -1
    for beat in all_beats:
        if beat["start_unit"] <= prev_end:
            raise RuntimeError("Story-beat ranges overlap after chunk merge")
        prev_end = beat["end_unit"]

    scenes_data = []
    for beat in all_beats:
        a = beat["start_unit"]
        b = beat["end_unit"]
        scenes_data.append({
            "start_time": _local_fast_sec_to_ts(source_units[a]["source_start"]),
            "end_time": _local_fast_sec_to_ts(source_units[b]["source_end"]),
            "raw_script": beat["raw_script"],
            "script": beat["narration"],
            "narration": beat["narration"],
            "evidence_unit_index": a,
            "evidence_unit_end": b,
            "evidence_unit_span": b - a + 1,
        })
    if not scenes_data:
        detail = failed_chunks[0][1] if failed_chunks else "no valid beats"
        raise RuntimeError(f"No meaningful story beats generated: {detail}")
    if failed_chunks:
        log_status(video_id, f"{part_tag}{len(failed_chunks)} story-beat chunk(s) skipped; no filler narration was added.")
    log_status(video_id, f"{part_tag}Meaningful story beats {len(scenes_data)} ခု ရရှိပါပြီ။")
    return scenes_data


def _ranked_local_scene_analysis(video_id, part_tag, local_path, temp_dir,
                                 target_minutes, target_lang, recap_style,
                                 recap_ratio, chunk_label="", prev_context=""):
    """Ranked-local-v2: local selection owns timestamps; Gemini only writes text.

    This deliberately has no per-chunk word floor and no free-form timestamps.
    The source index and selected ranges are deterministic; a failed Gemini
    request falls back to the locally transcribed dialogue instead of failing
    the whole job. The existing renderer consumes the same scenes_data shape.
    """
    log_status(video_id, f"{part_tag}Ranked Local v2: source index နှင့် dialogue ကို local ဖြင့် တည်ဆောက်နေပါသည်...")
    source_units_raw = _local_fast_detect_beats(local_path, min_beat_sec=12.0,
                                                target_duration_sec=target_minutes * 60.0)
    if not source_units_raw:
        raise RuntimeError("Ranked local source index is empty")
    wav_path = os.path.join(temp_dir, "ranked_v2_audio.wav")
    subprocess.run(["ffmpeg", "-y", "-i", local_path, "-ac", "1", "-ar", "16000", wav_path],
                   check=True, capture_output=True)
    whisper_model = _local_fast_get_whisper()
    transcript_segments, _info = whisper_model.transcribe(
        wav_path, beam_size=1, vad_filter=True, language="my" if str(target_lang).lower() in ("my", "burmese") else None
    )
    transcript = [{"start": float(x.start), "end": float(x.end), "text": str(x.text or "").strip()}
                  for x in transcript_segments if str(x.text or "").strip()]

    units = []
    for i, (a, b) in enumerate(source_units_raw):
        dialogue = " ".join(x["text"] for x in transcript if x["end"] > a and x["start"] < b).strip()
        # Transparent local score: speech density + a small position prior.
        # It is a selector only; it never changes source timestamps.
        density = min(1.0, len(dialogue) / 180.0)
        position = i / max(1, len(source_units_raw) - 1)
        units.append({"id": f"u{i:05d}", "unit_index": i,
                      "source_start": round(float(a), 3), "source_end": round(float(b), 3),
                      "dialogue": dialogue[:1200],
                      "importance": round(0.78 * density + 0.22 * (0.5 + 0.5 * math.sin(position * math.pi)), 5)})
    if not any(u["dialogue"] for u in units):
        raise RuntimeError("Ranked local source index has no usable dialogue")

    total_source = units[-1]["source_end"]
    # Output ratio is narration runtime, not the amount of source footage that
    # may be inspected. A 40% recap still needs most of the movie's evidence
    # available so the writer can cover setup, turns, consequences, and ending.
    # Selecting only 40% of source here was the direct cause of short outputs.
    evidence_fraction = max(0.60, min(1.0, (float(recap_ratio) / 100.0) * 2.0))
    target_source = max(5.0, min(total_source, total_source * evidence_fraction))
    selected = set()
    # Mandatory chronological coverage prevents top-score retrieval from
    # selecting only the opening scene.
    for bin_no in range(5):
        lo = int(len(units) * bin_no / 5.0)
        hi = max(lo + 1, int(len(units) * (bin_no + 1) / 5.0))
        candidates = [u for u in units[lo:hi] if u["dialogue"]]
        if candidates:
            selected.add(max(candidates, key=lambda x: x["importance"])["unit_index"])
    selected_duration = sum(units[i]["source_end"] - units[i]["source_start"] for i in selected)
    for u in sorted((x for x in units if x["dialogue"]), key=lambda x: (-x["importance"], x["unit_index"])):
        if selected_duration >= target_source:
            break
        if u["unit_index"] not in selected:
            selected.add(u["unit_index"])
            selected_duration += u["source_end"] - u["source_start"]
    selected_units = [u for u in units if u["unit_index"] in selected]
    selected_units.sort(key=lambda x: x["unit_index"])

    # Merge only adjacent selected units. A beat never jumps across omitted
    # source evidence, which prevents unrelated shots under one narration.
    groups, current = [], []
    for u in selected_units:
        if current and (u["unit_index"] != current[-1]["unit_index"] + 1 or
                         u["source_end"] - current[0]["source_start"] > 25.0):
            groups.append(current); current = []
        current.append(u)
    if current:
        groups.append(current)
    evidence_groups = []
    for gi, group in enumerate(groups):
        evidence_groups.append({"beat_id": f"b{gi:04d}", "unit_ids": [u["id"] for u in group],
                                "source_start": group[0]["source_start"],
                                "source_end": group[-1]["source_end"],
                                "dialogue": " ".join(u["dialogue"] for u in group).strip()[:2400]})

    # Keep requests bounded. The selector has already reduced the source;
    # each request contains only evidence IDs and transcript, never video.
    batch_size = max(8, min(32, int(os.environ.get("RANKED_V2_BATCH_BEATS", "24"))))
    style_block = RECAP_STYLES.get(recap_style, RECAP_STYLES[RECAP_STYLE_DEFAULT])["voice"].format(target_lang=target_lang)
    all_written = []
    for batch_start in range(0, len(evidence_groups), batch_size):
        batch = evidence_groups[batch_start:batch_start + batch_size]
        written = None
        if gemini_client:
            prompt = f"""Write natural spoken {target_lang} movie-recap narration from the supplied evidence ledger.
{style_block}
Return ONLY JSON array. Each object must contain beat_id, evidence_ids, and narration.
Use only the listed evidence IDs. Do not create timestamps, do not invent events, and do not use generic filler.
Keep chronological order, explain action and consequence only when supported, and keep narration concise but complete.
Every beat_id must appear exactly once.

EVIDENCE LEDGER:
{json.dumps(batch, ensure_ascii=False, indent=2)}"""
            def _call_ranked_writer():
                client, model = get_gemini_model(purpose="recap")
                return client.models.generate_content(
                    model=model, contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        response_mime_type="application/json", max_output_tokens=8192,
                        response_schema={"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
                            "beat_id": {"type": "STRING"}, "evidence_ids": {"type": "ARRAY", "items": {"type": "STRING"}},
                            "narration": {"type": "STRING"}},
                            "required": ["beat_id", "evidence_ids", "narration"]}}
                    )
                )
            try:
                response = call_gemini_with_retry(_call_ranked_writer, video_id=video_id,
                    part_tag=part_tag, label=f"Ranked v2 Writer {batch_start // batch_size + 1}",
                    max_attempts=2, purpose="recap")
                data = _safe_response_json(response)
                allowed = {x["beat_id"]: x for x in batch}
                written = []
                for item in data if isinstance(data, list) else []:
                    bid = str(item.get("beat_id", "")); ids = item.get("evidence_ids", [])
                    text = str(item.get("narration", "") or "").strip()
                    if bid not in allowed or not isinstance(ids, list) or not text:
                        continue
                    if not set(str(x) for x in ids).issubset(set(allowed[bid]["unit_ids"])):
                        continue
                    written.append((bid, text))
                if len(written) != len(batch):
                    written = None
            except Exception:
                written = None
        if written is None:
            # Deterministic safety fallback. It is intentionally extractive,
            # never generic or fabricated; the UI/QA can mark it for review.
            written = [(x["beat_id"], x["dialogue"]) for x in batch if x["dialogue"]]
            log_status(video_id, f"{part_tag}Ranked v2 writer မရနိုင်သဖြင့် evidence transcript ကို local fallback အဖြစ်သုံးပါမည်။")
        all_written.extend(written)

    by_id = dict(all_written)
    scenes_data = []
    for item in evidence_groups:
        text = str(by_id.get(item["beat_id"], "") or "").strip()
        if not text:
            continue
        scenes_data.append({"start_time": _local_fast_sec_to_ts(item["source_start"]),
                            "end_time": _local_fast_sec_to_ts(item["source_end"]),
                            "raw_script": item["dialogue"], "script": text, "narration": text,
                            "evidence_ids": item["unit_ids"],
                            "evidence_unit_index": int(item["unit_ids"][0][1:]),
                            "evidence_unit_end": int(item["unit_ids"][-1][1:]),
                            "evidence_unit_span": len(item["unit_ids"])})
    if not scenes_data:
        raise RuntimeError("Ranked local produced no evidence-grounded narration")
    log_status(video_id, f"{part_tag}Ranked Local v2: {len(scenes_data)} evidence beats selected; Gemini writer batches bounded.")
    return scenes_data

def process_video_segment(video_id, local_path, logo_path, font_path, settings,
                           file_suffix="", prev_context="", chunk_label=""):
    """
    Does the actual recap work (analyze -> script -> TTS -> sync -> render)
    for ONE video file. When called from run_advanced_pipeline for a
    multi-part movie, file_suffix separates each part's output files and
    prev_context/chunk_label give Gemini the continuity info it needs to
    keep narrating the same story instead of restarting or wrapping up
    early.
    """
    audio_path   = os.path.join(DOWNLOAD_DIR, f"{video_id}{file_suffix}_tts.mp3")
    srt_path     = os.path.join(DOWNLOAD_DIR, f"{video_id}{file_suffix}_subs.srt")
    final_video  = os.path.join(DOWNLOAD_DIR, f"{video_id}{file_suffix}_final.mp4")
    text_result  = os.path.join(DOWNLOAD_DIR, f"{video_id}{file_suffix}_script.txt")
    temp_dir     = tempfile.mkdtemp(prefix=f"recap_{video_id}{file_suffix}_")
    part_tag     = f"[Part {chunk_label}] " if chunk_label else ""
    result       = {"success": False, "last_script": ""}
    qa = {
        "sync_mode": str(settings.get('sync_mode', 'strict')),
        "fallback_policy": str(settings.get('fallback_policy', 'freeze')),
        "analysis_engine": str(settings.get('analysis_engine', 'local_fast')),
        "mode": "dialogue" if str(settings.get('dialogue_mode', 'false')).strip().lower() in ('true', '1', 'yes', 'on') else "auto",
        "workflow_mode": settings.get('workflow_mode', 'auto'),
        "transform_summary": dict(settings.get('transform_summary') or {}),
        "dialogue_mode": str(settings.get('dialogue_mode', 'false')).strip().lower() in ('true', '1', 'yes', 'on'),
        "dialogue_voice_map": {
            "narrator": settings.get('narrator_voice_id') or settings.get('voice_id'),
            "male": settings.get('male_voice_id'),
            "female": settings.get('female_voice_id'),
        },
        "scenes": 0, "audio_segments": 0, "audio_video_duration_error_ms": None,
        "max_audio_stretch": None, "loops_allowed": str(settings.get('fallback_policy','freeze')) == 'loop',
        "source_duration_sec": None, "target_duration_sec": None, "narration_duration_sec": None,
        "tts_rate": None, "expected_wpm": None, "wpm_source": None,
        "target_words": None, "narration_words": None, "narration_word_ratio": None,
        "target_tolerance": 0.10, "source_coverage_ratio": None, "coverage_bins_hit": 0,
        "story_duration_gate": "pending", "timestamp_gate": "pending",
        "evidence_mapping_gate": "pending", "mapping_gate": "pending", "final_media_gate": "pending",
        "needs_review": False, "warnings": [], "errors": []
    }
    job_qa[video_id] = qa
    try:
        max_audio_stretch = max(1.0, min(1.25, float(settings.get('max_audio_stretch', 1.12))))
    except (TypeError, ValueError):
        max_audio_stretch = 1.12

    if not gemini_client:
        log_status(video_id, f"❌ GOOGLE_API_KEY မရှိပါ။ HuggingFace Space → Settings → Secrets တွင် GOOGLE_API_KEY ထည့်ပါ။")
        return result

    # Source clip duration, used to give Gemini a concrete compression
    # target (e.g. "condense this ~10 minute clip to ~4 minutes of
    # narration") instead of narrating every scene 1:1, which is what made
    # output length simply track source length before.
    try:
        src_dur_sec = float(subprocess.check_output(
            ["ffprobe","-v","error","-show_entries","format=duration",
             "-of","default=noprint_wrappers=1:nokey=1", local_path]
        ).decode().strip())
    except Exception:
        src_dur_sec = 0.0
    if not math.isfinite(src_dur_sec) or src_dur_sec <= 0.05:
        log_status(video_id, f"❌ Source duration မမှန်ပါ ({src_dur_sec!r}s) — target recap duration မတွက်နိုင်ပါ။")
        return result
    try:
        _ratio = float(settings.get('recap_ratio', 40))
        recap_ratio = max(15.0, min(90.0, _ratio)) if math.isfinite(_ratio) else 40.0
    except (TypeError, ValueError):
        recap_ratio = 40.0
    target_minutes = (src_dur_sec / 60.0) * (recap_ratio / 100.0) if src_dur_sec > 0 else 0
    target_duration_sec = target_minutes * 60.0
    if not math.isfinite(target_duration_sec) or target_duration_sec <= 0:
        log_status(video_id, f"❌ Target recap duration မမှန်ပါ ({target_duration_sec!r}s; ratio={recap_ratio!r})")
        return result
    qa['source_duration_sec'] = round(src_dur_sec, 3)
    qa['target_duration_sec'] = round(target_duration_sec, 3)
    # Use the same normalized rate that Edge-TTS receives. Prefer one measured
    # local calibration per voice/rate, then fall back to a conservative
    # Burmese estimate. Actual TTS duration remains the final truth.
    voice_type_for_budget = settings.get('voice_model_type')
    voice_id_for_budget = settings.get('voice_id')
    v_rate_mult = normalize_tts_rate(settings.get('v_rate', 1.0) or 1.0)
    if tts_rate_affects_duration(voice_type_for_budget, voice_id_for_budget):
        expected_wpm, wpm_source = calibrate_tts_wpm(
            voice_type_for_budget, voice_id_for_budget, v_rate_mult,
            settings.get('v_pitch', 0)
        )
    else:
        expected_wpm, wpm_source = 130.0, 'backend_rate_not_applied'
    target_words = int(round(target_minutes * expected_wpm)) if target_minutes > 0 else 0
    qa['tts_rate'] = round(v_rate_mult, 3)
    qa['expected_wpm'] = round(expected_wpm, 2)
    qa['wpm_source'] = wpm_source
    log_status(video_id, f"{part_tag}Target calibration: rate={v_rate_mult:.2f}x, expected={expected_wpm:.1f} WPM ({wpm_source}), target={target_words} words / {target_minutes:.2f} minutes")

    try:
        # ── STEP 1: Analyze video & write recap script ──────────────────────
        target_lang     = settings.get('target_lang', 'Burmese')
        scenes_data     = []
        _dialogue_clusters = []
        analysis_engine = settings.get('analysis_engine', 'local_fast')
        if qa.get('dialogue_mode'):
            log_status(video_id, f"🎙️ Dialogue mode active — narrator={qa['dialogue_voice_map']['narrator']}, male={qa['dialogue_voice_map']['male']}, female={qa['dialogue_voice_map']['female']}")

        # Auto Gemini full-movie mode may provide one merged, globally-timed
        # scene list after analyzing several short evidence clips. In that
        # case this function is the single final TTS/render pass; never call
        # Gemini again and never split the published output into parts.
        if settings.get('_precomputed_scenes'):
            scenes_data = [dict(x) for x in settings['_precomputed_scenes']]
            log_status(video_id, f"{part_tag}Merged full-movie analysis ကို final narration/render အတွက် သုံးပါမည် ({len(scenes_data)} scenes)...")

        if analysis_engine in ('ranked_local_v2', 'local_fast'):
            try:
                if analysis_engine == 'ranked_local_v2':
                    scenes_data = _ranked_local_scene_analysis(
                        video_id=video_id, part_tag=part_tag, local_path=local_path, temp_dir=temp_dir,
                        target_minutes=target_minutes, target_lang=target_lang,
                        recap_style=settings.get('recap_style', RECAP_STYLE_DEFAULT), recap_ratio=recap_ratio,
                        chunk_label=chunk_label, prev_context=prev_context)
                else:
                    scenes_data = local_scene_analysis(
                        video_id=video_id, part_tag=part_tag, local_path=local_path, temp_dir=temp_dir,
                        target_minutes=target_minutes, target_words=target_words, target_lang=target_lang,
                        recap_style=settings.get('recap_style', RECAP_STYLE_DEFAULT), recap_ratio=recap_ratio,
                        chunk_label=chunk_label, prev_context=prev_context,
                        min_beat_sec=float(settings.get('min_beat_sec', 35.0)))
            except Exception as _lf_err:
                # Stable/local-first mode must not silently switch to the expensive
                # Gemini Video uploader. A clear error is safer than a different
                # pipeline with different timestamp behavior and quota usage.
                log_status(video_id, f"{part_tag}❌ Local-Fast analysis မအောင်မြင်ပါ ({_lf_err}) — Gemini Video fallback ကို ပိတ်ထားပါသည်။ Dependencies/transcript ကို စစ်ပါ။")
                raise RuntimeError(f"Local-Fast analysis failed: {_lf_err}") from _lf_err

        if gemini_client and analysis_engine not in ('local_fast', 'ranked_local_v2') and not settings.get('_precomputed_scenes'):
            try:
                log_status(video_id, f"{part_tag}AI မှ ဗီဒီယိုဇာတ်ဝင်ခန်းများကို စက္ကန့်တိကျစွာ ခွဲခြမ်းစိတ်ဖြာနေပါသည်...")

                # Gemini only needs to SEE/HEAR this well enough to identify
                # scenes and write narration - it doesn't need the original's
                # full resolution/bitrate. Uploading a small compressed copy
                # instead cuts both the upload time and Gemini's own
                # file-processing ("PROCESSING" state) time, which matters a
                # lot on a slow/free-tier connection. Audio is kept (just at
                # lower bitrate) in case dialogue informs the narration. The
                # original local_path is untouched and is still what every
                # actual output clip gets re-encoded from later - this copy
                # is thrown away after analysis and never affects quality.
                analysis_path = local_path
                try:
                    _ac = os.path.join(temp_dir, "gemini_analysis_copy.mp4")
                    subprocess.run(["ffmpeg", "-y", "-i", local_path,
                                     "-vf", "scale=-2:480,fps=15",
                                     "-c:v", "libx264", "-preset", "veryfast", "-crf", "30",
                                     "-c:a", "aac", "-b:a", "64k", _ac],
                                    check=True, capture_output=True, timeout=300)
                    analysis_path = _ac
                except Exception as _ac_err:
                    print(f"[Analysis-copy] compression failed, uploading original instead: {_ac_err}")
                    analysis_path = local_path

                vf = gemini_client.files.upload(file=analysis_path)
                def _state_name(f):
                    s = getattr(f, 'state', None)
                    return getattr(s, 'name', s)
                while _state_name(vf) == 'PROCESSING':
                    time.sleep(5); vf = gemini_client.files.get(name=vf.name)

                def _reupload_video_file():
                    # Gemini's File API scopes an uploaded file to whichever
                    # API key/project uploaded it. When quota pressure forces
                    # a rotation to a different key mid-job, that key can't
                    # see the old upload at all ("permission denied / file
                    # may not exist") - the fix is to re-upload under
                    # whichever key is active right now.
                    nonlocal vf
                    new_vf = gemini_client.files.upload(file=analysis_path)
                    while _state_name(new_vf) == 'PROCESSING':
                        time.sleep(5); new_vf = gemini_client.files.get(name=new_vf.name)
                    vf = new_vf

                if _state_name(vf) == 'ACTIVE':
                    continuity_block = ""
                    if chunk_label:
                        continuity_block = f"""
IMPORTANT — CONTINUITY CONTEXT:
This clip is PART {chunk_label} of a single longer movie that has been split
into consecutive parts purely for processing reasons. It is NOT the whole
movie and NOT a standalone short film.
- Do not open with a fresh "introduction" as if this were the start of the
  movie (unless this literally is part 1).
- Do not write a concluding wrap-up ("and that was the movie...") unless
  chunk_label shows this is the FINAL part.
- Simply continue the recap narration naturally, as a direct continuation
  of the story.
{f'- Here is roughly where the previous part left off, for tone/continuity: "{prev_context}"' if prev_context else ''}
"""
                    src_minutes = src_dur_sec / 60.0 if src_dur_sec > 0 else target_minutes
                    # Give Gemini a concrete editorial budget, not only a
                    # duration suggestion. A bounded beat count prevents a
                    # short list of one-line captions from passing the JSON
                    # schema while missing the narration target.
                    _strict_target_words = max(1, int(target_words or 0))
                    _strict_min_words = max(1, int(round(_strict_target_words * 0.85)))
                    _strict_max_words = max(_strict_min_words, int(round(_strict_target_words * 1.15)))
                    _strict_beat_count = max(4, min(14, int(round(_strict_target_words / 34.0)))) if _strict_target_words else 6
                    _strict_min_beat_words = max(14, int(round(_strict_target_words / max(_strict_beat_count, 1) * 0.70))) if _strict_target_words else 14
                    _strict_max_beat_words = max(_strict_min_beat_words + 4, int(round(_strict_target_words / max(_strict_beat_count, 1) * 1.45))) if _strict_target_words else 80
                    length_guidance = (
                        f"This clip is about {src_minutes:.1f} minutes long if fully narrated "
                        f"1:1. At a {recap_ratio:.0f}% recap ratio, your job is to condense it "
                        f"into roughly {target_minutes:.1f} minutes of narration"
                        + (f" — at this ratio, keep MOST of the story: include most scenes and "
                           f"their detail, and only skip moments that are truly redundant or pure "
                           f"filler with no story value. This is not a tight highlight reel."
                           if recap_ratio >= 55 else ".")
                        + f" Concretely: your total word count across ALL entries combined "
                        f"must land between {int(target_words*0.85)} and {int(target_words*1.15)} "
                        f"words — keep a running mental count as you write and adjust (add more "
                        f"beats if you're running short, compress further if you're running long) "
                        f"so you land in that range by the end, rather than checking only at the end. "
                        f"Spread your coverage across the FULL clip duration ({src_minutes:.1f} "
                        f"minutes) — do not concentrate all your narrated scenes in only the first "
                        f"portion of the clip and leave the rest thin."
                        if target_minutes > 0 else
                        "Your job is NOT to narrate every scene - it's to CONDENSE this into a "
                        "tight recap covering only what matters to the story."
                    )
                    _dialogue_mode = str(settings.get('dialogue_mode', 'false')).strip().lower() in ('true', '1', 'yes', 'on')
                    _dialogue_transcript = []
                    _dialogue_clusters = []
                    if _dialogue_mode:
                        # Dialogue is transcript-first: use local ASR as source
                        # evidence before asking Gemini to rewrite anything.
                        # The transcript is bounded and remains evidence only;
                        # it is never passed directly to the TTS renderer.
                        _dialogue_transcript = _dialogue_transcript_evidence(
                            local_path, video_id=video_id, part_tag=part_tag
                        )
                        _dialogue_clusters = _dialogue_transcript_clusters(_dialogue_transcript)
                        qa['dialogue_transcript_segments'] = len(_dialogue_transcript)
                        qa['dialogue_transcript_clusters'] = len(_dialogue_clusters)
                        qa['dialogue_transcript_used'] = bool(_dialogue_transcript)
                    _script_example = '[NARRATOR] Narration text explaining this exact scene...' if _dialogue_mode else 'Narration text explaining this exact scene...'
                    _dialogue_evidence_block = ""
                    if _dialogue_mode:
                        _dialogue_evidence_block = f"""
DIALOGUE SOURCE TRANSCRIPT (local Whisper; evidence only):
{json.dumps(_dialogue_transcript, ensure_ascii=False)}

DIALOGUE TIMING CLUSTERS (these are the source speech intervals that must be
covered by one or more returned scenes):
{json.dumps(_dialogue_clusters, ensure_ascii=False)}

Use this transcript to preserve the source conversation order and wording.
Do not copy the transcript as a raw narration. Rewrite it naturally in the
selected recap style, and do not add a character line unless the video or this
transcript supports it. If two speakers are visibly or audibly distinguishable,
assign [MALE] and [FEMALE] consistently to those two grounded speakers instead
of collapsing the exchange into narration. Use [NARRATOR] only when no speaker
can be grounded or for connective explanation.
"""
                    dialogue_contract = """
DIALOGUE MODE CONTRACT:
- The local Whisper transcript above is the primary speech evidence. The video
  is the visual evidence. Reconcile both before writing the recap.
- Timestamps are RELATIVE TO THIS UPLOADED ANALYSIS CLIP, not the full movie.
- Return at least one scene overlapping every dialogue timing cluster above.
  Do not collapse several separated clusters into one broad scene. If a
  cluster contains multiple turns, keep the scene range tight around that
  exchange and emit multiple tagged blocks in chronological order.
- Write a cinematic mix of narration and source-grounded dialogue.
- Prefix every spoken block with exactly one tag: [NARRATOR], [MALE], or [FEMALE].
- Use [MALE]/[FEMALE] for grounded character turns. If a speech cluster has two or more turns and two speakers can be distinguished, do not label every turn [NARRATOR].
- Every multi-turn speech cluster must contain a short back-and-forth when supported: [NARRATOR] setup, [MALE] line, [FEMALE] response, then [NARRATOR] consequence.
- Keep each character turn short and conversational; never output a raw transcript dump or isolated subtitle fragments.
- Do not invent quotations, names, motives, or dialogue that is not supported by the video/transcript.
- Keep dialogue short and natural, preserve the conversation order, and use narration to connect the exchange to the plot.
- The tags are control metadata, not spoken text; do not count them toward the word budget.
""" if _dialogue_mode else ""
                    prompt = f"""You are a professional movie-recap YouTube scriptwriter with 10 years of
experience creating high-retention CONDENSED recap narration.

Watch this ENTIRE video clip from start to finish. {length_guidance}

HARD NARRATION CONTRACT — DO NOT TREAT THESE AS SUGGESTIONS:
- This window's Python-calculated target is {_strict_target_words} narration words.
- The only accepted total range is {_strict_min_words}-{_strict_max_words} words.
- Return approximately {_strict_beat_count} chronological story beats; never return
  a tiny 1-3 beat summary for a full analysis window.
- Each beat must contain at least {_strict_min_beat_words} words and should normally
  stay below {_strict_max_beat_words} words. The sum of all script fields is the
  contract; do not count timestamps, JSON keys, or this prompt.
- Do not stop early. Before returning, count the script fields yourself and expand
  supported cause, action, reaction, consequence, and character-goal details until
  the total is inside the accepted range.
- If the evidence truly cannot support the budget, return the complete chronological
  window without filler or invented facts. The local validator will decide whether
  the window can proceed.

STRICT OUTPUT CHECKLIST BEFORE YOU RESPOND:
1. Return only a JSON array, with no Markdown or commentary.
2. Every object must have a non-empty start_time, end_time, and script.
3. Keep timestamps chronological and inside this video window.
4. Cover the beginning, middle, and final important event of this window.
5. Do not omit the last story beat merely to finish early.
6. Count only script fields; target {_strict_min_words}-{_strict_max_words} words.

To do this:
- Identify the ESSENTIAL story beats: setup, key relationships, main
  conflicts, turning points, twists, and the ending/resolution. Every one
  of these must be included - the viewer must be able to follow and
  understand the complete story from your recap alone.
- SKIP or heavily compress: repetitive shots, filler dialogue, scene
  transitions with no new story information, and minor moments that don't
  change the plot. Multiple minutes of source footage can become a single
  short narrated beat if nothing story-critical happens.
- Do not pad — every entry should carry real story weight. If in doubt
  whether a moment matters to the plot, prefer leaving it out over
  including it "just in case".
- Still go through the clip in chronological order and cover its FULL
  duration end-to-end (don't stop after the opening) - condensing means
  being selective about WHAT you narrate, not stopping early.
{continuity_block}
{RECAP_STYLES.get(settings.get('recap_style', RECAP_STYLE_DEFAULT), RECAP_STYLES[RECAP_STYLE_DEFAULT])['voice'].format(target_lang=target_lang)}
{_dialogue_evidence_block}
{dialogue_contract}

GLOBAL LANGUAGE RULE (applies no matter which style above is selected):
If {target_lang} is Burmese/Myanmar, write in natural SPOKEN Myanmar only -
sentence-final particles like "...တယ်", "...ပါတယ်", "...ရဲ့", casual
connective words a person would actually say out loud. NEVER use formal
written/literary Myanmar book-style endings such as "...သည်", "...ခဲ့သည်",
"...ဖြစ်သည်" - that register reads like a textbook or news article being
read aloud, not like a person telling a story. This overrides nothing
about tone/formality the style above asks for (a "formal documentary"
style is still formal in word choice and pacing) - it only fixes the
GRAMMATICAL ending register, which must always be the spoken one.

SHARED TECHNICAL RULES (apply regardless of the style above):
- SCENE BOUNDARIES: each entry's start_time/end_time must match ONE
  continuous visual moment or action in the actual footage - do not merge
  several unrelated shots/cuts into a single entry, and do not guess
  timestamps; use the real moments where the visual content changes.
- IMPORTANT: You MUST output ONLY a valid JSON array. No markdown, no
  extra commentary, no text outside the JSON.
- Do not add a reported word-count field as a substitute for writing the
  required narration. Python will count the returned script fields itself.
- Do not use generic filler, repeated sentences, raw transcript fragments,
  object labels, or unsupported events to inflate the count.
- Return the final important event in this window; never stop after only the
  opening just because the first few beats are complete.
- JSON structure must be exactly like this:
[
  {{
    "start_time": "00:00:00",
    "end_time": "00:00:10",
    "script": "{_script_example}"
  }}
]
"""

                    def _call_gemini():
                        _c, _m = get_gemini_model("recap")
                        return _c.models.generate_content(
                            model=_m,
                            contents=[prompt, vf],
                            config=genai_types.GenerateContentConfig(
                                response_mime_type="application/json",
                                max_output_tokens=16384,
                                response_schema={
                                    "type": "ARRAY",
                                    "items": {
                                        "type": "OBJECT",
                                        "properties": {
                                            "start_time": {"type": "STRING"},
                                            "end_time":   {"type": "STRING"},
                                            "script":     {"type": "STRING"},
                                        },
                                        "required": ["start_time", "end_time", "script"],
                                    },
                                },
                            )
                        )
                    response = call_gemini_with_retry(
                        _call_gemini,
                        video_id=video_id, part_tag=part_tag, label="Video Analysis",
                        max_attempts=3, purpose="recap", file_recovery_fn=_reupload_video_file,
                    )
                    translated_text = response.text
                    # Detect Gemini FAILED / safety-blocked responses (empty or block message).
                    cands = getattr(response, "candidates", None) or []
                    if cands:
                        fr = str(getattr(cands[0], "finish_reason", "") or "")
                        if fr.upper() in ("SAFETY", "FINISH_REASON_UNSPECIFIED", "OTHER", "BLOCKED"):
                            raise Exception(f"Gemini blocked the response (finish_reason={fr}). Try a shorter/different script or reduce content.")
                    if not translated_text or not translated_text.strip():
                        raise Exception("Gemini returned an empty response (possible safety block or quota exhaustion).")

                    # Error လုံးဝမတက်နိုင်သော String Replace စနစ်
                    raw_json = translated_text.strip()
                    raw_json = raw_json.replace("```json", "").replace("```", "").strip()

                    try:
                        scenes_data = json.loads(raw_json)
                        log_status(video_id, f"{part_tag}AI မှ ဇာတ်ဝင်ခန်း ({len(scenes_data)}) ခန်းကို အတိအကျ ခွဲထုတ်ပေးလိုက်ပါသည်။")
                    except Exception as json_err:
                        print(f"JSON Parse Error: {json_err}. Attempting recovery...")
                        try:
                            # 1st level Regex Recovery
                            match = re.search(r'\[\s*\{.*\}\s*\]', raw_json, re.DOTALL)
                            if match:
                                scenes_data = json.loads(match.group(0))
                            else:
                                # 2nd level Regex Block Recovery
                                matches = re.finditer(r'\{[^{}]*\}', raw_json)
                                for m in matches:
                                    obj_str = m.group(0)
                                    try:
                                        obj = json.loads(obj_str)
                                        if 'start_time' in obj and 'script' in obj:
                                            scenes_data.append(obj)
                                    except:
                                        st_m = re.search(r'"start_time"\s*:\s*"([^"]+)"', obj_str)
                                        et_m = re.search(r'"end_time"\s*:\s*"([^"]+)"', obj_str)
                                        sc_m = re.search(r'"script"\s*:\s*"(.*?)"\s*\}', obj_str, re.DOTALL)
                                        if st_m and sc_m:
                                            st = st_m.group(1)
                                            et = et_m.group(1) if et_m else st
                                            sc = sc_m.group(1).replace('"', '').replace('\\', '').strip()
                                            scenes_data.append({"start_time": st, "end_time": et, "script": sc})

                            if scenes_data:
                                log_status(video_id, f"{part_tag}AI မှ ဇာတ်ဝင်ခန်း ({len(scenes_data)}) ခန်းကို ခွဲထုတ်ပေးလိုက်ပါသည်။ (Recovered)")
                            else:
                                raise Exception("No valid scenes extracted.")
                        except Exception as e2:
                            raise Exception("AI failed to output valid JSON timestamps.")

                    # ── Post-check: the length instruction above is a steering
                    # suggestion, not an enforced limit, so Gemini can miss the
                    # target badly in either direction (this is what caused
                    # wildly inconsistent part lengths at the same nominal
                    # ratio). One corrective regeneration pass when the miss
                    # is large enough to matter.
                    _gemini_auto_mode = str(settings.get('analysis_engine', '')).strip().lower() == 'gemini_auto'
                    if target_words > 0 and scenes_data and not _gemini_auto_mode:
                        for _fix_round in range(2):
                            actual_words = sum(len(re.findall(r'\S+', str(s.get('script', '')))) for s in scenes_data)
                            ratio = actual_words / target_words if target_words else 1.0
                            if not (ratio < 0.8 or ratio > 1.25):
                                break
                            direction = "too SHORT" if ratio < 1 else "too LONG"
                            log_status(video_id, f"{part_tag}⚠️ Recap length {direction} ({actual_words} words / {target_words} target) — AI ကို ပြန်ညှိခိုင်းနေပါသည် ({_fix_round+1}/2)...")
                            fix_prompt = (
                                f"Your previous attempt at this exact task produced a script with "
                                f"{actual_words} total words, but the target was {target_words} words "
                                f"(covering roughly {target_minutes:.1f} minutes of spoken narration). "
                                f"That is {'far too short' if ratio < 1 else 'far too long'} — "
                                f"{'you must cover significantly MORE of this clip: include more distinct scenes/moments and narrate each in more detail, spread across the full length of the clip, not just the first portion of it' if ratio < 1 else 'cut it down significantly, keeping only the most essential story beats'}.\n\n"
                                f"Redo the FULL task below, but this time actually hit a total word "
                                f"count between {int(target_words*0.85)} and {int(target_words*1.15)} "
                                f"words summed across all entries. Keep a running count as you write "
                                f"so you land in that range — this is a hard requirement, not a "
                                f"suggestion.\n\n{prompt}"
                            )
                            def _call_gemini_fix():
                                _c, _m = get_gemini_model("recap")
                                return _c.models.generate_content(
                                    model=_m,
                                    contents=[fix_prompt, vf],
                                    config=genai_types.GenerateContentConfig(
                                        response_mime_type="application/json",
                                        max_output_tokens=16384,
                                        response_schema={
                                            "type": "ARRAY",
                                            "items": {
                                                "type": "OBJECT",
                                                "properties": {
                                                    "start_time": {"type": "STRING"},
                                                    "end_time":   {"type": "STRING"},
                                                    "script":     {"type": "STRING"},
                                                },
                                                "required": ["start_time", "end_time", "script"],
                                            },
                                        },
                                    )
                                )
                            try:
                                fix_response = call_gemini_with_retry(
                                    _call_gemini_fix, video_id=video_id, part_tag=part_tag,
                                    label="Length Correction", max_attempts=3, purpose="recap",
                                    file_recovery_fn=_reupload_video_file,
                                )
                                fix_scenes = _safe_response_json(fix_response)
                                fix_words = sum(len(str(s.get('script', '')).split()) for s in fix_scenes)
                                # Only switch to the corrected version if it actually landed closer to target
                                if fix_scenes and abs(fix_words - target_words) < abs(actual_words - target_words):
                                    scenes_data = fix_scenes
                                    log_status(video_id, f"{part_tag}✅ Length ပြင်ဆင်ပြီးပါပြီ ({fix_words} words) — ဆက်လုပ်ပါမည်။")
                                else:
                                    log_status(video_id, f"{part_tag}Length ပြင်ဆင်မှု ပိုမကောင်းခဲ့လို့ မူလ script ကိုပဲ ဆက်သုံးပါမည်။")
                                    break
                            except Exception as fix_err:
                                print(f"[LENGTH FIX] Corrective pass failed, keeping original: {fix_err}")
                                break

                    try:
                        gemini_client.files.delete(name=vf.name)
                    except Exception:
                        pass

                    if target_words > 0 and scenes_data:
                        final_words = sum(len(re.findall(r'\S+', str(s.get('script', '')))) for s in scenes_data)
                        final_ratio = final_words / target_words if target_words else 1.0
                        if final_ratio < 0.6:
                            log_status(video_id, f"{part_tag}⚠️ Note: ဒီ part ရဲ့ script က target ရဲ့ {final_ratio*100:.0f}% ပဲ ရောက်ပါသေးတယ် (correction 2 ကြိမ်ပြီးလည်း) — clip ဒီအပိုင်းထဲမှာ ဇာတ်လမ်းအရေးပါတဲ့ content နည်းနေတာ ဖြစ်နိုင်ပါတယ်။")

            except Exception as e:
                log_status(video_id, f"❌ {part_tag}AI Video Understanding Error: {e}")
                # BUG FIX: previously fell back to a single 1-minute placeholder
                # scene with generic apology text. Since the final video's
                # length is driven by the TTS audio duration (not the declared
                # scene end_time), that short placeholder narration produced a
                # ~2 second output that still looked like a "successful" run.
                # Failing loudly here is more honest than silently shipping a
                # broken 2-second video.
                raise Exception(f"Video analysis failed, no recap was generated: {e}")

        # One final text-only edit pass fixes disconnected captions, repeated
        # hooks, and unnatural Burmese without touching timestamps/evidence.
        # Older/Gemini-video scenes may not have the new raw_script field.
        # Normalize them without changing their narration text.
        for _sc in scenes_data:
            _sc.setdefault('raw_script', str(_sc.get('script', '') or '').strip())
        if qa.get('dialogue_mode'):
            _tagged_dialogue_scenes = sum(
                1 for _sc in scenes_data
                if any(role != 'NARRATOR' for role, _txt in _dialogue_voice_lines(_sc.get('script', '')))
            )
            qa['dialogue_tagged_scene_count'] = _tagged_dialogue_scenes
            qa['dialogue_scene_count'] = len(scenes_data)
            _covered_clusters = 0
            for _cluster in _dialogue_clusters:
                try:
                    _cs, _ce = float(_cluster['start']), float(_cluster['end'])
                    if any(
                        parse_time_to_sec(_sc.get('start_time')) < _ce and
                        parse_time_to_sec(_sc.get('end_time')) > _cs
                        for _sc in scenes_data
                    ):
                        _covered_clusters += 1
                except (TypeError, ValueError, KeyError):
                    continue
            qa['dialogue_clusters_covered'] = _covered_clusters
            qa['dialogue_cluster_coverage_ratio'] = round(
                _covered_clusters / len(_dialogue_clusters), 4
            ) if _dialogue_clusters else None
            if str(settings.get('recap_style', '')).strip() == 'short_drama_translation':
                if _dialogue_clusters and _tagged_dialogue_scenes == 0:
                    qa['needs_review'] = True
                    qa['warnings'].append('Short-drama format produced no character voice tags; output is narrator-only.')
                elif _dialogue_clusters and _covered_clusters < len(_dialogue_clusters):
                    qa['warnings'].append('Some source speech clusters are not represented in Dialogue scenes.')
            if _dialogue_clusters and _covered_clusters < len(_dialogue_clusters):
                qa['needs_review'] = True
                qa['warnings'].append(
                    f"Dialogue visual coverage is {_covered_clusters}/{len(_dialogue_clusters)} speech clusters"
                )
            if _tagged_dialogue_scenes == 0:
                log_status(video_id, f"⚠️ Dialogue mode မှာ Gemini က tagged character dialogue မထုတ်ပေးသေးပါ — {len(scenes_data)} scenes ကို narrator voice နဲ့သာ TTS လုပ်ပါမည်။")
            else:
                log_status(video_id, f"🎭 Dialogue tags ရရှိပါပြီ — {_tagged_dialogue_scenes}/{len(scenes_data)} scenes မှာ character voice ပြောင်းသုံးပါမည်။")
        _before_polish = [dict(_s) for _s in scenes_data]
        _ranked_v2_analysis = str(settings.get('analysis_engine', 'local_fast')).strip().lower() in ('ranked_local_v2', 'gemini_auto')
        if not _ranked_v2_analysis:
            scenes_data = polish_recap_script(
                scenes_data, target_lang, settings.get('recap_style', RECAP_STYLE_DEFAULT),
                video_id, part_tag=part_tag
            )
        else:
            log_status(video_id, f"{part_tag}ရွေးထားသော analysis script ကို ထပ်မပြောင်းဘဲ TTS သို့ ပို့ပါမည်။")
        # Editorial polish must improve delivery, never reduce narration to
        # subtitle fragments or object labels. Keep the stronger pre-polish
        # sentence when the editor returns an unusably short line.
        _short_after_polish = 0
        for _i, _sc in enumerate(scenes_data):
            _new_text = str(_sc.get('script', '') or '').strip()
            _old_text = str(_before_polish[_i].get('script', '') or '').strip() if _i < len(_before_polish) else ''
            _new_words = len(re.findall(r'\S+', _new_text))
            _old_words = len(re.findall(r'\S+', _old_text))
            _generic_polish = any(x in _new_text for x in (
                'ဒီအပိုင်းမှာတော့ ဇာတ်လမ်းရဲ့ အခြေအနေကို ဆက်ပြီးဖော်ပြနေပါတယ်',
                'ဒီအပိုင်းမှာ ဇာတ်လမ်းရဲ့ အခြေအနေတစ်ခုကို ဆက်လက်ဖော်ပြထားပြီး',
            ))
            if _new_words <= 3 or len(_new_text) < 18 or _generic_polish:
                _short_after_polish += 1
                if len(_old_text) >= 18 and not any(x in _old_text for x in (
                    'ဒီအပိုင်းမှာတော့ ဇာတ်လမ်းရဲ့ အခြေအနေကို ဆက်ပြီးဖော်ပြနေပါတယ်',
                    'ဒီအပိုင်းမှာ ဇာတ်လမ်းရဲ့ အခြေအနေတစ်ခုကို ဆက်လက်ဖော်ပြထားပြီး',
                )) and (_old_words > _new_words or _generic_polish):
                    _sc['script'] = _old_text
            _sc['narration'] = str(_sc.get('script', '') or '').strip()
        if _short_after_polish:
            log_status(video_id, f"{part_tag}⚠️ Editorial polish က narration fragment {_short_after_polish} ခု ထုတ်ခဲ့လို့ မူရင်းစာကြောင်းကို ထိန်းထားပါသည်။")

        # Gemini receives target_words in its prompt, but only the returned
        # narration can prove whether the request was followed. Count the
        # exact script field that will be sent to TTS and expose the ratio in QA.
        _narration_words = sum(
            len(re.findall(r"\S+", str(_sc.get('script', '') or '')))
            for _sc in scenes_data
        )
        qa['target_words'] = int(target_words) if target_words else 0
        qa['narration_words'] = int(_narration_words)
        qa['narration_word_ratio'] = round(
            (_narration_words / target_words) if target_words else 1.0, 4
        )
        if target_words:
            log_status(video_id, f"{part_tag}Gemini narration target check: {_narration_words}/{target_words} words ({_narration_words / target_words * 100:.1f}%)")

        with open(text_result, 'w', encoding='utf-8') as f:
            for s in scenes_data:
                f.write(f"[{s.get('start_time')} - {s.get('end_time')}] {s.get('script')}\n")
        raw_text_result = os.path.join(DOWNLOAD_DIR, f"{video_id}{file_suffix}_raw_script.txt")
        with open(raw_text_result, 'w', encoding='utf-8') as f:
            for s in scenes_data:
                f.write(f"[{s.get('start_time')} - {s.get('end_time')}] {s.get('raw_script', '')}\n")

        # Carry the tail of this part's narration forward so the next part
        # (if any) can continue the story smoothly instead of starting cold.
        if scenes_data:
            tail = " ".join(s.get('script', '') for s in scenes_data[-2:])
            result["last_script"] = tail[-600:]

        # Gemini occasionally returns timestamps from the wider source/movie
        # context even though it was given a 5-minute analysis clip (for
        # example 00:00-26:00 for a 00:00-05:00 clip). Auto analysis windows
        # must never fail the whole job for that model formatting error. Clamp
        # the suggested range to the actual local clip; the renderer still
        # owns the final source clock and the warning remains visible in QA.
        _is_auto_analysis_window = bool(settings.get('_analysis_only')) and str(
            settings.get('analysis_engine', '')).strip().lower() == 'gemini_auto'
        if _is_auto_analysis_window:
            _clamped_scenes = []
            _clamped_count = 0
            for _sc in scenes_data:
                try:
                    _a = parse_time_to_sec(_sc.get('start_time'))
                    _b = parse_time_to_sec(_sc.get('end_time'))
                    _ca = max(0.0, min(src_dur_sec, _a))
                    _cb = max(0.0, min(src_dur_sec, _b))
                    if abs(_ca - _a) > 0.01 or abs(_cb - _b) > 0.01:
                        _clamped_count += 1
                    if _cb <= _ca + 0.05:
                        continue
                    _item = dict(_sc)
                    _item['start_time'] = format_timestamp(_ca)
                    _item['end_time'] = format_timestamp(_cb)
                    _clamped_scenes.append(_item)
                except Exception:
                    _clamped_scenes.append(_sc)
            if _clamped_count:
                scenes_data = _clamped_scenes
                qa['needs_review'] = True
                qa['warnings'].append(
                    f"Gemini returned {_clamped_count} out-of-range timestamp(s); "
                    f"clamped to this {src_dur_sec:.1f}s analysis window"
                )
                log_status(
                    video_id,
                    f"{part_tag}⚠️ Gemini timestamp { _clamped_count } ခုကို "
                    f"analysis window {src_dur_sec:.1f}s အတွင်း deterministic clamp လုပ်ထားပါသည်။"
                )
            if not scenes_data:
                raise Exception("Timestamp normalization removed every scene in the Auto analysis window")

        timeline_check = validate_scene_timeline(scenes_data, src_dur_sec)
        qa['timestamp_gate'] = 'pass' if not timeline_check['errors'] else 'fail'
        qa['timestamp_warnings'] = timeline_check['warnings'][:20]
        if timeline_check['errors']:
            qa['errors'].extend(timeline_check['errors'][:20])
            raise Exception(f"Timestamp validation failed: {timeline_check['errors'][0]}")
        qa['warnings'].extend(timeline_check['warnings'][:20])

        # Local-fast story beats may cover several consecutive five-second
        # source units. Validate the range, chronology, and non-overlap rather
        # than requiring the obsolete one-unit-per-narration shape.
        if str(settings.get('analysis_engine', 'local_fast')).strip().lower() in ('local_fast', 'ranked_local_v2'):
            evidence_errors = []
            ranges = []
            for i, sc in enumerate(scenes_data):
                a = sc.get('evidence_unit_index')
                b = sc.get('evidence_unit_end')
                span = sc.get('evidence_unit_span')
                if not isinstance(a, int) or not isinstance(b, int) or b < a or span != b - a + 1:
                    evidence_errors.append(f"scene {i+1}: invalid story-beat evidence range")
                else:
                    ranges.append((a, b))
            for prev, cur in zip(ranges, ranges[1:]):
                if cur[0] <= prev[1]:
                    evidence_errors.append('local-fast story-beat evidence ranges overlap or are out of order')
                    break
            qa['evidence_mapping_gate'] = 'pass' if not evidence_errors else 'fail'
            if evidence_errors:
                qa['errors'].extend(evidence_errors[:20])
                raise Exception(f"Evidence mapping validation failed: {evidence_errors[0]}")
        else:
            qa['evidence_mapping_gate'] = 'not_applicable'

        def _coverage_metrics(scenes, source_duration):
            if not source_duration or source_duration <= 0:
                return {"ratio": 0.0, "bins_hit": 0, "ranges": 0}
            ranges = []
            for _sc in scenes:
                try:
                    a = max(0.0, min(source_duration, parse_time_to_sec(_sc.get('start_time'))))
                    b = max(a, min(source_duration, parse_time_to_sec(_sc.get('end_time'))))
                    if b - a > 0.05: ranges.append((a, b))
                except Exception:
                    continue
            ranges.sort()
            merged = []
            for a, b in ranges:
                if merged and a <= merged[-1][1] + 0.05:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], b))
                else:
                    merged.append((a, b))
            covered = sum(b-a for a,b in merged)
            bins_hit = 0
            for bi in range(5):
                lo, hi = source_duration * bi / 5.0, source_duration * (bi+1) / 5.0
                if any(b > lo + 0.25 and a < hi - 0.25 for a,b in ranges): bins_hit += 1
            return {"ratio": round(min(1.0, covered/source_duration), 4), "bins_hit": bins_hit, "ranges": len(ranges)}

        cov = _coverage_metrics(scenes_data, src_dur_sec)
        qa['source_coverage_ratio'] = cov['ratio']
        qa['coverage_bins_hit'] = cov['bins_hit']
        required_bins = 5 if src_dur_sec >= 1800 else 4
        if cov['bins_hit'] < required_bins:
            qa['needs_review'] = True
            qa['warnings'].append(f"Story coverage only reaches {cov['bins_hit']}/5 timeline zones (required {required_bins}/5)")
            # Coverage is a story-quality warning, not an A/V rendering error.
            # Do not discard an otherwise correctly synced recap just because
            # Gemini returned too few narrative ranges. Only zero coverage is
            # unsafe because it would produce narration with no source evidence.
            if cov['bins_hit'] == 0:
                raise Exception(f"Coverage gate failed: only {cov['bins_hit']}/5 source timeline zones have narration")
        else:
            qa['mapping_gate'] = 'pass'

        # Used only by the Auto Gemini analysis stage. It validates and
        # returns the script/timestamps without TTS, scene rendering, or any
        # published MP4. The caller later merges all chunk results and runs
        # exactly one final narration/render pass over the original movie.
        if settings.get('_analysis_only'):
            result['success'] = True
            result['analysis_scenes'] = scenes_data
            result['qa'] = qa
            # Analysis windows are private working steps. Their intermediate
            # script files must not appear as downloadable Auto outputs.
            for _p in (text_result, raw_text_result):
                try:
                    if os.path.exists(_p):
                        os.remove(_p)
                except Exception:
                    pass
            return result

        qa['scenes'] = len(scenes_data)
        # Auditable handoff: this is the exact text that will be sent to TTS.
        # Keeping it beside the generated script makes it possible to prove
        # whether a narration problem came before or inside voice generation.
        tts_manifest_path = os.path.join(DOWNLOAD_DIR, f"{video_id}{file_suffix}_tts_input.json")
        tts_manifest = []
        for _i, _sc in enumerate(scenes_data):
            _text = str(_sc.get('script', '') or '').strip()
            if not _text:
                raise Exception(f"Narration input is empty for scene {_i + 1}")
            tts_manifest.append({
                "scene_index": _i,
                "source_start": _sc.get('start_time'),
                "source_end": _sc.get('end_time'),
                "evidence_unit_index": _sc.get('evidence_unit_index'),
                "raw_script": str(_sc.get('raw_script', '') or '').strip(),
                "script_chars": len(_text),
                "narration": _text,
            })
        with open(tts_manifest_path, 'w', encoding='utf-8') as _mf:
            json.dump(tts_manifest, _mf, ensure_ascii=False, indent=2)
        qa['tts_input_manifest'] = tts_manifest_path
        log_status(video_id, f"{part_tag}AI Voice အသံများ သွင်းယူနေပါသည်... (ဇာတ်ဝင်ခန်း {len(scenes_data)} ခု)")

        async def _tts_all(scenes):
            # Edge-TTS commonly returns an empty stream when several Burmese
            # websocket requests are opened together. One request at a time is
            # slower but materially more reliable; override only after testing.
            try:
                _tts_concurrency = max(1, min(2, int(os.environ.get('TTS_CONCURRENCY', '1'))))
            except (TypeError, ValueError):
                _tts_concurrency = 1
            sem = asyncio.Semaphore(_tts_concurrency)
            _dialogue_mode = str(settings.get('dialogue_mode', 'false')).strip().lower() in ('true', '1', 'yes', 'on')
            _dialogue_voices = {
                'NARRATOR': settings.get('narrator_voice_id') or settings.get('voice_id') or 'my-MM-NilarNeural',
                'MALE': settings.get('male_voice_id') or 'my-MM-ThihaNeural',
                'FEMALE': settings.get('female_voice_id') or 'my-MM-NilarNeural',
            }

            async def _one_dialogue(i, sc, out):
                lines = _dialogue_voice_lines(sc.get('script', ''))
                if not lines:
                    raise ValueError(f'empty dialogue script at scene {i}')
                line_paths, line_timings = [], []
                cursor = 0.0
                for li, (role, text_part) in enumerate(lines):
                    lp = os.path.join(temp_dir, f"seg_{i}_line_{li}.mp3")
                    words = await generate_audio_only(
                        text_part, settings.get('voice_model_type', 'edge'),
                        _dialogue_voices.get(role, _dialogue_voices['NARRATOR']), lp,
                        settings.get('v_rate'), settings.get('v_pitch')
                    )
                    line_paths.append(lp)
                    for w in (words or []):
                        line_timings.append({**w, 'start': w['start'] + cursor, 'end': w['end'] + cursor})
                    cursor += probe_dur(lp)
                    if li < len(lines) - 1:
                        pause = os.path.join(temp_dir, f"seg_{i}_pause_{li}.mp3")
                        run_cmd(["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono", "-t", "0.15", "-c:a", "libmp3lame", "-b:a", "64k", pause], "Dialogue pause", video_id)
                        line_paths.append(pause)
                        cursor += probe_dur(pause)
                list_path = os.path.join(temp_dir, f"seg_{i}_dialogue_concat.txt")
                with open(list_path, 'w', encoding='utf-8') as lf:
                    for lp in line_paths:
                        lf.write("file '" + lp.replace("'", "'\\''") + "'\n")
                run_cmd(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path, "-c:a", "libmp3lame", "-b:a", "128k", out], "Dialogue Audio Concat", video_id)
                return out, line_timings

            async def _one(i, sc):
                out = os.path.join(temp_dir, f"seg_{i}.mp3")
                async with sem:
                    try:
                        _tts_text = _spoken_script_text(sc.get('script', '') if _dialogue_mode else str(sc.get('script', '') or '').strip())
                        if not _tts_text:
                            raise ValueError(f"empty script at scene {i}")
                        log_status(video_id, f"{part_tag}TTS scene {i+1}/{len(scenes)} သို့ script {len(_tts_text)} chars ပို့နေပါသည်...")
                        if _dialogue_mode:
                            out, words = await _one_dialogue(i, sc, out)
                        else:
                            words = await generate_audio_only(_tts_text, settings.get('voice_model_type'), settings.get('voice_id'), out, settings.get('v_rate'), settings.get('v_pitch'))
                        return (out, words)
                    except Exception as _tts_err:
                        print(f"[TTS] scene {i} failed: {_tts_err}")
                        return (None, None)
            return await asyncio.gather(*[_one(i,sc) for i,sc in enumerate(scenes)])

        def probe_dur(p):
            return float(subprocess.check_output(["ffprobe","-v","error","-show_entries","format=duration","-of","default=noprint_wrappers=1:nokey=1", p]).decode().strip())

        def _repair_scripts_for_target(scenes, target_sec, actual_sec, round_no):
            if not gemini_client:
                return None
            current_words = _script_word_count(scenes)
            # Once at least one TTS pass exists, its measured WPM is more useful
            # than the initial calibration for the next repair. This prevents a
            # 4.3-minute result from being repaired with an undersized word
            # budget when the voice is slower than the calibration sample.
            measured_wpm = (current_words / (actual_sec / 60.0)) if current_words and actual_sec > 0.1 else float(expected_wpm or 130.0)
            repair_target_words = max(1, int(round((target_sec / 60.0) * measured_wpm)))
            repair_min_words = max(1, int(round(repair_target_words * 0.85)))
            repair_max_words = max(repair_min_words, int(round(repair_target_words * 1.15)))
            compact = [{"index": i, "start_time": sc.get('start_time'), "end_time": sc.get('end_time'), "script": sc.get('script','')} for i, sc in enumerate(scenes)]
            prompt = f"""You are repairing a movie recap narration timeline. The target spoken duration is {target_sec:.1f} seconds, but the current script produces {actual_sec:.1f} seconds. This is repair pass {round_no}.
Return ONLY a JSON array with one object per input index: {{"index": integer, "script": string}}.
The current script contains {current_words} words. The measured TTS rate is {measured_wpm:.1f} words/minute. Therefore the repaired narration must contain {repair_target_words} words, with an acceptable range of {repair_min_words}-{repair_max_words} words. This word budget is a hard requirement, not a suggestion.
Keep every index, source time range, and story-beat evidence mapping unchanged. Expand or compress only the existing source-grounded narration. Do not invent events, motives, names, relationships, or details. Never use generic filler such as "ဒီအပိုင်းမှာ..." or object labels. If the evidence cannot support the target duration, preserve accuracy and accept a shorter narration instead of padding. Keep natural spoken {target_lang} and preserve the selected recap style. Do not delete the final story beats or move narration into the opening.

CURRENT SCENES:
{json.dumps(compact, ensure_ascii=False)}"""
            def _call_repair():
                c, m = get_gemini_model('recap')
                return c.models.generate_content(model=m, contents=prompt, config=genai_types.GenerateContentConfig(
                    response_mime_type='application/json',
                    max_output_tokens=16384, response_schema={"type":"ARRAY","items":{"type":"OBJECT","properties":{"index":{"type":"INTEGER"},"script":{"type":"STRING"}},"required":["index","script"]}}))
            try:
                resp = call_gemini_with_retry(_call_repair, video_id=video_id, part_tag=part_tag, label='Duration Repair', max_attempts=3, purpose='recap')
                data = _safe_response_json(resp)
                by_idx = {int(x.get('index')): str(x.get('script','')).strip() for x in data if str(x.get('script','')).strip()}
                if len(by_idx) != len(scenes): return None
                return [{**sc, 'script': by_idx[i], 'narration': by_idx[i]} for i, sc in enumerate(scenes)]
            except Exception as _repair_err:
                print(f"[DURATION REPAIR] failed: {_repair_err}")
                return None

        # TTS is the source of truth for Burmese duration. If it undershoots the
        # requested recap ratio, repair the script and synthesize again; never
        # silently publish a 3-minute file for an 11-minute target.
        tts_results = None
        valid_data = []
        _ranked_v2 = str(settings.get('analysis_engine', 'local_fast')).strip().lower() == 'ranked_local_v2'
        # Ranked Local v2 deliberately does not enter the old four-pass
        # duration-repair loop: rewriting the same evidence cannot create new
        # supported story content and wastes Gemini quota. Its measured audio
        # is rendered as-is and reported as needs_review when outside target.
        _duration_passes = 1 if _ranked_v2 else 4
        for _duration_pass in range(_duration_passes):
            loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
            try: tts_results = loop.run_until_complete(_tts_all(scenes_data))
            finally: loop.close()
            if _ranked_v2:
                _failed_tts = [i for i, (_p, _w) in enumerate(tts_results) if not _p or not os.path.exists(_p)]
                if _failed_tts:
                    qa['needs_review'] = True
                    qa['warnings'].append(
                        f"TTS provider returned no audio for {len(_failed_tts)} scene(s); those evidence scenes were skipped"
                    )
                    log_status(video_id, f"{part_tag}TTS မရသော scene {len(_failed_tts)} ခုကို skip လုပ်ပြီး ကျန်တဲ့ source-grounded scenes နဲ့ ဆက်လုပ်ပါမည်။")
                    scenes_data = [sc for i, sc in enumerate(scenes_data) if i not in set(_failed_tts)]
                    tts_results = [result for i, result in enumerate(tts_results) if i not in set(_failed_tts)]
                    if not scenes_data:
                        raise Exception("TTS provider returned no usable audio for any scene")
            valid_data = [(sc, p, w) for sc, (p, w) in zip(scenes_data, tts_results) if p and os.path.exists(p)]
            if len(valid_data) != len(scenes_data):
                raise Exception(f"TTS failed for {len(scenes_data)-len(valid_data)} scene(s); refusing partial recap")
            actual_sec = sum(probe_dur(p) for _, p, _ in valid_data)
            qa['narration_duration_sec'] = round(actual_sec, 3)
            # The target is a planning contract, not a reason to reject a
            # usable recap for a small TTS/WPM difference. Keep the normal
            # pass band tight, but only make the job fatal when the measured
            # narration is materially short or materially long
            # (above 125%) after the bounded repair attempts.
            lower, upper = target_duration_sec * 0.90, target_duration_sec * 1.10
            _auto_min_ratio = 0.30 if str(settings.get('analysis_engine', '')).strip().lower() == 'gemini_auto' else 0.70
            soft_lower = target_duration_sec * _auto_min_ratio
            soft_upper = target_duration_sec * 1.25
            if target_duration_sec <= 0 or lower <= actual_sec <= upper:
                qa['story_duration_gate'] = 'pass'
                break
            qa['story_duration_gate'] = 'repairing'
            if actual_sec > upper:
                # A long narration can be safely repaired by the same bounded
                # Gemini pass; the source evidence and ranges remain unchanged.
                log_status(video_id, f"{part_tag}Narration {actual_sec/60:.1f}m သည် target {target_minutes:.1f}m ထက်ရှည်နေပါသည် — script ချုံ့နေပါသည် ({_duration_pass+1}/4)...")
            else:
                log_status(video_id, f"{part_tag}Narration {actual_sec/60:.1f}m သည် target {target_minutes:.1f}m ထက်တိုနေပါသည် — story detail ဖြည့်နေပါသည် ({_duration_pass+1}/4)...")
            repaired = None if _ranked_v2 else _repair_scripts_for_target(scenes_data, target_duration_sec, actual_sec, _duration_pass+1)
            if not repaired:
                # Duration repair is an optimization, not evidence validation.
                # If Gemini is unavailable/retired and the mismatch is modest,
                # keep the already valid, source-grounded narration and render
                # it instead of turning a usable recap into a hard error.
                mismatch = abs(actual_sec - target_duration_sec) / max(target_duration_sec, 1.0)
                if actual_sec >= soft_lower and actual_sec <= soft_upper:
                    qa['story_duration_gate'] = 'pass_with_warning'
                    qa['needs_review'] = True
                    qa['warnings'].append(
                        f"Narration is near target; measured duration kept without unsupported padding "
                        f"({actual_sec:.1f}s vs {target_duration_sec:.1f}s target; "
                        f"{actual_sec / max(target_duration_sec, 0.001) * 100:.1f}%)"
                    )
                    log_status(video_id, f"{part_tag}Narration target မပြည့်သေးသော်လည်း measured {actual_sec/60:.1f}m ကို warning နဲ့ ဆက်သုံးပါမည်။ Fatal error သတ်မှတ်ချက်မှာ target ၏ {_auto_min_ratio * 100:.0f}% အောက် ဖြစ်ပါသည်။")
                    break
                # Do not turn a usable source-grounded recap into a failed
                # job merely because Gemini could not expand/repair it. The
                # requested duration is a planning target, not proof that
                # missing story content can be invented safely. For Auto,
                # render any non-empty validated narration and expose the
                # mismatch in QA; only empty/failed TTS remains fatal.
                if (str(settings.get('analysis_engine', '')).strip().lower() == 'gemini_auto'
                        and actual_sec > 0.5 and valid_data):
                    qa['story_duration_gate'] = 'pass_with_warning'
                    qa['needs_review'] = True
                    qa['warnings'].append(
                        f"Auto narration kept at {actual_sec:.1f}s although target was "
                        f"{target_duration_sec:.1f}s; no unsupported filler was added"
                    )
                    log_status(
                        video_id,
                        f"{part_tag}⚠️ Auto narration target မပြည့်သေးသော်လည်း "
                        f"grounded narration {actual_sec/60:.1f}m ကို warning နဲ့ render ဆက်လုပ်ပါမည်။"
                    )
                    break
                qa['story_duration_gate'] = 'fail'
                qa['needs_review'] = True
                qa['errors'].append(
                    f"Narration target not reached after repair: {actual_sec:.1f}s vs {target_duration_sec:.1f}s target"
                )
                raise Exception(
                    f"Narration target materially missed after repair: "
                    f"{actual_sec/60:.1f}m vs {target_minutes:.1f}m target "
                    f"({actual_sec / max(target_duration_sec, 0.001) * 100:.1f}%; fatal below {_auto_min_ratio * 100:.0f}%)"
                )
            scenes_data = repaired
        else:
            pass

        # A repair response may be returned on the last allowed pass. Recheck
        # the measured result once more before declaring failure.
        _final_measured_sec = float(qa.get('narration_duration_sec') or 0.0)
        _final_auto_min_ratio = 0.30 if str(settings.get('analysis_engine', '')).strip().lower() == 'gemini_auto' else 0.70
        _soft_lower_sec = target_duration_sec * _final_auto_min_ratio
        _soft_upper_sec = target_duration_sec * 1.25
        if (qa.get('story_duration_gate') == 'repairing'
                and _soft_lower_sec <= _final_measured_sec <= _soft_upper_sec):
            qa['story_duration_gate'] = 'pass_with_warning'
            qa['needs_review'] = True
            qa['warnings'].append(
                f"Final narration is within the soft target band: {_final_measured_sec:.1f}s "
                f"vs {target_duration_sec:.1f}s target"
            )

        if qa.get('story_duration_gate') not in ('pass', 'pass_with_warning'):
            qa['needs_review'] = True
            qa['story_duration_gate'] = 'fail'
            qa['errors'].append(f"Narration target not met: {qa.get('narration_duration_sec', 0):.1f}s vs {target_duration_sec:.1f}s")
            raise Exception(
                f"Narration target not met after repair: "
                f"{qa.get('narration_duration_sec', 0)/60:.1f}m vs {target_minutes:.1f}m target"
            )

        # Persist the final duration-repaired narration, not the rejected
        # short first pass that may have triggered the repair loop.
        with open(text_result, 'w', encoding='utf-8') as f:
            for _sc in scenes_data:
                f.write(f"[{_sc.get('start_time')} - {_sc.get('end_time')}] {_sc.get('script')}\n")

        valid_scenes, seg_paths, word_lists = zip(*valid_data)
        valid_scenes = list(valid_scenes); seg_paths = list(seg_paths); word_lists = list(word_lists)

        # Root fix for A/V drift: video clips are always encoded at 30fps,
        # so ffmpeg silently snaps any "-t <duration>" request to the
        # nearest frame boundary (1/30s) when building each clip - but the
        # audio track was being concatenated from the RAW (un-quantized)
        # TTS durations. Those two clocks differ by a fraction of a frame
        # per scene, and that difference compounds scene after scene until
        # it's visibly out of sync a few minutes in - regardless of which
        # tools/settings (freeze, speed rate, etc.) are in use, since both
        # paths always hit this same frame-rounding step. Quantizing here,
        # once, and then forcing BOTH video and audio to this exact value
        # makes drift architecturally impossible instead of merely reduced.
        FRAME_RATE = 30
        def _quantize_to_frame(d):
            return round(d * FRAME_RATE) / FRAME_RATE

        raw_durs = [probe_dur(p) for p in seg_paths]

        # Footage-aware audio tempo nudge: if a scene's real footage is
        # SHORTER than its TTS narration, speeding the AUDIO up by a
        # modest, barely-perceptible amount (capped at 1.25x - beyond
        # that starts sounding unnatural) lets the video play at its own
        # normal 1x speed with NO stretch/freeze/loop needed at all. This
        # is more precise (zero visual side-effects, since real footage
        # is never slowed/repeated) and faster (one cheap audio pass per
        # affected scene vs. the multi-step video planning those
        # fallbacks require) than adapting on the video side. Scenes
        # whose shortfall is too large for a natural-sounding speed-up
        # still fall through to the existing video-side handling below,
        # now a rare case instead of the default path.
        def _footage_avail_sec(idx):
            try:
                s = valid_scenes[idx]
                return max(0.0, parse_time_to_sec(s.get('end_time')) - parse_time_to_sec(s.get('start_time')))
            except Exception:
                return None

        _tempo_nudged_count = [0]

        def _tempo_job(idx):
            v_avail = _footage_avail_sec(idx)
            if v_avail is None or v_avail <= 0.3:
                return
            d = raw_durs[idx]
            if d <= v_avail * 1.05:
                return  # already fits (within 5%) - nothing to do
            factor = d / v_avail
            if factor > max_audio_stretch:
                print(f"[tempo-nudge] scene {idx}: TTS={d:.2f}s footage={v_avail:.2f}s factor={factor:.2f} > {max_audio_stretch:.2f} -> too large, leaving for video-side fallback")
                return  # too much speed-up would sound unnatural - leave for video-side fallback
            sped_path = os.path.join(temp_dir, f"tempo_{idx:05d}.mp3")
            try:
                run_cmd(["ffmpeg", "-y", "-i", seg_paths[idx], "-af", f"atempo={factor:.4f}",
                          "-c:a", "libmp3lame", "-b:a", "128k", sped_path],
                        f"Tempo-nudge Audio {idx} (x{factor:.2f})", video_id)
                seg_paths[idx] = sped_path
                raw_durs[idx] = v_avail
                _tempo_nudged_count[0] += 1
                print(f"[tempo-nudge] scene {idx}: TTS={d:.2f}s footage={v_avail:.2f}s -> sped up x{factor:.3f} OK")
            except Exception as e:
                print(f"[tempo-nudge] scene {idx} failed, leaving as-is: {e}")

        import concurrent.futures as _cf
        with _cf.ThreadPoolExecutor(max_workers=4) as _pool:
            list(_pool.map(_tempo_job, range(len(seg_paths))))
        if _tempo_nudged_count[0] > 0:
            log_status(video_id, f"{part_tag}🎚️ Scene {_tempo_nudged_count[0]} ခုကို footage အတိုင်း audio tempo ညှိပြီးပါပြီ (video ကို ဘာမှ မထိခိုက်ပါ)။")

        seg_durs = [_quantize_to_frame(d) for d in raw_durs]
        a_dur    = sum(seg_durs)
        qa['audio_segments'] = len(seg_durs)
        qa['max_audio_stretch'] = max_audio_stretch
        mapping = []
        audio_cursor = 0.0
        for idx, (scene, seg_dur) in enumerate(zip(valid_scenes, seg_durs)):
            source_start = parse_time_to_sec(scene.get('start_time'))
            source_end = parse_time_to_sec(scene.get('end_time'))
            mapping.append({
                "index": idx,
                "audio_start": round(audio_cursor, 3),
                "audio_end": round(audio_cursor + seg_dur, 3),
                "source_start": round(source_start, 3),
                "source_end": round(source_end, 3),
                "source_duration": round(max(0.0, source_end - source_start), 3),
                "audio_duration": round(seg_dur, 3),
            })
            audio_cursor += seg_dur
        qa['audio_timeline_duration_sec'] = round(audio_cursor, 3)
        qa['mapping_preview'] = (mapping[:3] + ([{"ellipsis": True}] if len(mapping) > 6 else []) + mapping[-3:])
        qa['mapping_gate'] = 'pass' if abs(audio_cursor - a_dur) <= 0.001 else 'fail'
        if qa['mapping_gate'] != 'pass':
            qa['errors'].append('Audio timeline mapping does not sum to the rendered audio duration')
            raise Exception('Audio timeline mapping validation failed')

        # Build per-scene audio segments trimmed/padded to the EXACT same
        # quantized duration used for the matching video segment, then
        # concatenate those (not the raw files) - so the audio track's
        # cumulative timeline always lands on the identical clock as the
        # video's cumulative timeline, scene by scene, with zero drift.
        log_status(video_id, f"{part_tag}အသံစဉ်ကို frame-clock နှင့် ကိုက်ညီအောင် ချိန်ညှိနေပါသည်...")
        quantized_seg_paths = [None] * len(seg_paths)
        def _quantize_audio_job(idx):
            p, qdur = seg_paths[idx], seg_durs[idx]
            qpath = os.path.join(temp_dir, f"qaudio_{idx:05d}.mp3")
            run_cmd(["ffmpeg", "-y", "-i", p, "-af", f"apad,atrim=0:{qdur}",
                      "-t", str(qdur), "-c:a", "libmp3lame", "-b:a", "128k", qpath],
                    f"Quantize Audio {idx}", video_id)
            quantized_seg_paths[idx] = qpath

        import concurrent.futures as _cf
        with _cf.ThreadPoolExecutor(max_workers=4) as _pool:
            _futs = {_pool.submit(_quantize_audio_job, i): i for i in range(len(seg_paths))}
            _q_errors = []
            for _fut in _cf.as_completed(_futs):
                try:
                    _fut.result()
                except Exception as _e:
                    _q_errors.append(str(_e))
        if _q_errors:
            raise Exception(f"FFmpeg Error during audio quantization: {_q_errors[0]}")

        # Verify: ffmpeg can silently produce a slightly-off duration even
        # on this quantize pass. Check each segment and repair by looping
        # if it's off, same safety net already used for video clips.
        for idx, qpath in enumerate(quantized_seg_paths):
            qdur = seg_durs[idx]
            if qdur <= 0.02 or not os.path.exists(qpath):
                continue
            try:
                _actual = float(subprocess.check_output(
                    ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                     "-of", "default=noprint_wrappers=1:nokey=1", qpath]).decode().strip())
            except Exception:
                _actual = 0.0
            if abs(qdur - _actual) > 0.01:
                _fixed = qpath + ".fixed.mp3"
                try:
                    run_cmd(["ffmpeg", "-y", "-stream_loop", "-1", "-i", qpath,
                              "-t", str(qdur), "-c:a", "libmp3lame", "-b:a", "128k", _fixed],
                            "Audio Duration Repair", video_id)
                    if os.path.exists(_fixed) and os.path.getsize(_fixed) > 200:
                        os.replace(_fixed, qpath)
                except Exception as _repair_err:
                    print(f"[AUDIO REPAIR] Failed for {qpath}: {_repair_err}")

        concat_a = os.path.join(temp_dir, 'audio_concat.txt')
        with open(concat_a,'w') as f:
            for p in quantized_seg_paths: f.write(f"file '{p}'\n")
        run_cmd(["ffmpeg","-y","-f","concat","-safe","0","-i",concat_a,"-c:a","libmp3lame","-b:a","128k", audio_path], "Concat Audio", video_id)

        log_status(video_id, f"{part_tag}စာတန်းထိုး တွက်ချက်နေပါသည်...")
        MAX_SUB     = 28
        srt_lines   = []
        cursor      = 0.0
        sub_idx     = 1
        scene_breaks = []  # per-scene list of LOCAL (0..seg_dur) subtitle row-end times, used to align freeze frames to natural pauses

        def _wrap_overlong(token, limit):
            # Burmese (and some other scripts) are often written with very
            # few spaces, so a single "word" from .split() can be an entire
            # long phrase. Without this, that phrase would be emitted as one
            # oversized subtitle line that overflows the frame. Break it
            # into fixed-width chunks as a fallback.
            return [token[i:i+limit] for i in range(0, len(token), limit)] if len(token) > limit else [token]

        def _rows_from_words(words, limit):
            # Groups edge-tts's real per-word timestamps into subtitle rows.
            # This gives EXACT audio<->subtitle sync (not an estimate),
            # because each row's start/end comes directly from the TTS
            # engine's own word boundaries.
            rows = []
            cur_text, cur_start, cur_end = "", None, None
            for w in words:
                wtxt = w["text"]
                cand = (cur_text + " " + wtxt).strip() if cur_text else wtxt
                if len(cand) <= limit or not cur_text:
                    if cur_start is None: cur_start = w["start"]
                    cur_text, cur_end = cand, w["end"]
                else:
                    rows.append((cur_text, cur_start, cur_end))
                    cur_text, cur_start, cur_end = wtxt, w["start"], w["end"]
            if cur_text:
                rows.append((cur_text, cur_start, cur_end))
            return rows

        for sc, dur, words in zip(valid_scenes, seg_durs, word_lists):
            sent   = _spoken_script_text(sc.get('script', ''))
            breaks = []

            if words:
                # EXACT PATH: real word-level timing from edge-tts
                rows = _rows_from_words(words, MAX_SUB)
                if rows:
                    for row_text, st_local, et_local in rows:
                        st_local = max(0.0, min(st_local, dur))
                        et_local = max(st_local + 0.1, min(et_local, dur))
                        st_g, et_g = cursor + st_local, cursor + et_local
                        srt_lines.append(f"{sub_idx}\n{format_timestamp(st_g)} --> {format_timestamp(et_g)}\n{row_text}\n")
                        sub_idx += 1
                        breaks.append(et_local)
                else:
                    words = None  # fall through to estimate path below

            if not words:
                # ESTIMATE PATH: no word-level timing available (Gemini/Edge-TTS
                # voices don't expose it) - distribute proportionally by
                # character count, as before.
                sub_words = sent.split()
                sub_rows  = []
                line      = ""
                for w in sub_words:
                    t = (line + " " + w).strip()
                    if len(t) <= MAX_SUB:
                        line = t
                    else:
                        if line: sub_rows.append(line)
                        if len(w) > MAX_SUB:
                            sub_rows.extend(_wrap_overlong(w, MAX_SUB))
                            line = ""
                        else:
                            line = w
                if line: sub_rows.append(line)
                if not sub_rows:
                    sub_rows = _wrap_overlong(sent, MAX_SUB) if sent else [sent]

                weights = [max(len(r),1) for r in sub_rows]
                total_w = sum(weights)
                lc_local = 0.0
                for row, w in zip(sub_rows, weights):
                    ld = (w / total_w) * dur
                    st_local = lc_local
                    et_local = min(lc_local + ld - 0.04, dur)
                    st_g, et_g = cursor + st_local, cursor + max(et_local, st_local + 0.1)
                    srt_lines.append(f"{sub_idx}\n{format_timestamp(st_g)} --> {format_timestamp(et_g)}\n{row}\n")
                    sub_idx += 1; lc_local += ld
                    breaks.append(et_local)

            scene_breaks.append(breaks)
            cursor += dur

        sub_language = (settings.get('sub_language') or '').strip()
        log_status(video_id, f"{part_tag}စာတန်းဘာသာ: sub_language='{sub_language}', srt_lines={len(srt_lines)}")
        if sub_language and sub_language != 'Voice Language' and srt_lines:
            try:
                log_status(video_id, f"🌐 စာတန်းများကို {sub_language} ဘာသာသို့ AI ဖြင့် ပြန်ဆိုနေပါသည်...")
                srt_lines = translate_srt_blocks(srt_lines, sub_language, video_id=video_id)
                log_status(video_id, f"🌐 စာတန်းဘာသာပြန်ပြီးစီးပါပြီ ({sub_language})")
            except Exception as tr_err:
                log_status(video_id, f"⚠️ စာတန်းဘာသာပြန်မှု မအောင်မြင် — မူရင်းဘာသာဖြင့် ဆက်လက်အသုံးပြုမည်: {tr_err}")
        elif sub_language and srt_lines:
            log_status(video_id, f"🌐 စာတန်းဘာသာပြန်ရန်: '{sub_language}'")

        with open(srt_path,'w',encoding='utf-8') as f: f.write('\n'.join(srt_lines))

        log_status(video_id, f"{part_tag}ဇာတ်ဝင်ခန်းများကို အသံနှင့် ကွက်တိဖြစ်အောင် ပေါင်းစပ်ချိန်ညှိနေပါသည် (100% Semantic Sync)...")
        v_dur = float(subprocess.check_output(["ffprobe","-v","error","-show_entries","format=duration","-of","default=noprint_wrappers=1:nokey=1", local_path]).decode().strip())

        res            = settings.get('resolution','1280:720')
        try:
            rw, rh = int(res.split(':')[0]), int(res.split(':')[1])
        except Exception:
            rw, rh = 1280, 720
        ratio = str(settings.get('aspect_ratio', '16:9')).strip()
        def _even(value):
            return max(2, int(round(float(value) / 2.0) * 2))
        if ratio == '9:16':
            out_h = _even(max(rw, rh))
            out_w = _even(out_h * 9.0 / 16.0)
        elif ratio == '1:1':
            side = _even(min(rw, rh))
            out_w, out_h = side, side
        elif ratio == '4:3':
            out_h = _even(rh)
            out_w = _even(out_h * 4.0 / 3.0)
        else:
            out_w = _even(rw)
            out_h = _even(out_w * 9.0 / 16.0)
        res = f"{out_w}:{out_h}"
        out_w, out_h = int(res.split(':')[0]), int(res.split(':')[1])
        res_x          = res.replace(':','x')
        log_status(video_id, f"{part_tag}Output: {out_w}x{out_h} (ratio={ratio}, base_res={settings.get('resolution','1280:720')})")
        # The checkbox is authoritative. When it is off, this renderer never
        # extracts a still frame or uses zoompan; short narration/source
        # mismatches are filled with the same scene's moving footage instead.
        freeze_en      = str(settings.get('freeze_enabled', '')).strip().lower() in ('true', '1', 'on', 'yes')
        sync_mode      = str(settings.get('sync_mode', 'strict')).strip().lower()
        fallback_policy = str(settings.get('fallback_policy', 'freeze')).strip().lower()
        if not freeze_en:
            log_status(video_id, f"{part_tag}Freeze Frame ပိတ်ထားပါသည် — shortfall ဖြစ်လျှင် အဲဒီ scene ရဲ့ moving footage ကိုပဲ loop/fit လုပ်ပါမည်။")
        try:
            max_audio_stretch = max(1.0, min(1.25, float(settings.get('max_audio_stretch', 1.12))))
        except (TypeError, ValueError):
            max_audio_stretch = 1.12
        req_play       = max(0.5, float(settings.get('freeze_interval', 5.0)))
        req_freeze     = float(settings.get('freeze_duration', 3.0))
        z_pow          = float(settings.get('zoom_power', 1.5))

        PREVIEW_DIR = os.path.join(BASE_DIR, "data", "previews")
        os.makedirs(PREVIEW_DIR, exist_ok=True)
        preview_state = {"scene": 0, "total": 0, "step": "Starting...", "frame": ""}
        update_preview(video_id, preview_state)

        scale_f        = (f"fps=30,scale={res}:force_original_aspect_ratio=decrease,pad={res}:(ow-iw)/2:(oh-ih)/2,format=yuv420p")
        clip_list      = []
        clip_expected  = {}  # path -> expected duration, for post-encode verification
        # Segment encodes (Move/Freeze) are independent of each other — each
        # reads/writes its own file — so instead of running them one-by-one
        # (blocking on ffmpeg startup+encode for every single scene segment),
        # we collect them here and fire them off concurrently afterward.
        # Frame extraction stays synchronous since it's a single fast read
        # and some encode commands depend on its output file existing first.
        encode_jobs    = []

        def _build_playback_segments(seg_dur, breaks, play_interval, freeze_dur):
            """
            Returns a list of ('play', length) / ('freeze', length) segments
            summing to seg_dur. Freeze points are snapped to the nearest
            natural pause (a subtitle row boundary, i.e. end of a
            sentence/clause) at or after the fixed interval target, instead
            of a raw timer tick that can land mid-word. interval/duration/
            zoom settings are still respected - this only changes *where*
            inside the scene the freeze lands.
            """
            segments = []
            usable_breaks = sorted(b for b in breaks if 0.05 < b < seg_dur - 0.05)
            t, bi = 0.0, 0
            while t < seg_dur - 0.05:
                target = t + play_interval
                while bi < len(usable_breaks) and usable_breaks[bi] < target:
                    bi += 1
                if bi >= len(usable_breaks) or usable_breaks[bi] >= seg_dur - 0.05:
                    segments.append(('play', seg_dur - t))
                    break
                snap = usable_breaks[bi]
                bi += 1
                play_len = snap - t
                if play_len > 0.02:
                    segments.append(('play', play_len))
                f_len = min(freeze_dur, seg_dur - snap)
                if f_len > 0.02:
                    segments.append(('freeze', f_len))
                t = snap + f_len
            return segments

        def _loop_fill_segments(v_start, v_end, need_dur):
            """
            When a scene's source footage (v_start..v_end) is shorter than
            the narration audio for that scene, and freeze frames are
            disabled, we must NOT drop the shortfall — that's what caused
            video to fall behind audio more and more as the recap went on
            (and could shrink total output far below the intended length).
            Instead, loop the scene's OWN footage from the start again to
            fill the remaining time. Real motion keeps playing (never a
            freeze), and total video time always equals the audio time —
            sync stays exact no matter how long the recap is.
            """
            window = v_end - v_start
            if window <= 0.1:
                return []
            segs, pos, remaining, guard = [], v_start, need_dur, 0
            while remaining > 0.02 and guard < 500:
                guard += 1
                avail = v_end - pos
                take = min(remaining, avail)
                if take > 0.02:
                    segs.append((pos, take))
                    remaining -= take
                    pos += take
                if pos >= v_end - 0.02:
                    pos = v_start
            return segs

        # Frame-exact accumulator: converts each sub-segment's floating
        # length (seconds) into an exact frame count via CUMULATIVE
        # rounding within the scene, rather than rounding each piece's
        # length independently. `-t <seconds>` on an ffmpeg encode can
        # land a fraction of a frame short/long, and those independent
        # roundings compound scene after scene into visible A/V drift
        # later in long recaps. Rounding the running cumulative time
        # instead (and taking the difference from what's already been
        # allocated) guarantees every scene's clips sum to EXACTLY
        # round(seg_dur * FRAME_RATE) frames - the same frame count the
        # matching audio segment was quantized to - by construction, so
        # there is nothing left to drift regardless of video length.
        def _telescope_frames(cum_t, cum_f, length):
            new_cum_t = cum_t + length
            target_f  = round(new_cum_t * FRAME_RATE)
            frames    = max(0, target_f - cum_f)
            return frames, new_cum_t, target_f

        for i, (scene, seg_dur) in enumerate(zip(valid_scenes, seg_durs)):
            _cum_t, _cum_f = 0.0, 0
            v_start = parse_time_to_sec(scene.get('start_time', '00:00:00'))
            v_end = parse_time_to_sec(scene.get('end_time', '00:00:05'))
            if v_end <= v_start: v_end = min(v_start + 5.0, v_dur)
            v_start = min(v_start, v_dur)
            v_end = min(v_end, v_dur)

            v_cursor = v_start
            plan = _build_playback_segments(seg_dur, scene_breaks[i], req_play, req_freeze) if freeze_en else [('play', seg_dur)]
            if not plan:
                plan = [('play', seg_dur)]

            preview_state["scene"] = i + 1
            preview_state["total"] = len(valid_scenes)
            preview_state["step"] = f"Scene {i+1}/{len(valid_scenes)}"

            for kind, length in plan:
                if length <= 0.02:
                    continue
                v_avail = v_end - v_cursor
                rem_dur = length

                # Exhaustion fallback: source footage for this scene ran out
                # before the requested segment length was filled.
                # If freeze is DISABLED, loop the scene's own footage to fill
                # the remaining time (never drop it — that broke A/V sync).
                if v_avail <= 0.05:
                    if not freeze_en:
                        print(f"[shortfall] scene {i}: footage fully exhausted (v_avail={v_avail:.2f}s), need {rem_dur:.2f}s more -> LOOP fill (enable Natural Freeze to use a freeze-frame here instead)")
                        for start_ts, ln in _loop_fill_segments(v_start, v_end, rem_dur):
                            frames, _cum_t, _cum_f = _telescope_frames(_cum_t, _cum_f, ln)
                            if frames <= 0:
                                continue
                            out_m = os.path.join(temp_dir, f"m_{i}_{len(clip_list)}.mp4")
                            encode_jobs.append((["ffmpeg","-y","-ss",str(start_ts),"-i",local_path,"-frames:v",str(frames),"-vf",scale_f,"-c:v","libx264","-preset","veryfast","-an",out_m], f"Move {i} (loop)"))
                            clip_list.append(out_m)
                            clip_expected[out_m] = frames / FRAME_RATE
                        continue
                    img = os.path.join(temp_dir, f"i_{i}_{len(clip_list)}.jpg")
                    fpt = max(0, v_end - 0.1)
                    run_cmd(["ffmpeg","-y","-ss",str(fpt),"-i",local_path,"-vframes","1","-q:v","2",img], f"Extract {i}", video_id)
                    frames, _cum_t, _cum_f = _telescope_frames(_cum_t, _cum_f, rem_dur)
                    frm = max(1, frames)
                    zexp = f"min(zoom+({z_pow}-1)/{frm},{z_pow})"
                    vf_freeze = f"scale={res}:force_original_aspect_ratio=decrease,pad={res}:(ow-iw)/2:(oh-ih)/2,zoompan=z='{zexp}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frm}:s={res_x}:fps=30,format=yuv420p"
                    out_z = os.path.join(temp_dir, f"z_{i}_{len(clip_list)}.mp4")
                    encode_jobs.append((["ffmpeg","-y","-loop","1","-i",img,"-frames:v",str(frm),"-vf", vf_freeze,"-c:v","libx264","-preset","veryfast","-an",out_z], f"Freeze {i}"))
                    clip_list.append(out_z)
                    clip_expected[out_z] = frm / FRAME_RATE
                    continue

                if kind == 'play':
                    # Mild footage shortfall: a gentle slow-motion stretch
                    # to fill the exact duration looks far better than an
                    # obvious repeat-loop or a static freeze - reserve
                    # those for shortfalls too large to stretch naturally.
                    # This matters far more for Local-Fast (short raw
                    # camera cuts exhaust quickly) than Gemini-Video
                    # (usually longer atomic scenes), but the same mild-
                    # shortfall logic helps either engine.
                    # NOTE: -ss/-t must stay BEFORE -i to limit INPUT
                    # reading; placed after -i they limit OUTPUT duration
                    # instead, which would silently cancel the stretch.
                    if (not freeze_en) and v_avail > 0.05 and v_avail < rem_dur and (rem_dur / v_avail) <= 1.8:
                        frames, _cum_t, _cum_f = _telescope_frames(_cum_t, _cum_f, rem_dur)
                        stretch = rem_dur / v_avail
                        print(f"[shortfall] scene {i}: need {rem_dur:.2f}s, footage {v_avail:.2f}s -> slow-stretch x{stretch:.2f} (video, audio tempo-nudge already tried upstream)")
                        out_m = os.path.join(temp_dir, f"m_{i}_{len(clip_list)}.mp4")
                        encode_jobs.append((["ffmpeg","-y","-ss",str(v_cursor),"-t",str(v_avail),"-i",local_path,
                                              "-vf",f"{scale_f},setpts={stretch:.6f}*PTS",
                                              "-r","30","-frames:v",str(frames),
                                              "-c:v","libx264","-preset","veryfast","-an",out_m],
                                             f"Move {i} (slow-stretch x{stretch:.2f})"))
                        clip_list.append(out_m)
                        clip_expected[out_m] = frames / FRAME_RATE
                        v_cursor = v_end
                        continue

                    p_len = min(rem_dur, v_avail)
                    if p_len >= 0.02:
                        frames, _cum_t, _cum_f = _telescope_frames(_cum_t, _cum_f, p_len)
                        out_m = os.path.join(temp_dir, f"m_{i}_{len(clip_list)}.mp4")
                        encode_jobs.append((["ffmpeg","-y","-ss",str(v_cursor),"-i",local_path,"-frames:v",str(frames),"-vf",scale_f,"-c:v","libx264","-preset","veryfast","-an",out_m], f"Move {i}"))
                        clip_list.append(out_m)
                        clip_expected[out_m] = frames / FRAME_RATE
                        v_cursor += p_len
                    leftover = rem_dur - p_len
                    if leftover >= 0.02:
                        # ran out of footage mid-segment
                        if freeze_en:
                            # Freeze the rest with zoom
                            img = os.path.join(temp_dir, f"i_{i}_{len(clip_list)}.jpg")
                            fpt = max(0, v_cursor - 0.1)
                            run_cmd(["ffmpeg","-y","-ss",str(fpt),"-i",local_path,"-vframes","1","-q:v","2",img], f"Extract {i}", video_id)
                            frames, _cum_t, _cum_f = _telescope_frames(_cum_t, _cum_f, leftover)
                            vf_freeze = f"scale={res}:force_original_aspect_ratio=decrease,pad={res}:(ow-iw)/2:(oh-ih)/2,format=yuv420p"
                            out_z = os.path.join(temp_dir, f"z_{i}_{len(clip_list)}.mp4")
                            encode_jobs.append((["ffmpeg","-y","-loop","1","-i",img,"-frames:v",str(frames),"-vf", vf_freeze,"-c:v","libx264","-preset","veryfast","-an",out_z], f"Freeze {i}"))
                            clip_list.append(out_z)
                            clip_expected[out_z] = frames / FRAME_RATE
                        else:
                            # Freeze disabled — loop the scene's own footage
                            # instead of dropping this time (keeps A/V sync exact)
                            print(f"[shortfall] scene {i}: footage ran out mid-scene, {leftover:.2f}s still needed -> LOOP fill (enable Natural Freeze to use a freeze-frame here instead)")
                            for start_ts, ln in _loop_fill_segments(v_start, v_end, leftover):
                                frames, _cum_t, _cum_f = _telescope_frames(_cum_t, _cum_f, ln)
                                if frames <= 0:
                                    continue
                                out_m = os.path.join(temp_dir, f"m_{i}_{len(clip_list)}.mp4")
                                encode_jobs.append((["ffmpeg","-y","-ss",str(start_ts),"-i",local_path,"-frames:v",str(frames),"-vf",scale_f,"-c:v","libx264","-preset","veryfast","-an",out_m], f"Move {i} (loop)"))
                                clip_list.append(out_m)
                                clip_expected[out_m] = frames / FRAME_RATE

                else:  # freeze
                    f_len = rem_dur
                    img = os.path.join(temp_dir, f"i_{i}_{len(clip_list)}.jpg")
                    fpt = max(0, v_cursor - 0.1)
                    run_cmd(["ffmpeg","-y","-ss",str(fpt),"-i",local_path,"-vframes","1","-q:v","2",img], f"Extract {i}", video_id)
                    frames, _cum_t, _cum_f = _telescope_frames(_cum_t, _cum_f, f_len)
                    frm = max(1, frames)
                    zexp = f"min(zoom+({z_pow}-1)/{frm},{z_pow})"
                    vf_freeze = f"scale={res}:force_original_aspect_ratio=decrease,pad={res}:(ow-iw)/2:(oh-ih)/2,zoompan=z='{zexp}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frm}:s={res_x}:fps=30,format=yuv420p"
                    out_z = os.path.join(temp_dir, f"z_{i}_{len(clip_list)}.mp4")
                    encode_jobs.append((["ffmpeg","-y","-loop","1","-i",img,"-frames:v",str(frm),"-vf", vf_freeze,"-c:v","libx264","-preset","veryfast","-an",out_z], f"Freeze {i}"))
                    clip_list.append(out_z)
                    clip_expected[out_z] = frm / FRAME_RATE

        # Run all queued Move/Freeze segment encodes concurrently instead of
        # one-by-one. Each job writes to its own unique file, so this is
        # safe — clip_list order (built above) is untouched either way.
        if encode_jobs:
            log_status(video_id, f"{part_tag}ဇာတ်ဝင်ခန်း {len(encode_jobs)} ခု ကို တစ်ပြိုင်နက် Encode ပြုလုပ်နေပါသည်...")
            import concurrent.futures
            errors = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as _pool:
                futures = {_pool.submit(run_cmd, cmd, label, video_id): label for cmd, label in encode_jobs}
                for fut in concurrent.futures.as_completed(futures):
                    try:
                        fut.result()
                    except Exception as _e:
                        errors.append(str(_e))
            if errors:
                raise Exception(f"FFmpeg Error during segment encode: {errors[0]}")

            # Verify every clip actually has the duration we asked for.
            # Segments above are now requested with -frames:v (an exact
            # frame count from the telescoping accumulator), so this is a
            # SAFETY NET rather than the primary sync mechanism: ffmpeg can
            # still exit 0 while producing a SHORTER file than requested if
            # the seek point runs out of real frames before that many
            # frames could be encoded - no error is ever raised for this.
            # That silent shortfall is exactly what causes "video freezes /
            # falls behind while audio keeps playing" with no visible
            # error anywhere. Repair any short clip by looping it to fill
            # its own gap. The threshold is tightened to about one frame
            # (was 0.15s / ~4.5 frames) since real shortfalls should now be
            # rare and small drifts should no longer be tolerated silently.
            _repair_threshold = 1.0 / FRAME_RATE
            for _path, _expected in clip_expected.items():
                if _expected <= 0.02 or not os.path.exists(_path):
                    continue
                try:
                    _actual = float(subprocess.check_output(
                        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                         "-of", "default=noprint_wrappers=1:nokey=1", _path]).decode().strip())
                except Exception:
                    _actual = 0.0
                if _expected - _actual > _repair_threshold:
                    log_status(video_id, f"{part_tag}⚠️ Clip တစ်ခု တိုနေလို့ ({_actual:.1f}s/{_expected:.1f}s) ပြင်ဆင်နေပါသည်...")
                    _padded = _path + ".padded.mp4"
                    try:
                        run_cmd(["ffmpeg", "-y", "-stream_loop", "-1", "-i", _path,
                                  "-t", str(_expected), "-c", "copy", _padded],
                                "Duration Repair", video_id)
                        if os.path.exists(_padded) and os.path.getsize(_padded) > 500:
                            os.replace(_padded, _path)
                    except Exception as _repair_err:
                        print(f"[DURATION REPAIR] Failed for {_path}: {_repair_err}")

        log_status(video_id, f"{part_tag}ဗီဒီယိုများကို ဆက်စပ်နေပါသည် (Fast Merge)...")
        preview_state["step"] = "Merging video..."
        update_preview(video_id, preview_state)
        concat_v = os.path.join(temp_dir,'concat.txt')
        synced   = os.path.join(temp_dir,'synced.mp4')
        if not clip_list:
            raise Exception("No video clips generated — all scenes may have failed.")
        with open(concat_v,'w') as f:
            for c in clip_list: f.write(f"file '{c}'\n")

        run_cmd(["ffmpeg","-y","-f","concat","-safe","0","-i",concat_v,"-c:v","libx264","-preset","veryfast","-r","30","-vsync","cfr","-t",str(a_dur),"-an", synced], "Concat+Sync (CFR 30fps)", video_id)

        # ── STEP 5: EFFECTS + SUBTITLES + FINAL RENDER ──────────────
        log_status(video_id, f"{part_tag}Visual Effects နှင့် စာတန်းထိုးများ ထည့်သွင်းနေပါသည်...")
        filters = []
        last_v  = "0:v"

        if settings.get('bypass_mirror') == 'true':
            filters.append(f"[{last_v}]hflip[{last_v}_m]"); last_v = f"{last_v}_m"
        if settings.get('bypass_color') == 'true':
            filters.append(f"[{last_v}]eq=brightness=0.02:saturation=1.05:contrast=1.02[{last_v}_c]"); last_v = f"{last_v}_c"
        if settings.get('bypass_rotation') == 'true':
            filters.append(f"[{last_v}]rotate=1*PI/180:c=black:ow=iw:oh=ih,scale={out_w}:{out_h}[{last_v}_r]"); last_v = f"{last_v}_r"
        if settings.get('bypass_noise') == 'true':
            filters.append(f"[{last_v}]noise=alls=1:allf=t+u[{last_v}_n]"); last_v = f"{last_v}_n"

        # Aspect ratio is baked into `res` above, so clips already come out
        # at the true final size (e.g. 720x1280 for 9:16) - no re-crop needed.
        if settings.get('blur_enabled') == 'true':
            regions = []
            try:
                for r in json.loads(settings.get('blur_regions') or '[]'):
                    regions.append((float(r.get('y', 80)), float(r.get('h', 20)), float(r.get('strength', 25))))
            except Exception:
                pass
            if not regions:
                regions = [(float(settings.get('blur_y', 80)), float(settings.get('blur_h', 20)), float(settings.get('blur_strength', 25)))]
            for bi, (y, h, s) in enumerate(regions):
                filters.append(f"[{last_v}]split=2[base{bi}][bl{bi}]")
                filters.append(f"[bl{bi}]crop=iw:ih*{h}/100:0:ih*{y}/100,gblur=sigma={s}[blurred{bi}]")
                filters.append(f"[base{bi}][blurred{bi}]overlay=x=0:y=H*{y}/100[{last_v}_blur{bi}]")
                last_v = f"{last_v}_blur{bi}"

        inputs          = ["-i", synced]
        cur_input_idx   = 1
        ffmpeg_env      = None

        logo_text = settings.get('logo_text','').strip()
        if logo_path and os.path.exists(logo_path):
            inputs.extend(["-i", logo_path])
            filters.append(f"[{cur_input_idx}:v]scale=iw*0.15:-1,format=rgba,colorchannelmixer=aa=0.85[logo_s]")
            pos = settings.get('logo_pos','TOP LEFT')
            lx = {'TOP RIGHT':'W-w-15','BOTTOM LEFT':'15','BOTTOM RIGHT':'W-w-15'}.get(pos,'15')
            ly = {'TOP RIGHT':'15','BOTTOM LEFT':'H-h-15','BOTTOM RIGHT':'H-h-15'}.get(pos,'15')
            filters.append(f"[{last_v}][logo_s]overlay=x={lx}:y={ly}[{last_v}_lg]")
            last_v = f"{last_v}_lg"; cur_input_idx += 1

        if logo_text:
            wm_f   = DEFAULT_FONT_PATH if os.path.exists(DEFAULT_FONT_PATH) else ''
            pos    = settings.get('logo_pos','TOP LEFT')
            tx = {'TOP RIGHT':'w-tw-15','BOTTOM LEFT':'15','BOTTOM RIGHT':'w-tw-15'}.get(pos,'15')
            ty = {'TOP RIGHT':'15','BOTTOM LEFT':'h-th-15','BOTTOM RIGHT':'h-th-15'}.get(pos,'15')
            fa = f":fontfile='{wm_f}'" if wm_f else ''
            # P2-2: escape special chars for ffmpeg drawtext filter
            _safe = logo_text.replace("\\", "\\\\\\\\").replace("'", "\\\\'").replace(":", "\\\\:")
            filters.append(f"[{last_v}]drawtext=text='{_safe}'{fa}:fontsize=28:fontcolor=white@0.75:x={tx}:y={ty}:shadowcolor=black@0.6:shadowx=2:shadowy=2[{last_v}_wm]")
            last_v = f"{last_v}_wm"

        if settings.get('sub_enabled') == 'true' and os.path.exists(srt_path):
            fp = font_path if (font_path and os.path.exists(font_path)) else DEFAULT_FONT_PATH
            if not os.path.exists(fp): download_default_font()
            if os.path.exists(fp):
                c_map    = {"White":"&H00FFFFFF","Yellow":"&H0000FFFF","Green":"&H0000FF00"}
                col      = c_map.get(settings.get('sub_color'),"&H0000FFFF")
                fsz      = max(8, min(120, int(float(settings.get('sub_size','24')))))
                pos_pct  = int(settings.get('sub_position','20'))
                margin_v = max(10, int(10 + (650 * pos_pct / 100)))

                ass_path = srt_path.replace('.srt','.ass')
                srt_to_ass(srt_path, ass_path, fp, fsz, col, margin_v)

                f_dir    = os.path.dirname(os.path.abspath(fp))
                fc_cache = os.path.join(temp_dir,'fc_cache'); os.makedirs(fc_cache,exist_ok=True)
                fc_conf  = os.path.join(temp_dir,'fonts.conf')
                with open(fc_conf,'w') as _fc: _fc.write(f'<?xml version="1.0"?>\n<!DOCTYPE fontconfig SYSTEM "fonts.dtd">\n<fontconfig><dir>{f_dir}</dir><cachedir>{fc_cache}</cachedir></fontconfig>')
                ffmpeg_env = dict(os.environ)
                ffmpeg_env['FONTCONFIG_FILE'] = fc_conf

                ass_esc = ass_path.replace('\\','/')
                filters.append(f"[{last_v}]ass='{ass_esc}':fontsdir='{f_dir}'[{last_v}_sub]")
                last_v = f"{last_v}_sub"

        inputs.extend(["-i", audio_path])
        audio_idx = cur_input_idx

        cmd = ["ffmpeg","-y",*inputs]
        if filters:
            cmd.extend(["-filter_complex",";".join(filters),"-map",f"[{last_v}]"])
        else:
            cmd.extend(["-map","0:v"])
        # Calculate a sensible bitrate based on output resolution
        # Target ~2 Mbps for 720p, ~4 Mbps for 1080p, scale down for smaller
        pixel_count = out_w * out_h
        target_bitrate = str(max(800, min(6000, int(pixel_count / 300)))) + "k"
        cmd.extend([
            "-map", f"{audio_idx}:a",
            "-c:v","libx264","-preset","fast",
            "-b:v", target_bitrate,
            "-maxrate", f"{int(target_bitrate.replace('k',''))*2}k",
            "-bufsize", f"{int(target_bitrate.replace('k',''))*3}k",
            "-c:a","aac","-b:a","128k",
            "-af",f"apad,atrim=0:{a_dur}",
            "-t",str(a_dur),
            "-avoid_negative_ts","make_zero",
            "-vsync","cfr",
            final_video
        ])

        log_status(video_id, f"{part_tag}နောက်ဆုံးအဆင့် Rendering ပြုလုပ်နေပါသည်...")
        run_cmd(cmd, "Final Render", video_id, env=ffmpeg_env)

        # Audio is the master clock. Subtitle/effect filters and container
        # timestamps can introduce a small stream mismatch after the first
        # encode, so repair the final file once before declaring the job a
        # failure. This is intentionally a bounded fallback, not a silent
        # QA bypass: the repaired file is probed again below.
        try:
            _first_probe = probe_media(final_video)
            _av_ms = abs(float(_first_probe.get('video_duration', 0.0)) - float(_first_probe.get('audio_duration', 0.0))) * 1000.0
            _container_ms = abs(float(_first_probe.get('duration', 0.0)) - float(a_dur)) * 1000.0
            if _av_ms > 80.0 or _container_ms > 100.0:
                log_status(video_id, f"{part_tag}Audio master အတိုင်း final render ကို auto-repair လုပ်နေပါသည် ({_av_ms:.0f}ms)...")
                _repaired_final = final_video + '.audio_master.mp4'
                # A 0.5s pad is not enough when concat/CFR rounding leaves a
                # larger video shortfall (the reported job was 1.809s). Use
                # a generous temporary pad, then hard-trim both streams to
                # the same audio-master clock and frame count.
                _target_frames = max(1, int(math.ceil(float(a_dur) * FRAME_RATE)))
                _repair_cmd = [
                    'ffmpeg', '-y', '-i', final_video,
                    '-map', '0:v:0', '-map', '0:a:0',
                    '-vf', f'fps={FRAME_RATE},tpad=stop_mode=clone:stop_duration=3.0,trim=duration={a_dur:.6f},setpts=PTS-STARTPTS',
                    '-af', f'apad,atrim=0:{a_dur:.6f},asetpts=PTS-STARTPTS',
                    '-frames:v', str(_target_frames),
                    '-t', f'{a_dur:.6f}', '-vsync', 'cfr',
                    '-c:v', 'libx264', '-preset', 'veryfast', '-pix_fmt', 'yuv420p',
                    '-c:a', 'aac', '-b:a', '128k', '-movflags', '+faststart',
                    '-video_track_timescale', '90000',
                    _repaired_final,
                ]
                run_cmd(_repair_cmd, 'Audio-master A/V repair', video_id, env=ffmpeg_env)
                if os.path.exists(_repaired_final) and os.path.getsize(_repaired_final) > 1000:
                    os.replace(_repaired_final, final_video)
        except Exception as _repair_err:
            # Preserve the original file; the strict QA block below will
            # report the actual remaining mismatch if repair was impossible.
            print(f"[AUDIO MASTER REPAIR] skipped: {_repair_err}")
        try:
            media = probe_media(final_video)
            final_dur = media['duration']
            qa['audio_video_duration_error_ms'] = round(abs(final_dur - a_dur) * 1000, 2)
            qa['video_stream_duration_sec'] = round(media.get('video_duration', 0.0), 3)
            qa['audio_stream_duration_sec'] = round(media.get('audio_duration', 0.0), 3)
            qa['stream_duration_error_ms'] = round(abs(media.get('video_duration', 0.0) - media.get('audio_duration', 0.0)) * 1000, 2)
            qa['final_duration_sec'] = round(final_dur, 3)
            qa['expected_duration_sec'] = round(a_dur, 3)
            qa['final_media'] = media
            log_status(video_id, f"{part_tag}Final duration check: container={media.get('duration', 0.0):.3f}s, video={media.get('video_duration', 0.0):.3f}s, audio={media.get('audio_duration', 0.0):.3f}s, target={a_dur:.3f}s")
            critical = []
            if not media['has_video']: critical.append('final MP4 has no video stream')
            if not media['has_audio']: critical.append('final MP4 has no audio stream')
            if media['width'] <= 0 or media['height'] <= 0: critical.append('final MP4 has invalid dimensions')
            if final_dur <= 0.1: critical.append('final MP4 duration is empty')
            if qa['audio_video_duration_error_ms'] > 250:
                critical.append('Final audio/video duration differs by more than 250ms')
            if qa['stream_duration_error_ms'] > 100:
                critical.append('Final audio and video stream durations differ by more than 100ms')
            # A single 30fps frame (33.3ms) is too sensitive for container
            # metadata; keep normal renders under 100ms as pass-quality.
            if qa['audio_video_duration_error_ms'] > 100:
                qa['needs_review'] = True
                qa['warnings'].append('Final audio/video duration differs by more than 100ms')
            if critical:
                qa['final_media_gate'] = 'fail'
                qa['errors'].extend(critical)
                raise Exception('Final media QA failed: ' + '; '.join(critical))
            qa['final_media_gate'] = 'pass'
        except Exception as _qa_err:
            qa['needs_review'] = True
            qa['final_media_gate'] = 'fail'
            if str(_qa_err) not in qa['errors']:
                qa['errors'].append(str(_qa_err))
            qa['warnings'].append(f'QA probe failed: {_qa_err}')
            job_qa[video_id] = qa
            raise
        job_qa[video_id] = qa
        log_status(video_id, f"✅ {part_tag}အောင်မြင်ပါသည်။ (Sync QA: {qa.get('audio_video_duration_error_ms', '?')}ms)")
        result["success"]    = True
        result["video_path"] = final_video

        # Clean/video-only output mode: the .mp4 already has narration audio
        # and burned subtitles - the separate .mp3/.srt/.txt byproducts are
        # only ever needed to BUILD it, not to use it afterward. Remove them
        # so Download Files shows just the one video, and so they don't sit
        # around consuming disk space indefinitely.
        _is_auto = str(settings.get('analysis_engine', '')).strip().lower() == 'gemini_auto'
        if _is_auto:
            # Keep only the four user-facing Auto deliverables. These files
            # are internal audit/build artifacts and must not leak into the
            # download folder after a successful Auto render.
            for p in (raw_text_result, tts_manifest_path):
                try:
                    if p and os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass
        elif str(settings.get('clean_output', '')).strip().lower() in ('true', '1', 'on', 'yes'):
            for p in (audio_path, srt_path, text_result):
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass

    except Exception as e:
        qa['needs_review'] = True
        error_text = str(e)
        if error_text not in qa['errors']:
            qa['errors'].append(error_text)
        if isinstance(e, GeminiBillingError):
            qa['failure_category'] = 'gemini_billing_credits_depleted'
            qa['retryable'] = False
        result['error'] = error_text
        result['qa'] = qa
        job_qa[video_id] = qa
        log_status(video_id, f"❌ {part_tag}Error: {error_text}")
    finally:
        try: shutil.rmtree(temp_dir)
        except: pass

    return result


# ==========================================
# 5. MANUAL SCENE REVIEW + SYNC RENDERER (no Gemini)
# ==========================================
MANUAL_DIR = os.path.join(DATA_DIR, "manual")
MANUAL_THUMB_DIR = os.path.join(MANUAL_DIR, "thumbs")
os.makedirs(MANUAL_DIR, exist_ok=True)
os.makedirs(MANUAL_THUMB_DIR, exist_ok=True)
manual_jobs = {}          # job_id -> {"source","narration","manifest","stats"}


def _manual_probe(path):
    return float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", path]).decode().strip())


def _manual_has_audio(path):
    try:
        out = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", path]).decode().strip()
        return bool(out)
    except Exception:
        return False


def _manual_extract_audio(src, out_wav):
    """16 kHz mono wav for local transcription."""
    run_cmd(["ffmpeg", "-y", "-i", src, "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", out_wav],
            "Manual audio extract", "manual", None)
    return out_wav


def _manual_transcribe(path, label="manual"):
    """Local Whisper transcript -> [{"start","end","text"}]. Returns [] if unavailable."""
    try:
        wav = os.path.join(tempfile.mkdtemp(prefix="manual_asr_"), "a.wav")
        _manual_extract_audio(path, wav)
        model = _local_fast_get_whisper()
        segments, _info = model.transcribe(wav, beam_size=1)
        out = []
        for s in segments:
            text = (s.text or "").strip()
            if text:
                out.append({"start": round(float(s.start), 3), "end": round(float(s.end), 3), "text": text})
        try: shutil.rmtree(os.path.dirname(wav), ignore_errors=True)
        except Exception: pass
        return out
    except Exception as e:
        print(f"[MANUAL] transcription unavailable for {label}: {e}")
        return []


def _manual_parse_srt(text):
    """Parse SRT/VTT text into transcript segments (optional user-supplied subtitles)."""
    out = []
    for block in re.split(r"\n\s*\n", (text or "").replace("\r", "")):
        lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
        if not lines:
            continue
        ts_idx = next((i for i, ln in enumerate(lines) if "-->" in ln), None)
        if ts_idx is None:
            continue
        try:
            a_raw, b_raw = [x.strip() for x in lines[ts_idx].split("-->")[:2]]
            a = _manual_ts_to_sec(a_raw)
            b = _manual_ts_to_sec(b_raw)
        except Exception:
            continue
        txt = " ".join(lines[ts_idx + 1:]).strip()
        if txt and b > a:
            out.append({"start": round(a, 3), "end": round(b, 3), "text": txt})
    return out


def _manual_ts_to_sec(value):
    value = value.replace(",", ".")
    parts = value.split(":")
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h, m, s = 0, parts[0], parts[1]
    else:
        h, m, s = 0, 0, parts[0]
    return int(float(h)) * 3600 + int(float(m)) * 60 + float(s)


_BURMESE_STOPWORDS = set()


def _manual_tokens(text):
    """Language-agnostic token set: character trigrams + latin words.

    Burmese does not separate words with spaces, so character n-grams are a
    more reliable overlap signal than whitespace tokens; latin script keeps
    its words so proper nouns still match strongly.
    """
    t = (text or "").lower()
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return set()
    toks = set()
    for word in re.findall(r"[a-z0-9']{3,}", t):
        toks.add(word)
    compact = re.sub(r"\s+", "", t)
    for i in range(len(compact) - 2):
        gram = compact[i:i + 3]
        if gram.strip():
            toks.add(gram)
    return toks


def _manual_overlap(a, b):
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / float(min(len(a), len(b)) or 1)


def _manual_units(duration, quantum=5.0):
    return _five_second_evidence_units(duration, quantum=quantum)


def _manual_group_narration(transcript, audio_duration, max_len=8.0, min_len=1.0):
    """Group Whisper segments into review beats of roughly max_len seconds."""
    if not transcript:
        beats, cursor = [], 0.0
        while cursor < audio_duration - 0.05:
            end = min(audio_duration, cursor + 5.0)
            beats.append({"start": round(cursor, 3), "end": round(end, 3), "text": ""})
            cursor = end
        return beats
    beats, cur = [], None
    for seg in transcript:
        if cur is None:
            cur = {"start": seg["start"], "end": seg["end"], "text": seg["text"]}
            continue
        if seg["end"] - cur["start"] <= max_len and (seg["start"] - cur["end"]) <= 1.2:
            cur["end"] = seg["end"]
            cur["text"] = (cur["text"] + " " + seg["text"]).strip()
        else:
            if cur["end"] - cur["start"] < min_len and beats:
                beats[-1]["end"] = cur["end"]
                beats[-1]["text"] = (beats[-1]["text"] + " " + cur["text"]).strip()
            else:
                beats.append(cur)
            cur = {"start": seg["start"], "end": seg["end"], "text": seg["text"]}
    if cur is not None:
        beats.append(cur)
    return beats


def _manual_transcript_text(transcript, start, end):
    return " ".join(s["text"] for s in transcript if s["end"] > start and s["start"] < end)


def _manual_match_beats(beats, units, movie_transcript, source_duration, deep):
    """Score every narration beat against the 5-second source grid.

    Signals (weights are renormalised over whatever is actually available, so a
    missing movie transcript lowers confidence honestly instead of inventing a
    high score from chronology alone):
      35% transcript similarity   25% keyword overlap
      20% visual/dialogue presence 10% duration fit   10% chronological fit
    """
    unit_texts = {}
    if movie_transcript:
        for i, (u0, u1) in enumerate(units):
            unit_texts[i] = _manual_tokens(_manual_transcript_text(movie_transcript, u0, u1))
    narration_total = max(0.1, beats[-1]["end"] if beats else 0.1)

    for idx, beat in enumerate(beats):
        dur = max(0.4, beat["end"] - beat["start"])
        want = max(5.0, dur)
        n_tokens = _manual_tokens(beat.get("text", ""))
        expected = (beat["start"] / narration_total) * source_duration
        scored = []
        for i, (u0, _u1) in enumerate(units):
            end = min(source_duration, u0 + want)
            if end - u0 < min(4.0, want):
                continue
            text_sig = _manual_overlap(n_tokens, unit_texts.get(i, set())) if unit_texts else 0.0
            dur_sig = 1.0 - min(1.0, abs((end - u0) - dur) / max(1.0, dur))
            chrono_sig = 1.0 - min(1.0, abs(u0 - expected) / max(30.0, source_duration * 0.5))
            dialogue_sig = 1.0 if unit_texts.get(i) else (0.5 if (movie_transcript and deep) else 0.0)
            scored.append({"start": round(u0, 3), "end": round(end, 3), "unit_index": i,
                           "signals": {"text": round(text_sig, 3), "keyword": round(text_sig, 3),
                                       "dialogue": round(dialogue_sig, 3), "duration": round(dur_sig, 3),
                                       "chronology": round(chrono_sig, 3)}})
        if not scored:
            beat["candidates"] = [{"start": round(min(expected, max(0.0, source_duration - 5)), 3),
                                   "end": round(min(source_duration, expected + 5), 3), "score": 0.2,
                                   "reason": "no candidate window available"}]
            beat["confidence"] = 0.2
            beat["confidence_reason"] = "source shorter than expected"
            continue

        if unit_texts:
            weights = {"text": 0.35, "keyword": 0.25, "dialogue": 0.20, "duration": 0.10, "chronology": 0.10}
        else:
            weights = {"text": 0.0, "keyword": 0.0, "dialogue": 0.0, "duration": 0.55, "chronology": 0.45}
        for cand in scored:
            total_w = sum(w for w in weights.values() if w > 0) or 1.0
            cand["score"] = round(sum(cand["signals"][k] * w for k, w in weights.items()) / total_w, 3)
            if not unit_texts:
                # Without a movie transcript there is no semantic evidence at
                # all - only duration/chronology guesses. Never report that as
                # a confident match; such scenes must stay in the review queue.
                cand["score"] = round(min(0.45, cand["score"] * 0.45), 3)
        scored.sort(key=lambda x: (-x["score"], x["unit_index"]))
        top = scored[:3]
        best = top[0]
        for c in top:
            c["reason"] = _manual_reason(c, bool(unit_texts))
        beat["candidates"] = top
        beat["confidence"] = best["score"]
        beat["confidence_reason"] = best["reason"]
    return beats


def _manual_reason(candidate, has_transcript):
    sig = candidate["signals"]
    parts = []
    if sig["text"] >= 0.30:
        parts.append("dialogue text match")
    elif sig["text"] > 0.08:
        parts.append("weak dialogue match")
    if not has_transcript:
        parts.append("no movie transcript — duration+chronology only")
    if sig["duration"] >= 0.8:
        parts.append("duration fit")
    if sig["chronology"] >= 0.8:
        parts.append("chronology fit")
    if not parts:
        parts.append("visual/dialogue signal only")
    if candidate["score"] < 0.6:
        parts.append("candidate scores are close")
    return ", ".join(parts)


def _manual_build_review(job_id, source_path, narration_path, opts):
    source_duration = _manual_probe(source_path)
    audio_duration = _manual_probe(narration_path)
    deep = bool(opts.get("deep_transcript"))
    units = _manual_units(source_duration, quantum=float(opts.get("quantum", 5.0) or 5.0))

    movie_transcript = []
    if opts.get("movie_srt"):
        movie_transcript = _manual_parse_srt(opts["movie_srt"])
    if not movie_transcript and deep and _manual_has_audio(source_path):
        log_status(job_id, "Movie dialogue ကို local Whisper ဖြင့် ဖတ်နေပါသည် (deep matching)...")
        movie_transcript = _manual_transcribe(source_path, "movie")

    log_status(job_id, "Narration timing ကို local Whisper ဖြင့် ထုတ်နေပါသည်...")
    narration_transcript = _manual_transcribe(narration_path, "narration")
    beats = _manual_group_narration(narration_transcript, audio_duration)

    log_status(job_id, f"Narration beat {len(beats)} ခုကို source နဲ့ တိုက်စစ်နေပါသည်...")
    beats = _manual_match_beats(beats, units, movie_transcript, source_duration, deep)

    segments = []
    for i, beat in enumerate(beats):
        best = beat["candidates"][0]
        conf = float(beat.get("confidence", 0.3))
        segments.append({
            "id": f"beat_{i + 1:04d}",
            "audio_start": round(beat["start"], 3),
            "audio_end": round(beat["end"], 3),
            "source_clips": [{"start": best["start"], "end": best["end"]}],
            "candidates": beat["candidates"],
            "confidence": conf,
            "confidence_reason": beat.get("confidence_reason", ""),
            "signals": best["signals"],
            "narration_text": beat.get("text", ""),
            "status": "approved" if conf >= 0.8 else "unreviewed",
            "reviewed_by_user": conf >= 0.8,
            "skip_policy": "freeze",
            "motion_enabled": True,
            "auto_filled": False,
        })

    stats = {
        "segments": len(segments),
        "high": sum(1 for s in segments if s["confidence"] >= 0.8),
        "medium": sum(1 for s in segments if 0.6 <= s["confidence"] < 0.8),
        "low": sum(1 for s in segments if s["confidence"] < 0.6),
        "movie_transcript_segments": len(movie_transcript),
        "narration_transcript_segments": len(narration_transcript),
        "deep_transcript": deep,
        "source_units": len(units),
        "estimated_output_sec": round(sum(s["audio_end"] - s["audio_start"] for s in segments), 3),
    }
    return {
        "version": "1.0",
        "mode": "manual_review",
        "sync_mode": "strict",
        # Auto analysis first tries the next adjacent source shot when a
        # selected clip is shorter than the narration beat. The user can
        # still change this per job from Advanced settings or per clip in UI.
        "policy": opts.get("policy", "next_shot"),
        "transforms": {
            "narration_only_audio": True,
            "blur_sensitive": False,
            "crop_mode": "keep",
            "color_grade": "none",
            "speed_guard": "bounded",
            "subtitles": False,
            "overlay_text": "",
            "transition": "none",
        },
        "source_duration": round(source_duration, 3),
        "audio_duration": round(audio_duration, 3),
        "stats": stats,
        "segments": segments,
    }


def validate_manual_manifest(manifest, source_duration, audio_duration):
    if not isinstance(manifest, dict):
        raise ValueError("Manifest must be a JSON object")
    segments = manifest.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("segments must be a non-empty array")
    if len(segments) > 500:
        raise ValueError("Maximum 500 segments")
    errors, ids, prev_audio_end = [], set(), -0.001
    checked = []
    for pos, seg in enumerate(segments):
        if not isinstance(seg, dict):
            errors.append(f"segment {pos + 1}: must be an object")
            continue
        sid = str(seg.get("id") or f"segment_{pos + 1}")
        if sid in ids:
            errors.append(f"{sid}: duplicate id")
        ids.add(sid)
        try:
            a0, a1 = float(seg["audio_start"]), float(seg["audio_end"])
        except Exception:
            errors.append(f"{sid}: audio_start/audio_end required numbers")
            continue
        if not (math.isfinite(a0) and math.isfinite(a1)):
            errors.append(f"{sid}: audio times must be finite")
            continue
        if a0 < 0 or a1 <= a0 or a1 > audio_duration + 0.05:
            errors.append(f"{sid}: audio range outside narration file")
        if a0 < prev_audio_end - 0.05:
            errors.append(f"{sid}: audio ranges must be chronological and non-overlapping")
        prev_audio_end = a1
        if str(seg.get("status")) == "skipped":
            checked.append({**seg, "id": sid, "audio_start": a0, "audio_end": a1, "source_clips": []})
            continue
        clips = seg.get("source_clips")
        if not isinstance(clips, list) or not clips:
            errors.append(f"{sid}: source_clips must be a non-empty array")
            continue
        clean_clips = []
        for ci, clip in enumerate(clips):
            try:
                v0, v1 = float(clip["start"]), float(clip["end"])
            except Exception:
                errors.append(f"{sid}/clip {ci + 1}: start/end required numbers")
                continue
            if not (math.isfinite(v0) and math.isfinite(v1)):
                errors.append(f"{sid}/clip {ci + 1}: source times must be finite")
                continue
            if v0 < 0 or v1 <= v0 or v1 > source_duration + 0.05:
                errors.append(f"{sid}/clip {ci + 1}: source range outside video")
            clean_clips.append({"start": max(0.0, v0), "end": min(source_duration, v1)})
        checked.append({**seg, "id": sid, "audio_start": a0, "audio_end": a1, "source_clips": clean_clips})
    if errors:
        raise ValueError("; ".join(errors[:12]))
    return checked


def _manual_srt(segments):
    out = []
    for i, seg in enumerate(segments, 1):
        text = str(seg.get("narration_text", "")).strip()
        if not text:
            continue
        out.append(f"{i}\n{format_timestamp(float(seg['audio_start']))} --> {format_timestamp(float(seg['audio_end']))}\n{text}\n")
    return "\n".join(out)


def _manual_ass(segments, rw, rh):
    head = ("[Script Info]\nScriptType: v4.00+\nPlayResX: %d\nPlayResY: %d\n\n"
            "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, Bold, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
            "Style: Def,Padauk,%d,&H00FFFFFF,&H00000000,&H80000000,0,1,3,0,2,40,40,%d,1\n\n"
            "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
            % (rw, rh, max(18, int(rh * 0.045)), max(20, int(rh * 0.05))))
    lines = []
    for seg in segments:
        text = str(seg.get("narration_text", "")).strip().replace("\n", " ")
        if not text:
            continue
        lines.append("Dialogue: 0,%s,%s,Def,,0,0,0,,%s" % (
            _manual_ass_time(float(seg["audio_start"])), _manual_ass_time(float(seg["audio_end"])), text))
    return head + "\n".join(lines) + "\n"


def _manual_ass_time(seconds):
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:d}:{m:02d}:{s:05.2f}"


def _manual_video_filters(transforms, rw, rh):
    """Build the per-clip filter chain from the reversible transform options."""
    vf = f"fps=30,scale={rw}:{rh}:force_original_aspect_ratio=decrease,pad={rw}:{rh}:(ow-iw)/2:(oh-ih)/2"
    crop_mode = str(transforms.get("crop_mode", "keep"))
    if crop_mode == "vertical_916":
        vf += f",crop=ih*9/16:ih,scale={rw}:{rh}"
    elif crop_mode == "square_11":
        vf += f",crop=ih:ih,scale={rw}:{rh}"
    if transforms.get("blur_sensitive"):
        vf += ",boxblur=10:1"
    grade = str(transforms.get("color_grade", "none"))
    if grade == "muted":
        vf += ",eq=saturation=0.75:contrast=1.05"
    elif grade == "warm":
        vf += ",eq=saturation=1.05:gamma_r=1.03:gamma_b=0.98"
    elif grade == "bw":
        vf += ",hue=s=0"
    vf += ",format=yuv420p"
    return vf


def _manual_speed_ratio(clip_duration, target):
    if clip_duration <= 0.01 or target <= 0.01:
        return 1.0
    ratio = clip_duration / target
    if 0.5 <= ratio <= 2.0:
        return ratio
    return 1.0


def run_manual_render_job(job_id, video_path, audio_path, manifest, settings):
    temp_dir = tempfile.mkdtemp(prefix=f"manual_{job_id}_")
    preview = bool(settings.get("preview"))
    suffix = "_manual_preview" if preview else "_manual_final"
    out_path = os.path.join(DOWNLOAD_DIR, f"{job_id}{suffix}.mp4")
    srt_path = os.path.join(DOWNLOAD_DIR, f"{job_id}{suffix}.srt")
    qa = {"mode": "manual_review", "segments": 0, "source_duration_sec": None, "audio_duration_sec": None,
          "video_duration_sec": None, "duration_error_ms": None, "warnings": [], "status": "running",
          "preview": preview}
    job_qa[job_id] = qa
    set_job_state(job_id, state="running", part=0, total=0)
    try:
        source_duration = _manual_probe(video_path)
        audio_duration = _manual_probe(audio_path)
        segs = validate_manual_manifest(manifest, source_duration, audio_duration)
        policy = str(manifest.get("policy", "next_shot") or "next_shot")
        transforms = manifest.get("transforms") or {}
        qa.update({"segments": len(segs), "source_duration_sec": round(source_duration, 3),
                   "audio_duration_sec": round(audio_duration, 3), "policy": policy,
                   "transforms": {k: v for k, v in transforms.items() if v not in (False, "", "none", None)}})
        set_job_state(job_id, state="running", part=0, total=len(segs))
        resolution = str(settings.get("resolution", "1280:720"))
        try:
            rw, rh = [int(x) for x in resolution.split(":", 1)]
        except Exception:
            rw, rh = 1280, 720
        vf = _manual_video_filters(transforms, rw, rh)
        units = _manual_units(source_duration, quantum=5.0)
        concat_paths = []
        last_frame_holder = {"path": None}
        for idx, seg in enumerate(segs):
            sid = re.sub(r"[^A-Za-z0-9_-]", "_", str(seg["id"]))
            target = max(0.04, float(seg["audio_end"]) - float(seg["audio_start"]))
            clip_paths = []
            if seg.get("status") == "skipped":
                policy_skip = str(seg.get("skip_policy", "freeze"))
                holder = last_frame_holder["path"]
                if holder and policy_skip in ("hold_previous", "freeze"):
                    clip_paths.append(holder)
                else:
                    black = os.path.join(temp_dir, f"{idx:04d}_black.mp4")
                    run_cmd(["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=black:s={rw}x{rh}:r=30:d={max(0.5, target)}",
                             "-vf", "format=yuv420p", "-an", "-c:v", "libx264", "-preset", "veryfast", black],
                            f"Manual skip black {sid}", job_id)
                    clip_paths.append(black)
            else:
                # Per-segment Motion OFF is an explicit user choice from the
                # mobile review UI. Use the selected clip's last frame for
                # this narration beat and do not silently add another shot.
                if seg.get("motion_enabled") is False and seg.get("source_clips"):
                    chosen = seg["source_clips"][-1]
                    still = os.path.join(temp_dir, f"{idx:04d}_still.jpg")
                    run_cmd(["ffmpeg", "-y", "-ss", str(max(0.0, float(chosen["end"]) - 0.08)), "-i", video_path,
                             "-frames:v", "1", "-q:v", "2", still], f"Manual still {sid}", job_id)
                    frozen = os.path.join(temp_dir, f"{idx:04d}_frozen.mp4")
                    run_cmd(["ffmpeg", "-y", "-loop", "1", "-i", still, "-t", str(target),
                             "-vf", vf, "-an", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", frozen],
                            f"Manual freeze {sid}", job_id)
                    clip_paths.append(frozen)
                else:
                    for ci, clip in enumerate(seg["source_clips"]):
                        cp = os.path.join(temp_dir, f"{idx:04d}_{ci:03d}.mp4")
                        dur = max(0.05, clip["end"] - clip["start"])
                        run_cmd(["ffmpeg", "-y", "-ss", str(clip["start"]), "-i", video_path, "-t", str(dur),
                                 "-vf", vf, "-an", "-c:v", "libx264", "-preset", "veryfast",
                                 "-video_track_timescale", "90000", cp], f"Manual clip {sid}", job_id)
                        clip_paths.append(cp)

            list_path = os.path.join(temp_dir, f"{idx:04d}_clips.txt")
            with open(list_path, "w") as f:
                for cp in clip_paths:
                    f.write("file '" + cp.replace("'", "'\\''") + "'\n")
            joined = os.path.join(temp_dir, f"{idx:04d}_joined.mp4")
            run_cmd(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", joined],
                    f"Manual join {sid}", job_id)

            covered = sum(max(0.0, float(c["end"]) - float(c["start"])) for c in seg.get("source_clips", []))
            shortfall = target - covered
            speed_ratio = 1.0
            extra_clip = None
            if shortfall > 0.05 and policy == "next_shot" and seg.get("source_clips") and seg.get("motion_enabled") is not False:
                last_end = float(seg["source_clips"][-1]["end"])
                need = shortfall
                guard = 0
                while need > 0.05 and last_end < source_duration - 0.05 and guard < 12:
                    take = min(need, max(1.0, min(5.0, source_duration - last_end)))
                    ep = os.path.join(temp_dir, f"{idx:04d}_extra{guard}.mp4")
                    run_cmd(["ffmpeg", "-y", "-ss", str(last_end), "-i", video_path, "-t", str(take),
                             "-vf", vf, "-an", "-c:v", "libx264", "-preset", "veryfast", ep],
                            f"Manual next shot {sid}", job_id)
                    clip_paths.append(ep)
                    last_end += take
                    need -= take
                    guard += 1
                    extra_clip = ep
                if extra_clip:
                    list_path2 = os.path.join(temp_dir, f"{idx:04d}_clips2.txt")
                    with open(list_path2, "w") as f:
                        for cp in clip_paths:
                            f.write("file '" + cp.replace("'", "'\\''") + "'\n")
                    joined2 = os.path.join(temp_dir, f"{idx:04d}_joined2.mp4")
                    run_cmd(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path2, "-c", "copy", joined2],
                            f"Manual join2 {sid}", job_id)
                    joined = joined2
            elif shortfall > 0.05 and policy == "speed" and covered > 0.2:
                speed_ratio = _manual_speed_ratio(covered, target)

            norm = os.path.join(temp_dir, f"{idx:04d}_norm.mp4")
            if speed_ratio != 1.0:
                run_cmd(["ffmpeg", "-y", "-i", joined, "-vf", f"setpts=PTS/{speed_ratio},fps=30",
                         "-t", str(target), "-an", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", norm],
                        f"Manual speed fit {sid}", job_id)
            else:
                run_cmd(["ffmpeg", "-y", "-i", joined,
                         "-vf", f"tpad=stop_mode=clone:stop_duration={target},trim=duration={target},setpts=PTS-STARTPTS",
                         "-t", str(target), "-an", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", norm],
                        f"Manual fit {sid}", job_id)
            last_frame_holder["path"] = norm

            ap = os.path.join(temp_dir, f"{idx:04d}_audio.m4a")
            run_cmd(["ffmpeg", "-y", "-ss", str(seg["audio_start"]), "-t", str(target), "-i", audio_path,
                     "-af", f"apad,atrim=0:{target}", "-c:a", "aac", "-b:a", "160k", ap],
                    f"Manual audio {sid}", job_id)
            mux = os.path.join(temp_dir, f"{idx:04d}_mux.mp4")
            run_cmd(["ffmpeg", "-y", "-i", norm, "-i", ap, "-map", "0:v:0", "-map", "1:a:0",
                     "-c:v", "copy", "-c:a", "copy", "-shortest", mux], f"Manual mux {sid}", job_id)
            concat_paths.append(mux)
            set_job_state(job_id, state="running", part=idx + 1, total=len(segs))
            log_status(job_id, f"{'Preview' if preview else 'Final'} render {idx + 1}/{len(segs)} ခု ပြီးပါပြီ...")

        final_list = os.path.join(temp_dir, "final.txt")
        with open(final_list, "w") as f:
            for cp in concat_paths:
                f.write("file '" + cp.replace("'", "'\\''") + "'\n")
        raw_out = os.path.join(temp_dir, "raw_final.mp4")
        run_cmd(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", final_list, "-c", "copy", raw_out],
                "Manual final concat", job_id)

        if transforms.get("subtitles"):
            ass_path = os.path.join(temp_dir, "manual_subs.ass")
            with open(ass_path, "w", encoding="utf-8") as f:
                f.write(_manual_ass(segs, rw, rh))
            fonts_arg = FONTS_DIR.replace("\\", "/").replace(":", "\\:")
            run_cmd(["ffmpeg", "-y", "-i", raw_out, "-vf",
                     f"subtitles='{ass_path.replace(chr(92), '/').replace(':', chr(92) + ':')}':fontsdir='{fonts_arg}'",
                     "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-c:a", "copy", out_path],
                    "Manual subtitles", job_id)
        else:
            shutil.copyfile(raw_out, out_path)

        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(_manual_srt(segs))
        final_duration = _manual_probe(out_path)
        expected = sum(float(x["audio_end"]) - float(x["audio_start"]) for x in segs)
        qa.update({"video_duration_sec": round(final_duration, 3), "expected_duration_sec": round(expected, 3),
                   "duration_error_ms": round(abs(final_duration - expected) * 1000, 2), "status": "pass"})
        if qa["duration_error_ms"] > 1000 / 30:
            qa["warnings"].append("Final duration differs by more than one frame")
            qa["needs_review"] = True
        set_job_state(job_id, state="done", part=len(segs), total=len(segs))
        log_status(job_id, f"✅ Manual {'preview' if preview else 'final'} render ပြီးပါပြီ — {qa['duration_error_ms']}ms duration error")
    except Exception as e:
        qa["status"] = "error"
        qa["warnings"].append(str(e))
        set_job_state(job_id, state="error", part=0, total=qa.get("segments", 0))
        log_status(job_id, f"❌ Manual render error: {e}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _manual_thumb_path(job_id, time_sec):
    key = f"{float(time_sec):.2f}".replace(".", "_")
    return os.path.join(MANUAL_THUMB_DIR, job_id, f"{key}.jpg")


def _manual_make_thumb(job_id, video_path, time_sec):
    out = _manual_thumb_path(job_id, time_sec)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out):
        return out
    run_cmd(["ffmpeg", "-y", "-ss", str(max(0.0, float(time_sec))), "-i", video_path, "-frames:v", "1",
             "-vf", "scale=160:-2", "-q:v", "4", out], "Manual thumb", job_id)
    return out


@app.post("/api/manual/analyze")
async def manual_analyze(video_file: UploadFile = File(...), narration_file: UploadFile = File(...),
                         deep_transcript: str = Form("true"), movie_srt: str = Form(""),
                         policy: str = Form("next_shot"), _: str = Depends(require_token)):
    job_id = "manual_review_" + str(int(time.time() * 1000))
    vp = os.path.join(DOWNLOAD_DIR, job_id + "_source" + os.path.splitext(video_file.filename or ".mp4")[1].lower())
    ap = os.path.join(DOWNLOAD_DIR, job_id + "_narration" + os.path.splitext(narration_file.filename or ".mp3")[1].lower())
    try:
        with open(vp, "wb") as f:
            shutil.copyfileobj(video_file.file, f)
        with open(ap, "wb") as f:
            shutil.copyfileobj(narration_file.file, f)
        manifest = _manual_build_review(job_id, vp, ap, {
            "deep_transcript": str(deep_transcript).lower() in ("true", "1", "on", "yes"),
            "movie_srt": movie_srt or "",
            "policy": policy,
        })
        manual_jobs[job_id] = {"source": vp, "narration": ap, "manifest": manifest}
        mp = os.path.join(MANUAL_DIR, job_id + "_manifest.json")
        with open(mp, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        job_qa[job_id] = {"mode": "manual_review", "status": "draft", **manifest["stats"]}
        set_job_state(job_id, state="draft", part=0, total=manifest["stats"]["segments"], message="Review required")
        log_status(job_id, f"Manual draft ready — {manifest['stats']['segments']} beats "
                           f"(high {manifest['stats']['high']} / medium {manifest['stats']['medium']} / low {manifest['stats']['low']})")
        return {"ok": True, "job_id": job_id, "manifest": manifest, "stats": manifest["stats"],
                "video_url": f"/api/manual/media/{job_id}/video",
                "audio_url": f"/api/manual/media/{job_id}/audio"}
    except Exception as e:
        for q in (vp, ap):
            try:
                os.remove(q)
            except Exception:
                pass
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)


def _manual_job_files(job_id, prefix):
    if not re.fullmatch(r"manual_review_\d+", job_id or ""):
        raise HTTPException(404, "not found")
    for name in os.listdir(DOWNLOAD_DIR):
        if name.startswith(job_id + prefix):
            return os.path.join(DOWNLOAD_DIR, name)
    raise HTTPException(404, "media not found")


@app.get("/api/manual/media/{job_id}/{kind}")
def manual_media(job_id: str, kind: str):
    if kind == "video":
        path = _manual_job_files(job_id, "_source")
    elif kind == "audio":
        path = _manual_job_files(job_id, "_narration")
    else:
        raise HTTPException(404, "unknown media kind")
    return FileResponse(path)


@app.get("/api/manual/thumb/{job_id}")
def manual_thumb(job_id: str, t: float = 0.0):
    video_path = _manual_job_files(job_id, "_source")
    path = _manual_make_thumb(job_id, video_path, t)
    return FileResponse(path)


@app.get("/api/manual/draft/{job_id}")
def manual_draft(job_id: str):
    mp = os.path.join(MANUAL_DIR, job_id + "_manifest.json")
    if not os.path.exists(mp):
        raise HTTPException(404, "draft not found")
    return json.loads(open(mp, encoding="utf-8").read())


@app.post("/api/manual/draft/{job_id}")
async def manual_save_draft(job_id: str, request: Request, _: str = Depends(require_token)):
    mp = os.path.join(MANUAL_DIR, job_id + "_manifest.json")
    if not os.path.exists(mp):
        raise HTTPException(404, "draft not found")
    manifest = await request.json()
    vp = _manual_job_files(job_id, "_source")
    ap = _manual_job_files(job_id, "_narration")
    checked = validate_manual_manifest(manifest, _manual_probe(vp), _manual_probe(ap))
    manifest["segments"] = checked
    manifest["stats"] = _manual_stats(checked, manifest.get("stats", {}))
    with open(mp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    manual_jobs.setdefault(job_id, {})["manifest"] = manifest
    job_qa.setdefault(job_id, {}).update({"status": "saved", **manifest["stats"]})
    return {"ok": True, "manifest": manifest, "stats": manifest["stats"]}


def _manual_stats(segments, previous=None):
    stats = dict(previous or {})
    stats.update({
        "segments": len(segments),
        "high": sum(1 for s in segments if float(s.get("confidence", 0)) >= 0.8),
        "medium": sum(1 for s in segments if 0.6 <= float(s.get("confidence", 0)) < 0.8),
        "low": sum(1 for s in segments if float(s.get("confidence", 0)) < 0.6),
        "approved": sum(1 for s in segments if s.get("status") == "approved"),
        "changed": sum(1 for s in segments if s.get("status") == "changed"),
        "skipped": sum(1 for s in segments if s.get("status") == "skipped"),
        "unreviewed": sum(1 for s in segments if s.get("status") not in ("approved", "changed", "skipped")),
        "estimated_output_sec": round(sum(float(s["audio_end"]) - float(s["audio_start"]) for s in segments), 3),
    })
    stats["needs_review"] = sum(1 for s in segments
                                if float(s.get("confidence", 0)) < 0.8
                                and s.get("status") not in ("approved", "changed", "skipped"))
    return stats


@app.post("/api/manual/preview/{job_id}")
async def manual_preview(job_id: str, request: Request, background_tasks: BackgroundTasks,
                         resolution: str = "1280:720", _: str = Depends(require_token)):
    mp = os.path.join(MANUAL_DIR, job_id + "_manifest.json")
    if not os.path.exists(mp):
        raise HTTPException(404, "draft not found")
    manifest = await request.json()
    vp = _manual_job_files(job_id, "_source")
    ap = _manual_job_files(job_id, "_narration")
    checked = validate_manual_manifest(manifest, _manual_probe(vp), _manual_probe(ap))
    if not checked:
        return JSONResponse({"ok": False, "error": "no renderable segments"}, status_code=400)
    preview_id = job_id + "_p" + str(int(time.time() * 1000))
    background_tasks.add_task(run_manual_render_job, preview_id, vp, ap, manifest,
                              {"resolution": resolution, "preview": True})
    return {"ok": True, "job_id": preview_id, "preview_url": f"/downloads/{preview_id}_manual_preview.mp4"}


@app.post("/api/manual/final/{job_id}")
async def manual_final(job_id: str, request: Request, background_tasks: BackgroundTasks,
                       resolution: str = "1280:720", _: str = Depends(require_token)):
    mp = os.path.join(MANUAL_DIR, job_id + "_manifest.json")
    if not os.path.exists(mp):
        raise HTTPException(404, "draft not found")
    manifest = await request.json()
    vp = _manual_job_files(job_id, "_source")
    ap = _manual_job_files(job_id, "_narration")
    checked = validate_manual_manifest(manifest, _manual_probe(vp), _manual_probe(ap))
    stats = _manual_stats(checked, manifest.get("stats", {}))
    allow_unreviewed = str(manifest.get("allow_unreviewed", "false")).lower() in ("true", "1", "yes")
    if stats["needs_review"] and not allow_unreviewed:
        return JSONResponse({"ok": False, "unresolved": [s["id"] for s in checked
                                                         if float(s.get("confidence", 0)) < 0.8
                                                         and s.get("status") not in ("approved", "changed", "skipped")],
                             "error": f"Low-confidence scene {stats['needs_review']} ခုကို review မလုပ်ရသေးပါ"}, status_code=409)
    if str(manifest.get("rights_ack", "false")).lower() not in ("true", "1", "yes"):
        return JSONResponse({"ok": False, "error": "Copyright/reuse responsibility acknowledgement လိုအပ်ပါသည်"}, status_code=409)
    manifest["segments"] = checked
    manifest["stats"] = stats
    with open(mp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    final_id = job_id + "_final_" + str(int(time.time() * 1000))
    background_tasks.add_task(run_manual_render_job, final_id, vp, ap, manifest,
                              {"resolution": resolution, "preview": False})
    return {"ok": True, "job_id": final_id, "video_url": f"/downloads/{final_id}_manual_final.mp4",
            "srt_url": f"/downloads/{final_id}_manual_final.srt", "stats": stats}


@app.get("/api/manual/logs/{job_id}")
def manual_logs(job_id: str):
    return {"log": task_logs.get(job_id, "Waiting..."), "state": job_state.get(job_id, {}).get("state", "unknown"),
            "qa": job_qa.get(job_id, {})}


@app.get("/api/manual/template")
def manual_template():
    return {"version": "1.0", "mode": "manual_review", "sync_mode": "strict", "policy": "next_shot",
            "transforms": {"narration_only_audio": True, "blur_sensitive": False, "crop_mode": "keep",
                           "color_grade": "none", "subtitles": False, "overlay_text": ""},
            "segments": [{"id": "beat_0001", "audio_start": 0.0, "audio_end": 5.0,
                          "source_clips": [{"start": 12.0, "end": 17.0}],
                          "narration_text": "ဒီနေရာမှာ narration ရေးပါ",
                          "confidence": 0.9, "status": "approved", "reviewed_by_user": True}]}


@app.post("/api/manual/render")
async def manual_render(background_tasks: BackgroundTasks, video_file: UploadFile = File(...),
                        narration_file: UploadFile = File(...), manifest_file: UploadFile = File(...),
                        resolution: str = Form("1280:720"), _: str = Depends(require_token)):
    try:
        manifest = json.loads((await manifest_file.read()).decode("utf-8-sig"))
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"Invalid manifest JSON: {e}"}, status_code=400)
    job_id = "manual_" + str(int(time.time() * 1000))
    vp = os.path.join(DOWNLOAD_DIR, job_id + "_input.mp4")
    ap = os.path.join(DOWNLOAD_DIR, job_id + "_narration" + os.path.splitext(narration_file.filename or ".mp3")[1].lower())
    with open(vp, "wb") as f:
        shutil.copyfileobj(video_file.file, f)
    with open(ap, "wb") as f:
        shutil.copyfileobj(narration_file.file, f)
    try:
        validate_manual_manifest(manifest, _manual_probe(vp), _manual_probe(ap))
    except Exception as e:
        for q in (vp, ap):
            try:
                os.remove(q)
            except Exception:
                pass
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    background_tasks.add_task(run_manual_render_job, job_id, vp, ap, manifest, {"resolution": resolution})
    return {"ok": True, "job_id": job_id, "message": "Manual render started"}

# ==========================================
# 5. VOICE PREVIEW API
# ==========================================
@app.post("/api/preview_voice")
async def preview_voice(
    text:       str = Form("မင်္ဂလာပါ။ ဒီအသံကို ကြိုက်ပါသလား။"),
    voice_type: str = Form("edge"),
    voice_id:   str = Form("my-MM-ThihaNeural"),
    v_rate:     str = Form("1.3"),
    v_pitch:    str = Form("0"),
):
    import uuid
    pid  = str(uuid.uuid4())[:8]
    path = os.path.join(DOWNLOAD_DIR, f"preview_{pid}.mp3")
    try:
        await generate_audio_only(text, voice_type, voice_id, path, v_rate, v_pitch)
        if os.path.exists(path) and os.path.getsize(path) > 100:
            return {"status":"ok","url":f"/downloads/preview_{pid}.mp3"}
        else:
            return {"status":"error","message":"Audio file was empty or not created"}
    except asyncio.TimeoutError:
        return {"status":"error","message":"Voice generation timed out (90s limit). Server may have network issues with Microsoft/Google servers."}
    except Exception as e:
        return {"status":"error","message":str(e)}

# ==========================================
# 6. REST ENDPOINTS
# ==========================================
@app.get("/", response_class=HTMLResponse)
async def read_root():
    # The maintained frontend for this app is indexMRS.html.  Prefer it over
    # any stale legacy index.html so backend/frontend edits are actually what
    # users see at the root URL.
    candidates = [
        os.path.join(BASE_DIR, "indexMRS.html"),
        os.path.join(BASE_DIR, "index.html"),
        os.path.join(BASE_DIR, "index(8)_simplified.html"),
    ]
    for html_path in candidates:
        if os.path.exists(html_path):
            with open(html_path, "r", encoding="utf-8") as f:
                return f.read()
    raise HTTPException(404, "Frontend index file not found")

@app.post("/api/upload_font")
async def upload_font(font_file: UploadFile = File(...), _: str = Depends(require_token)):
    fname = os.path.basename(font_file.filename or "").strip()
    if not fname or not fname.lower().endswith(('.ttf', '.otf')):
        return {"status":"error","message":"Only .ttf / .otf allowed"}
    path = os.path.join(FONTS_DIR, fname)
    with open(path,"wb") as f: f.write(await font_file.read())
    return {"status":"success","filename":fname}

@app.get("/api/fonts")
async def get_fonts():
    if not os.path.exists(FONTS_DIR): return {"fonts":[]}
    fonts = sorted([f for f in os.listdir(FONTS_DIR) if f.lower().endswith(('.ttf','.otf'))])
    return {"fonts":fonts}

@app.get("/api/edge_voice_profiles")
async def edge_voice_profiles():
    """Return selectable Burmese delivery profiles, not separately trained voices."""
    return {
        "profiles": [{"id": key, **value} for key, value in EDGE_BURMESE_PROFILES.items()],
        "note": "Profiles use the two available Burmese Edge voices with bounded delivery offsets.",
    }

@app.post("/api/upload_chunk")
async def upload_chunk(
    upload_id:    str = Form(...),
    chunk_index:  int = Form(...),
    chunk:        UploadFile = File(...),
    _:            str = Depends(require_token),
):
    """
    Receives one small piece of a large file. Mobile-data connections
    routinely drop mid-transfer on multi-hundred-MB uploads; sending the
    file as many small chunks means a dropped connection only has to retry
    the current few-MB chunk, not restart the whole upload from zero.
    """
    safe_id = re.sub(r'[^a-zA-Z0-9_-]', '', upload_id)[:64]
    if not safe_id:
        return JSONResponse({"ok": False, "error": "invalid upload_id"}, status_code=400)
    chunk_dir = os.path.join(UPLOAD_STAGING_DIR, safe_id)
    os.makedirs(chunk_dir, exist_ok=True)
    chunk_path = os.path.join(chunk_dir, f"{chunk_index:08d}.part")
    with open(chunk_path, "wb") as f:
        shutil.copyfileobj(chunk.file, f)
    return {"ok": True, "received": chunk_index}

@app.put("/api/upload_chunk_raw/{upload_id}/{chunk_index}")
async def upload_chunk_raw(upload_id: str, chunk_index: int, request: Request, _: str = Depends(require_token)):
    """
    Same as /api/upload_chunk, but as a plain PUT with the raw chunk bytes
    as the body - no multipart/form-data wrapping, upload_id/chunk_index in
    the URL instead of form fields. Some routers/firewalls specifically
    filter or mangle POST+multipart bodies (which look like file uploads to
    deep packet inspection) while leaving a plain PUT with a binary body
    alone. Client falls back to this when the normal endpoint's requests
    never reach the server at all (not even a slow/failed attempt - no
    request logged server-side).
    """
    safe_id = re.sub(r'[^a-zA-Z0-9_-]', '', upload_id)[:64]
    if not safe_id:
        return JSONResponse({"ok": False, "error": "invalid upload_id"}, status_code=400)
    chunk_dir = os.path.join(UPLOAD_STAGING_DIR, safe_id)
    os.makedirs(chunk_dir, exist_ok=True)
    chunk_path = os.path.join(chunk_dir, f"{chunk_index:08d}.part")
    body = await request.body()
    if not body:
        return JSONResponse({"ok": False, "error": "empty body"}, status_code=400)
    with open(chunk_path, "wb") as f:
        f.write(body)
    return {"ok": True, "received": chunk_index}

@app.get("/api/upload_status/{upload_id}")
async def upload_status(upload_id: str):
    """Which chunk indices has the server already received for this
    upload_id? Lets the frontend resume a retried upload without
    re-sending chunks that already succeeded."""
    safe_id = re.sub(r'[^a-zA-Z0-9_-]', '', upload_id)[:64]
    chunk_dir = os.path.join(UPLOAD_STAGING_DIR, safe_id)
    received = []
    if os.path.isdir(chunk_dir):
        for fn in os.listdir(chunk_dir):
            if fn.endswith(".part"):
                try:
                    received.append(int(fn[:-5]))
                except ValueError:
                    pass
    return {"ok": True, "received": sorted(received)}

def _do_upload_finalize(upload_id: str, total_chunks: int, filename: str):
    """Assembles all received chunks back into one file, in order."""
    safe_id = re.sub(r'[^a-zA-Z0-9_-]', '', upload_id)[:64]
    chunk_dir = os.path.join(UPLOAD_STAGING_DIR, safe_id)
    if not os.path.isdir(chunk_dir):
        return JSONResponse({"ok": False, "error": "no chunks found for this upload_id"}, status_code=400)

    ext = os.path.splitext(filename)[1] or ".mp4"
    final_path = os.path.join(UPLOAD_STAGING_DIR, f"{safe_id}{ext}")
    try:
        with open(final_path, "wb") as out:
            for i in range(total_chunks):
                part_path = os.path.join(chunk_dir, f"{i:08d}.part")
                if not os.path.exists(part_path):
                    return JSONResponse({"ok": False, "error": f"missing chunk {i}/{total_chunks}"}, status_code=400)
                with open(part_path, "rb") as pf:
                    shutil.copyfileobj(pf, out)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    shutil.rmtree(chunk_dir, ignore_errors=True)
    return {"ok": True, "uploaded_id": safe_id + ext}

@app.post("/api/upload_finalize")
async def upload_finalize(
    upload_id:    str = Form(...),
    total_chunks: int = Form(...),
    filename:     str = Form("video.mp4"),
    _:            str = Depends(require_token),
):
    return _do_upload_finalize(upload_id, total_chunks, filename)

@app.get("/api/upload_finalize_get")
async def upload_finalize_get(
    upload_id:    str,
    total_chunks: int,
    filename:     str = "video.mp4",
    _:            str = Depends(require_token),
):
    """Same as /api/upload_finalize, but as a plain GET with query params -
    no request body at all. Chunks already switch from POST to PUT when a
    network blocks POST+multipart specifically, but finalize was still
    hardcoded to POST+multipart regardless - meaning every chunk could
    succeed via PUT and the upload would STILL fail right at the very end
    on this one leftover POST call. GET has no body to filter, so it goes
    through even where both POST and PUT+raw-body might not."""
    return _do_upload_finalize(upload_id, total_chunks, filename)

# ==========================================
# 4c. DIALOGUE V2 — three styles, audio-master timeline
#     recap / recap_dialogue / dialogue_only   (see dialogue_v2.py)
# ==========================================
try:
    import dialogue_v2 as dv2
except Exception as _dv2_import_err:   # keep the rest of the app alive
    dv2 = None
    print(f"[DV2] dialogue_v2.py could not be imported: {_dv2_import_err}")

DV2_CACHE_DIR = os.path.join(DATA_DIR, "dv2_cache")
os.makedirs(DV2_CACHE_DIR, exist_ok=True)
DV2_STYLE_LABEL = {
    "recap": "Recap ဇာတ်ကြောင်းပြော",
    "recap_dialogue": "Recap + Dialogue",
    "dialogue_only": "Dialogue သက်သက် (အသံတစ်သံတည်း)",
}


def _dv2_out_size(settings):
    res = str(settings.get('resolution', '1280:720'))
    try:
        rw, rh = int(res.split(':')[0]), int(res.split(':')[1])
    except Exception:
        rw, rh = 1280, 720
    ratio = _safe_aspect_ratio(settings.get('aspect_ratio', '16:9'))
    def _even(v):
        return max(2, int(round(float(v) / 2.0) * 2))
    if ratio == '9:16':
        h = _even(max(rw, rh)); w = _even(h * 9.0 / 16.0)
    elif ratio == '1:1':
        w = h = _even(min(rw, rh))
    else:
        w = _even(rw); h = _even(w * 9.0 / 16.0)
    return w, h


def _dv2_make_deps(video_id, settings):
    """Bind dialogue_v2's injected capabilities to this app's Gemini/TTS/ffmpeg helpers."""
    voice_type = str(settings.get('voice_model_type', 'edge') or 'edge')
    rate = normalize_tts_rate(settings.get('v_rate', 1.3))
    try:
        pitch = int(float(settings.get('v_pitch', 0) or 0))
    except (TypeError, ValueError):
        pitch = 0

    def _state_name(f):
        s = getattr(f, 'state', None)
        return getattr(s, 'name', s)

    def _json_config(schema):
        return genai_types.GenerateContentConfig(
            response_mime_type="application/json", max_output_tokens=16384, response_schema=schema)

    def llm_text(prompt, schema, label):
        def _call():
            c, m = get_gemini_model("recap")
            return _safe_response_json(c.models.generate_content(
                model=m, contents=[prompt], config=_json_config(schema)))
        return call_gemini_with_retry(_call, video_id=video_id, label=label, max_attempts=4, purpose="recap")

    def llm_video(prompt, clip_path, schema, label):
        holder = {}
        def _upload():
            f = gemini_client.files.upload(file=clip_path)
            guard = 0
            while _state_name(f) == 'PROCESSING' and guard < 120:
                time.sleep(3); guard += 1
                f = gemini_client.files.get(name=f.name)
            if _state_name(f) != 'ACTIVE':
                raise RuntimeError(f"Gemini file upload not ACTIVE ({_state_name(f)})")
            holder['f'] = f
        _upload()
        def _call():
            c, m = get_gemini_model("recap")
            return _safe_response_json(c.models.generate_content(
                model=m, contents=[prompt, holder['f']], config=_json_config(schema)))
        try:
            return call_gemini_with_retry(_call, video_id=video_id, label=label, max_attempts=4,
                                          purpose="recap", file_recovery_fn=_upload)
        finally:
            try:
                gemini_client.files.delete(name=holder['f'].name)
            except Exception:
                pass

    light_model = os.environ.get("GEMINI_LIGHT_MODEL", "").strip()

    def llm_light(prompt, schema, label):
        """Optional cheaper model for rewrite/shorten calls (set env GEMINI_LIGHT_MODEL); falls back to the main model."""
        if not light_model:
            return llm_text(prompt, schema, label)
        def _call():
            return _safe_response_json(gemini_client.models.generate_content(
                model=light_model, contents=[prompt], config=_json_config(schema)))
        try:
            return call_gemini_with_retry(_call, video_id=video_id, label=label + " (light)", max_attempts=3, purpose="light")
        except GeminiBillingError:
            raise
        except Exception as e:  # noqa: BLE001
            log_status(video_id, f"ℹ️ light model မရလို့ main model သုံးပါမည်: {str(e)[:120]}")
            return llm_text(prompt, schema, label)

    def tts(text, voice_id, out_path):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                generate_audio_only(text, voice_type, voice_id, out_path, rate, pitch)) or []
        finally:
            loop.close()

    def transcribe(path):
        return _manual_transcribe(path, label=video_id)

    def run(cmd, label):
        run_cmd(cmd, label, video_id)

    def progress(done, total, label=""):
        set_job_state(video_id, state="running", part=int(done), total=int(total))

    return dv2.Deps(llm_video=llm_video, llm_text=llm_text, tts=tts, transcribe=transcribe,
                    probe=_manual_probe, run=run, log=lambda m: log_status(video_id, m),
                    progress=progress, fatal=(GeminiBillingError,), llm_light=llm_light)


def _dv2_srt(infos):
    """SRT from real TTS word timings (falls back to even split)."""
    MAX_SUB = 28
    out, idx, last_end = [], 1, 0.0
    for sc in infos:
        dur, off = sc["dur"], sc["offset"]
        rows = []
        cur, cs, ce = "", None, None
        for w in sc["words"] or []:
            t = str(w.get("text", "")).strip()
            if not t:
                continue
            cand = (cur + " " + t).strip() if cur else t
            if len(cand) <= MAX_SUB or not cur:
                if cs is None:
                    cs = float(w["start"])
                cur, ce = cand, float(w["end"])
            else:
                rows.append((cur, cs, ce)); cur, cs, ce = t, float(w["start"]), float(w["end"])
        if cur:
            rows.append((cur, cs, ce))
        if not rows:
            toks = sc["text"].split() or [sc["text"]]
            lines, line = [], ""
            for t in toks:
                c = (line + " " + t).strip()
                if len(c) <= MAX_SUB or not line:
                    line = c
                else:
                    lines.append(line); line = t
            if line:
                lines.append(line)
            total = sum(len(x) for x in lines) or 1
            acc = 0.0
            for x in lines:
                seg = dur * len(x) / total
                rows.append((x, acc, acc + seg)); acc += seg
        for text, a, b in rows:
            a = max(0.0, min(a, dur)); b = max(a + 0.1, min(b, dur))
            g0 = max(off + a, last_end); g1 = max(g0 + 0.1, off + b)
            out.append(f"{idx}\n{format_timestamp(g0)} --> {format_timestamp(g1)}\n{text}\n")
            idx += 1; last_end = g1
    return "\n".join(out)


def _dv2_render(video_id, local_path, scenes, settings, font_path, temp_dir):
    """Cut footage to the exact TTS length of every beat (audio is the master clock)."""
    import concurrent.futures as cf
    os.makedirs(temp_dir, exist_ok=True)
    FR = 30
    out_w, out_h = _dv2_out_size(settings)
    res = f"{out_w}:{out_h}"
    chain = [f"fps={FR}", f"scale={res}:force_original_aspect_ratio=decrease",
             f"pad={res}:(ow-iw)/2:(oh-ih)/2", "setsar=1"]
    if settings.get('bypass_mirror') == 'true':
        chain.append("hflip")
    if settings.get('bypass_color') == 'true':
        chain.append("eq=brightness=0.02:saturation=1.05:contrast=1.02")
    if settings.get('bypass_rotation') == 'true':
        chain.append(f"rotate=1*PI/180:c=black:ow=iw:oh=ih,scale={res}")
    if settings.get('bypass_noise') == 'true':
        chain.append("noise=alls=1:allf=t+u")
    chain.append("format=yuv420p")
    vf_clip = ",".join(chain)

    infos = [None] * len(scenes)

    def _prep(i):
        sc = scenes[i]
        raw = os.path.join(temp_dir, f"r{i:05d}.wav")
        run_cmd(["ffmpeg", "-y", "-i", sc["_tts_path"], "-ar", "48000", "-ac", "1",
                 "-c:a", "pcm_s16le", raw], f"DV2 decode {i}", video_id)
        real = _manual_probe(raw)
        n = max(3, int(math.ceil(real * FR - 1e-6)) + 1)   # always >= audio, never cuts speech
        dq = n / FR
        wav = os.path.join(temp_dir, f"a{i:05d}.wav")
        run_cmd(["ffmpeg", "-y", "-i", raw, "-af", f"apad=whole_dur={dq:.6f}", "-t", f"{dq:.6f}",
                 "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le", wav], f"DV2 audio {i}", video_id)
        clip = os.path.join(temp_dir, f"v{i:05d}.mp4")
        start = parse_time_to_sec(sc["start_time"])
        run_cmd(["ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", local_path,
                 "-vf", vf_clip + f",tpad=stop_mode=clone:stop_duration={dq + 2:.3f}",
                 "-frames:v", str(n), "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
                 "-pix_fmt", "yuv420p", "-r", str(FR), "-video_track_timescale", "90000", clip],
                f"DV2 clip {i}", video_id)
        infos[i] = {"wav": wav, "clip": clip, "n": n, "dur": dq, "real": real}
        return i

    done = 0
    with cf.ThreadPoolExecutor(max_workers=3) as pool:
        futs = [pool.submit(_prep, i) for i in range(len(scenes))]
        for f in cf.as_completed(futs):
            f.result()
            done += 1
            if done % 5 == 0 or done == len(scenes):
                log_status(video_id, f"🎬 Beat {done}/{len(scenes)} ကို အသံအတိုင်း ဖြတ်တပ်ပြီးပါပြီ...")
                set_job_state(video_id, state="running", part=done, total=len(scenes))

    off, srt_infos = 0.0, []
    for sc, inf in zip(scenes, infos):
        shortfall = max(0.0, inf["dur"] - (parse_time_to_sec(sc["end_time"]) - parse_time_to_sec(sc["start_time"])))
        srt_infos.append({"offset": off, "dur": inf["dur"], "words": sc["_tts_words"],
                          "text": " ".join(t for t in re.split(r"\[[^\]]{1,60}\]\s*", sc["script"]) if t.strip()).strip(),
                          "shortfall": shortfall})
        off += inf["dur"]
    a_dur = off

    def _concat(paths, out, codec_args):
        lst = out + ".txt"
        with open(lst, "w", encoding="utf-8") as f:
            for p in paths:
                f.write("file '" + p.replace("'", "'\\''") + "'\n")
        run_cmd(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst, *codec_args, out],
                f"DV2 concat {os.path.basename(out)}", video_id)

    audio_all = os.path.join(temp_dir, "audio_all.wav")
    video_all = os.path.join(temp_dir, "video_all.mp4")
    _concat([i["wav"] for i in infos], audio_all, ["-c", "copy"])
    _concat([i["clip"] for i in infos], video_all, ["-c", "copy"])

    srt_path = os.path.join(DOWNLOAD_DIR, f"{video_id}_subs.srt")
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(_dv2_srt(srt_infos))
    script_path = os.path.join(DOWNLOAD_DIR, f"{video_id}_script.txt")
    with open(script_path, "w", encoding="utf-8") as f:
        o = 0.0
        for sc, inf in zip(scenes, infos):
            f.write(f"[{format_timestamp(o)[:-4]} - {format_timestamp(o + inf['dur'])[:-4]}] {sc['script']}\n")
            o += inf["dur"]
    audio_path = os.path.join(DOWNLOAD_DIR, f"{video_id}_tts.mp3")
    run_cmd(["ffmpeg", "-y", "-i", audio_all, "-c:a", "libmp3lame", "-b:a", "128k", audio_path],
            "DV2 tts mp3", video_id)

    # final pass: blur / subtitles (only re-encode when needed)
    filters, last_v, env = [], "0:v", None
    if settings.get('blur_enabled') == 'true':
        regions = []
        try:
            for r in json.loads(settings.get('blur_regions') or '[]'):
                regions.append((float(r.get('y', 80)), float(r.get('h', 20)), float(r.get('strength', 25))))
        except Exception:
            pass
        if not regions:
            regions = [(float(settings.get('blur_y', 80)), float(settings.get('blur_h', 20)),
                        float(settings.get('blur_strength', 25)))]
        for bi, (y, h, s) in enumerate(regions):
            filters.append(f"[{last_v}]split=2[base{bi}][bl{bi}]")
            filters.append(f"[bl{bi}]crop=iw:ih*{h}/100:0:ih*{y}/100,gblur=sigma={s}[blurred{bi}]")
            filters.append(f"[base{bi}][blurred{bi}]overlay=x=0:y=H*{y}/100[vb{bi}]")
            last_v = f"vb{bi}"
    if settings.get('sub_enabled') == 'true' and os.path.getsize(srt_path) > 0:
        fp = font_path if (font_path and os.path.exists(font_path)) else DEFAULT_FONT_PATH
        if not os.path.exists(fp):
            download_default_font()
        if os.path.exists(fp):
            c_map = {"White": "&H00FFFFFF", "Yellow": "&H0000FFFF", "Green": "&H0000FF00", "Cyan": "&H00FFFF00"}
            col = c_map.get(settings.get('sub_color'), "&H0000FFFF")
            fsz = max(8, min(120, int(float(settings.get('sub_size', '24')))))
            pos_pct = int(float(settings.get('sub_position', '20')))
            margin_v = max(10, int(10 + (650 * pos_pct / 100)))
            ass_path = os.path.join(temp_dir, "dv2_subs.ass")
            srt_to_ass(srt_path, ass_path, fp, fsz, col, margin_v)
            f_dir = os.path.dirname(os.path.abspath(fp))
            fc_cache = os.path.join(temp_dir, 'fc_cache'); os.makedirs(fc_cache, exist_ok=True)
            fc_conf = os.path.join(temp_dir, 'fonts.conf')
            with open(fc_conf, 'w') as _fc:
                _fc.write(f'<?xml version="1.0"?>\n<!DOCTYPE fontconfig SYSTEM "fonts.dtd">\n'
                          f'<fontconfig><dir>{f_dir}</dir><cachedir>{fc_cache}</cachedir></fontconfig>')
            env = dict(os.environ); env['FONTCONFIG_FILE'] = fc_conf
            filters.append(f"[{last_v}]ass='{ass_path.replace(chr(92), '/')}':fontsdir='{f_dir}'[vs]")
            last_v = "vs"
    final_video = os.path.join(DOWNLOAD_DIR, f"{video_id}_final.mp4")
    cmd = ["ffmpeg", "-y", "-i", video_all, "-i", audio_all]
    if filters:
        px = out_w * out_h
        br = max(800, min(6000, int(px / 300)))
        cmd += ["-filter_complex", ";".join(filters), "-map", f"[{last_v}]",
                "-c:v", "libx264", "-preset", "fast", "-b:v", f"{br}k", "-maxrate", f"{br * 2}k",
                "-bufsize", f"{br * 3}k", "-pix_fmt", "yuv420p"]
    else:
        cmd += ["-map", "0:v", "-c:v", "copy"]
    cmd += ["-map", "1:a", "-c:a", "aac", "-b:a", "128k", "-t", f"{a_dur:.6f}", "-movflags", "+faststart", final_video]
    run_cmd(cmd, "DV2 final render", video_id, env=env)
    return {"final": final_video, "audio_sec": a_dur, "shortfall_beats": sum(1 for x in srt_infos if x["shortfall"] > 0.001),
            "infos": infos, "out_size": res}


def run_dialogue_v2_job(video_id, local_path, font_path, settings):
    style = str(settings.get('dialogue_style', '')).strip().lower()
    qa = {"video_id": video_id, "mode": f"dialogue_v2:{style}", "scenes": 0, "warnings": [], "errors": [],
          "needs_review": False, "state": "running"}
    job_qa[video_id] = qa
    set_job_state(video_id, state="running", part=0, total=0)
    temp_dir = tempfile.mkdtemp(prefix=f"dv2_{video_id}_")
    try:
        if dv2 is None:
            raise RuntimeError("dialogue_v2.py ကို mainMRS.py နဲ့ တူတဲ့ folder ထဲ မထည့်ထားပါ")
        pm = probe_media(local_path)
        src_dur = float(pm.get('duration') or 0.0)
        vd = float(pm.get('video_duration') or 0.0)
        if vd > 0 and src_dur > 0 and abs(src_dur - vd) / max(src_dur, vd) > 0.15:
            src_dur = vd
        if not math.isfinite(src_dur) or src_dur < 30:
            raise RuntimeError(f"ဗီဒီယို duration မမှန်ပါ ({src_dur:.1f}s)")
        try:
            tmin = float(settings.get('target_minutes') or 0)
        except (TypeError, ValueError):
            tmin = 0.0
        try:
            ratio = float(settings.get('recap_ratio') or 40)
        except (TypeError, ValueError):
            ratio = 40.0
        target = tmin * 60.0 if tmin > 0 else src_dur * ratio / 100.0
        if target > src_dur * 0.9:
            target = src_dur * 0.9
            qa['warnings'].append("target length was capped to 90% of the source length")
        voices = {"NARRATOR": settings.get('narrator_voice_id') or settings.get('voice_id') or "my-MM-NilarNeural",
                  "MALE": settings.get('male_voice_id') or "my-MM-ThihaNeural",
                  "FEMALE": settings.get('female_voice_id') or "my-MM-NilarNeural"}
        style_key = str(settings.get('recap_style', RECAP_STYLE_DEFAULT))
        style_text = ""
        if style != "dialogue_only" or style_key == "short_drama_translation":
            try:
                style_text = RECAP_STYLES.get(style_key, RECAP_STYLES[RECAP_STYLE_DEFAULT])['voice'].format(target_lang="Burmese")
            except Exception:
                style_text = ""
        log_status(video_id, f"🎯 {DV2_STYLE_LABEL.get(style, style)} — target {target / 60:.1f} မိနစ် (source {src_dur / 60:.1f} မိနစ်)")
        deps = _dv2_make_deps(video_id, settings)
        voice_pool = {"male": ["male_deep", "male_hero", "male_urgent", "elder_male", "villain", "teen_boy", "child_boy", "comic", "male_calm"],
                      "female": ["female_strong", "female_young", "elder_female", "teen_girl", "female_sad", "child_girl", "female_soft", "energetic"]}
        result = dv2.run_pipeline(local_path, src_dur, style, target, voices, deps, DV2_CACHE_DIR,
                                  os.path.join(temp_dir, "plan"), style_text=style_text, voice_pool=voice_pool)
        scenes, dqa = result["scenes"], result["qa"]
        qa.update(dqa)
        qa['scenes'] = len(scenes)
        log_status(video_id, f"🎞️ Beat {len(scenes)} ခုကို အသံနဲ့ frame-exact ဖြတ်တပ်နေပါသည်...")
        set_job_state(video_id, state="running", part=0, total=len(scenes))
        if settings.get('overlays'):
            rr = _dv2_render_overlay(video_id, local_path, scenes, settings, font_path, os.path.join(temp_dir, "render"))
        else:
            rr = _dv2_render(video_id, local_path, scenes, settings, font_path, os.path.join(temp_dir, "render"))
        media = probe_media(rr["final"])
        err_ms = abs(media['duration'] - rr['audio_sec']) * 1000
        stream_ms = abs(media.get('video_duration', 0.0) - media.get('audio_duration', 0.0)) * 1000
        qa.update({"narration_duration_sec": round(rr['audio_sec'], 3),
                   "final_duration_sec": round(media['duration'], 3),
                   "audio_video_duration_error_ms": round(err_ms, 2),
                   "stream_duration_error_ms": round(stream_ms, 2),
                   "footage_shortfall_beats": rr["shortfall_beats"], "output_size": rr["out_size"]})
        problems = []
        if not media['has_video'] or not media['has_audio']:
            problems.append("final MP4 is missing a video or audio stream")
        if err_ms > 250 or stream_ms > 150:
            problems.append(f"A/V duration mismatch {err_ms:.0f}ms / streams {stream_ms:.0f}ms")
        if problems:
            qa['errors'].extend(problems)
            raise RuntimeError("; ".join(problems))
        if rr["shortfall_beats"]:
            qa['warnings'].append(f"{rr['shortfall_beats']} beat(s) had slightly less footage than audio (last frame held)")
        qa['needs_review'] = bool(qa['warnings'] or qa.get('dv2_warnings'))
        qa['state'] = "done"
        job_qa[video_id] = qa
        set_job_state(video_id, state="done", part=len(scenes), total=len(scenes))
        log_status(video_id, f"✅ ပြီးပါပြီ — {rr['audio_sec'] / 60:.1f} မိနစ် (target {target / 60:.1f}m, "
                             f"{qa.get('dv2_length_ratio', 0) * 100:.0f}%) · A/V error {err_ms:.0f}ms")
    except Exception as e:
        qa['state'] = "error"; qa['needs_review'] = True
        if str(e) not in qa['errors']:
            qa['errors'].append(str(e))
        job_qa[video_id] = qa
        set_job_state(video_id, state="error", part=0, total=0)
        log_status(video_id, f"❌ Dialogue V2 error: {e}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        try:
            if local_path and os.path.exists(local_path):
                os.remove(local_path)
        except Exception as ce:
            print(f"[DV2] cleanup failed: {ce}")


# ==========================================
# 4d. STORY STUDIO — long-form, audio-first story retelling (10 styles)
# ==========================================
try:
    import story_engine as se
except Exception as _se_import_err:
    se = None
    print(f"[STORY] story_engine.py could not be imported: {_se_import_err}")


STORY_CKPT_DIR = os.path.join(DATA_DIR, "story_ckpt")
os.makedirs(STORY_CKPT_DIR, exist_ok=True)


# ---- overlay model (all values are % of the OUTPUT frame, so preview == render) -------------------
_SUB_COLORS = {"White": "&H00FFFFFF", "Yellow": "&H0000FFFF", "Green": "&H0000FF00", "Cyan": "&H00FFFF00"}


def _pct(v, lo, hi, dflt):
    try:
        return max(lo, min(hi, float(v)))
    except (TypeError, ValueError):
        return dflt


def _story_overlay_cfg(settings):
    raw = settings.get('overlays')
    try:
        d = json.loads(raw) if isinstance(raw, str) and raw.strip() else (raw if isinstance(raw, dict) else {})
    except Exception:
        d = {}
    blur = []
    for r in (d.get('blur') or [])[:6]:
        if not isinstance(r, dict):
            continue
        x, y = _pct(r.get('x'), 0, 98, 0), _pct(r.get('y'), 0, 98, 80)
        w, h = _pct(r.get('w'), 1, 100, 100), _pct(r.get('h'), 1, 100, 20)
        blur.append({"x": x, "y": y, "w": min(w, 100 - x), "h": min(h, 100 - y), "s": _pct(r.get('s'), 1, 60, 25)})
    logo = None
    lg = d.get('logo')
    if isinstance(lg, dict) and lg.get('enabled', True) and settings.get('logo_path'):
        logo = {"x": _pct(lg.get('x'), 0, 100, 3), "y": _pct(lg.get('y'), 0, 100, 3), "w": _pct(lg.get('w'), 2, 60, 12)}
    sb = d.get('sub') if isinstance(d.get('sub'), dict) else {}
    mode = str(sb.get('mode') or settings.get('sub_mode') or 'burn')
    sub = {"mode": mode if mode in ('srt', 'burn', 'none') else 'burn',
           "cx": _pct(sb.get('cx'), 0, 100, 50), "cy": _pct(sb.get('cy'), 0, 100, 88),
           "size": _pct(sb.get('size'), 1.5, 14, 4.5), "maxw": _pct(sb.get('maxw'), 30, 100, 88),
           "color": sb.get('color') if sb.get('color') in _SUB_COLORS else str(settings.get('sub_color') or 'Yellow')}
    if sub["color"] not in _SUB_COLORS:
        sub["color"] = "Yellow"
    return {"blur": blur, "logo": logo, "sub": sub}


def _story_vf(settings, with_format=True):
    out_w, out_h = _dv2_out_size(settings)
    res = f"{out_w}:{out_h}"
    chain = ["fps=30", f"scale={res}:force_original_aspect_ratio=decrease", f"pad={res}:(ow-iw)/2:(oh-ih)/2", "setsar=1"]
    if settings.get('bypass_mirror') == 'true':
        chain.append("hflip")
    if settings.get('bypass_color') == 'true':
        chain.append("eq=brightness=0.02:saturation=1.05:contrast=1.02")
    if settings.get('bypass_rotation') == 'true':
        chain.append(f"rotate=1*PI/180:c=black:ow=iw:oh=ih,scale={res}")
    if settings.get('bypass_noise') == 'true':
        chain.append("noise=alls=1:allf=t+u")
    if with_format:
        chain.append("format=yuv420p")
    return ",".join(chain), out_w, out_h


def _story_sub_rows(words, text, dur, max_sub=28):
    """Subtitle rows [(text, start, end)] relative to the start of a beat (clamped to 0..dur)."""
    rows, cur, cs, ce = [], "", None, None
    for w in words or []:
        t = str(w.get("text", "")).strip()
        if not t:
            continue
        cand = (cur + " " + t).strip() if cur else t
        if len(cand) <= max_sub or not cur:
            if cs is None:
                cs = float(w["start"])
            cur, ce = cand, float(w["end"])
        else:
            rows.append((cur, cs, ce)); cur, cs, ce = t, float(w["start"]), float(w["end"])
    if cur:
        rows.append((cur, cs, ce))
    if not rows and text:
        toks = text.split() or [text]
        lines, line = [], ""
        for t in toks:
            c = (line + " " + t).strip()
            if len(c) <= max_sub or not line:
                line = c
            else:
                lines.append(line); line = t
        if line:
            lines.append(line)
        total = sum(len(x) for x in lines) or 1
        acc = 0.0
        for x in lines:
            seg = dur * len(x) / total
            rows.append((x, acc, acc + seg)); acc += seg
    out, last = [], 0.0
    for t, a, b in rows:
        a = max(last, min(a, dur)); b = max(a + 0.1, min(b, dur + 0.1))
        out.append((t, a, b)); last = b
    return out


def _ass_ts(t):
    t = max(0.0, t)
    cs = int(round(t * 100))
    return f"{cs // 360000}:{(cs // 6000) % 60:02d}:{(cs // 100) % 60:02d}.{cs % 100:02d}"


def _story_ass_text(rows, W, H, sub, font_name):
    fs = max(8, int(round(sub["size"] / 100.0 * H)))
    mlr = max(0, int(round((100.0 - sub["maxw"]) / 200.0 * W)))
    outline = max(1, int(round(H * 0.0035)))
    px, py = int(round(sub["cx"] / 100.0 * W)), int(round(sub["cy"] / 100.0 * H))
    head = (f"[Script Info]\nScriptType: v4.00+\nPlayResX: {W}\nPlayResY: {H}\nWrapStyle: 0\nScaledBorderAndShadow: yes\n\n"
            "[V4+ Styles]\nFormat: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,"
            "Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding\n"
            f"Style: Default,{font_name},{fs},{_SUB_COLORS[sub['color']]},&H000000FF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,{outline},1,5,{mlr},{mlr},0,1\n\n"
            "[Events]\nFormat: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n")
    lines = [f"Dialogue: 0,{_ass_ts(a)},{_ass_ts(b)},Default,,0,0,0,,{{\\an5\\pos({px},{py})}}{t.replace(chr(10), chr(92) + 'N')}"
             for t, a, b in rows]
    return head + "\n".join(lines) + "\n"


def _story_graph(base_chain, cfg, W, H, ass_path=None, has_logo=False, tail="", fonts_dir=None):
    """ONE filter graph used by both the real render and the exact preview (so they cannot drift apart)."""
    parts = [f"[0:v]{base_chain}[b0]"]
    cur = "b0"
    for i, r in enumerate(cfg["blur"]):
        x = min(W - 2, max(0, int(round(r["x"] / 100.0 * W))))
        y = min(H - 2, max(0, int(round(r["y"] / 100.0 * H))))
        w = max(2, min(W - x, int(round(r["w"] / 100.0 * W))))
        h = max(2, min(H - y, int(round(r["h"] / 100.0 * H))))
        parts.append(f"[{cur}]split=2[sa{i}][sb{i}]")
        parts.append(f"[sb{i}]crop={w}:{h}:{x}:{y},gblur=sigma={r['s']:.1f}[bl{i}]")
        parts.append(f"[sa{i}][bl{i}]overlay={x}:{y}[b{i + 1}]")
        cur = f"b{i + 1}"
    if has_logo and cfg["logo"]:
        lg = cfg["logo"]
        lw = max(4, int(round(lg["w"] / 100.0 * W / 2.0)) * 2)
        lx, ly = int(round(lg["x"] / 100.0 * W)), int(round(lg["y"] / 100.0 * H))
        parts.append(f"[1:v]scale={lw}:-2,format=rgba[lg]")
        parts.append(f"[{cur}][lg]overlay={lx}:{ly}[bl_logo]")
        cur = "bl_logo"
    if ass_path:
        parts.append(f"[{cur}]ass='{ass_path.replace(chr(92), '/')}':fontsdir='{fonts_dir or FONTS_DIR}'[bsub]")
        cur = "bsub"
    parts.append(f"[{cur}]format=yuv420p{tail}[vout]")
    return ";".join(parts)


def _story_font_env(temp_dir, font_path):
    fp = font_path if (font_path and os.path.exists(font_path)) else DEFAULT_FONT_PATH
    if not os.path.exists(fp):
        download_default_font()
    f_dir = os.path.dirname(os.path.abspath(fp))
    fc_cache = os.path.join(temp_dir, 'fc_cache'); os.makedirs(fc_cache, exist_ok=True)
    fc_conf = os.path.join(temp_dir, 'fonts.conf')
    with open(fc_conf, 'w') as _fc:
        _fc.write(f'<?xml version="1.0"?>\n<!DOCTYPE fontconfig SYSTEM "fonts.dtd">\n'
                  f'<fontconfig><dir>{f_dir}</dir><cachedir>{fc_cache}</cachedir></fontconfig>')
    env = dict(os.environ); env['FONTCONFIG_FILE'] = fc_conf
    return env, get_real_font_name(fp), f_dir


def _story_render_preview(video_id, src, t, settings, font_path, out_jpg, overlays=True, sample_text=""):
    """Render ONE frame exactly like the real render does (same graph) - used by the preview designer."""
    cfg = _story_overlay_cfg(settings)
    base, W, H = _story_vf(settings, with_format=False)
    tmp = tempfile.mkdtemp(prefix="story_pv_")
    try:
        if not overlays:
            run_cmd(["ffmpeg", "-y", "-ss", f"{max(0.0, t):.3f}", "-i", src, "-vf", base + ",format=yuv420p",
                     "-frames:v", "1", "-q:v", "3", out_jpg], "Story preview frame", video_id)
            return W, H
        has_logo = bool(cfg["logo"] and settings.get('logo_path') and os.path.exists(settings['logo_path']))
        if not has_logo:
            cfg["logo"] = None
        ass_path, env, f_dir = None, None, None
        if sample_text and cfg["sub"]["mode"] != "none":
            env, fname, f_dir = _story_font_env(tmp, font_path)
            ass_path = os.path.join(tmp, "pv.ass")
            with open(ass_path, "w", encoding="utf-8") as f:
                f.write(_story_ass_text([(sample_text, 0.0, 9.0)], W, H, cfg["sub"], fname))
        graph = _story_graph(base, cfg, W, H, ass_path, has_logo, "", f_dir)
        cmd = ["ffmpeg", "-y", "-ss", f"{max(0.0, t):.3f}", "-i", src]
        if has_logo:
            cmd += ["-i", settings['logo_path']]
        cmd += ["-filter_complex", graph, "-map", "[vout]", "-frames:v", "1", "-q:v", "3", out_jpg]
        run_cmd(cmd, "Story preview render", video_id, env=env)
        return W, H
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _story_concat(video_id, paths, out, codec_args):
    lst = out + ".txt"
    with open(lst, "w", encoding="utf-8") as f:
        for p_ in paths:
            f.write("file '" + p_.replace("'", "'\\''") + "'\n")
    run_cmd(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst, *codec_args, out],
            f"Story concat {os.path.basename(out)}", video_id)


class _StoryStore:
    """Renders each chapter as soon as it is written and keeps it on disk, so a crash costs one chapter, not the job.
    Blur / logo / burned subtitles are applied here, inside the per-beat encode that happens anyway."""
    FR = 30

    def __init__(self, video_id, local_path, settings, root, font_path=None):
        self.video_id, self.local_path, self.settings, self.root = video_id, local_path, settings, root
        self.font_path = font_path
        self.cfg = _story_overlay_cfg(settings)
        self.logo_path = settings.get('logo_path') if (self.cfg["logo"] and os.path.exists(settings.get('logo_path') or "")) else None
        if not self.logo_path:
            self.cfg["logo"] = None
        self.base, self.out_w, self.out_h = _story_vf(settings, with_format=False)
        self.vf = self.base + ",format=yuv420p"
        self.burn = self.cfg["sub"]["mode"] == "burn"
        self.use_graph = bool(self.cfg["blur"] or self.cfg["logo"] or self.burn)
        os.makedirs(root, exist_ok=True)

    def dir(self, i):
        return os.path.join(self.root, f"ch_{int(i):03d}")

    def load(self, i, sig):
        d = self.dir(i)
        try:
            with open(os.path.join(d, "meta.json"), "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            return None
        if meta.get("sig") != sig:
            return None
        for name in ("video.mp4", "audio.wav"):
            fp = os.path.join(d, name)
            if not os.path.exists(fp) or os.path.getsize(fp) < 1000:
                return None
        return meta.get("scenes") or None

    def save(self, i, sig, scenes):
        import concurrent.futures as cf
        d = self.dir(i)
        shutil.rmtree(d, ignore_errors=True)
        tmp = os.path.join(d, "tmp")
        os.makedirs(tmp)
        FR, vid = self.FR, self.video_id
        infos = [None] * len(scenes)
        env, fname, f_dir = (None, None, None)
        if self.burn:
            env, fname, f_dir = _story_font_env(tmp, self.font_path)

        def prep(k):
            sc = scenes[k]
            raw = os.path.join(tmp, f"r{k:04d}.wav")
            run_cmd(["ffmpeg", "-y", "-i", sc["_tts_path"], "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le", raw],
                    f"Story decode {i}.{k}", vid)
            real = _manual_probe(raw)
            n = max(3, int(math.ceil(real * FR - 1e-6)) + 1)
            dq = n / FR
            wav = os.path.join(tmp, f"a{k:04d}.wav")
            run_cmd(["ffmpeg", "-y", "-i", raw, "-af", f"apad=whole_dur={dq:.6f}", "-t", f"{dq:.6f}", "-ar", "48000",
                     "-ac", "1", "-c:a", "pcm_s16le", wav], f"Story audio {i}.{k}", vid)
            clip = os.path.join(tmp, f"v{k:04d}.mp4")
            start = parse_time_to_sec(sc["start_time"])
            tail = f",tpad=stop_mode=clone:stop_duration={dq + 2:.3f}"
            enc = ["-frames:v", str(n), "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
                   "-pix_fmt", "yuv420p", "-r", str(FR), "-video_track_timescale", "90000", clip]
            if not self.use_graph:
                cmd = ["ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", self.local_path, "-vf", self.vf + tail] + enc
                run_cmd(cmd, f"Story clip {i}.{k}", vid)
            else:
                ass_path = None
                if self.burn:
                    text = " ".join(t for t in re.split(r"\[[^\]]{1,60}\]\s*", sc["script"]) if t.strip()).strip()
                    rows = _story_sub_rows(sc["_tts_words"], text, dq)
                    if rows:
                        ass_path = os.path.join(tmp, f"s{k:04d}.ass")
                        with open(ass_path, "w", encoding="utf-8") as f:
                            f.write(_story_ass_text(rows, self.out_w, self.out_h, self.cfg["sub"], fname))
                graph = _story_graph(self.base, self.cfg, self.out_w, self.out_h, ass_path, bool(self.logo_path), tail, f_dir)
                cmd = ["ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", self.local_path]
                if self.logo_path:
                    cmd += ["-i", self.logo_path]
                cmd += ["-filter_complex", graph, "-map", "[vout]"] + enc
                run_cmd(cmd, f"Story clip {i}.{k}", vid, env=env)
            infos[k] = {"wav": wav, "clip": clip, "dur": dq}

        with cf.ThreadPoolExecutor(max_workers=3) as pool:
            for f in [pool.submit(prep, k) for k in range(len(scenes))]:
                f.result()
        _story_concat(vid, [x["wav"] for x in infos], os.path.join(d, "audio.wav"), ["-c", "copy"])
        _story_concat(vid, [x["clip"] for x in infos], os.path.join(d, "video.mp4"), ["-c", "copy"])
        metas = []
        for sc, inf in zip(scenes, infos):
            short = max(0.0, inf["dur"] - (parse_time_to_sec(sc["end_time"]) - parse_time_to_sec(sc["start_time"])))
            metas.append({"_dur": inf["dur"], "script": sc["script"], "_kind": sc["_kind"], "_chapter": sc["_chapter"],
                          "_drift": sc.get("_drift", 0.0), "_tts_words": sc["_tts_words"], "_shortfall": short})
        shutil.rmtree(tmp, ignore_errors=True)
        with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:       # written last = checkpoint is valid
            json.dump({"sig": sig, "scenes": metas}, f, ensure_ascii=False)
        log_status(vid, f"💾 အခန်း {int(i) + 1} ကို render ပြီး checkpoint သိမ်းပြီးပါပြီ ({sum(m['_dur'] for m in metas):.0f}s)")
        return metas


def _story_assemble(video_id, store, chapter_idxs, metas, settings, font_path, temp_dir, out_prefix=None):
    """Pure mux: chapter videos/audios are already final (overlays + burned subtitles are baked in per beat)."""
    os.makedirs(temp_dir, exist_ok=True)
    a_dur = sum(m["_dur"] for m in metas)
    vlist = os.path.join(temp_dir, "v.txt")
    alist = os.path.join(temp_dir, "a.txt")
    for lst, name in ((vlist, "video.mp4"), (alist, "audio.wav")):
        with open(lst, "w", encoding="utf-8") as f:
            for ci in chapter_idxs:
                f.write("file '" + os.path.join(store.dir(ci), name).replace("'", "'\\''") + "'\n")
    off, infos = 0.0, []
    for m in metas:
        infos.append({"offset": off, "dur": m["_dur"], "words": m["_tts_words"],
                      "text": " ".join(t for t in re.split(r"\[[^\]]{1,60}\]\s*", m["script"]) if t.strip()).strip()})
        off += m["_dur"]
    prefix = out_prefix or os.path.join(DOWNLOAD_DIR, video_id)
    srt_path = prefix + "_subs.srt"
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(_dv2_srt(infos))
    with open(prefix + "_script.txt", "w", encoding="utf-8") as f:
        o = 0.0
        for m in metas:
            f.write(f"[{format_timestamp(o)[:-4]} - {format_timestamp(o + m['_dur'])[:-4]}] {m['script']}\n")
            o += m["_dur"]
    run_cmd(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", alist, "-c:a", "libmp3lame", "-b:a", "128k",
             prefix + "_tts.mp3"], "Story tts mp3", video_id)
    final = prefix + "_final.mp4"
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", vlist, "-f", "concat", "-safe", "0", "-i", alist,
           "-map", "0:v", "-c:v", "copy", "-map", "1:a", "-c:a", "aac", "-b:a", "128k", "-t", f"{a_dur:.6f}",
           "-movflags", "+faststart", final]
    run_cmd(cmd, "Story final mux", video_id)
    return {"final": final, "audio_sec": a_dur, "out_size": f"{store.out_w}:{store.out_h}",
            "shortfall_beats": sum(1 for m in metas if m.get("_shortfall", 0) > 0.001), "burned": store.burn,
            "srt": srt_path}


def _dv2_render_overlay(video_id, local_path, scenes, settings, font_path, temp_dir):
    """Dialogue tool renderer with the Story Studio overlay model (blur rects / logo / placed subtitles)."""
    os.makedirs(temp_dir, exist_ok=True)
    store = _StoryStore(video_id, local_path, settings, os.path.join(temp_dir, "store"), font_path)
    for sc in scenes:
        sc["_chapter"] = 0
    metas = store.save(0, "dialogue", scenes)
    rr = _story_assemble(video_id, store, [0], metas, settings, font_path, os.path.join(temp_dir, "assemble"))
    rr["infos"] = [{"dur": m["_dur"]} for m in metas]
    return rr


def _story_purge_stale(max_age_days=3):
    now = time.time()
    try:
        for name in os.listdir(STORY_CKPT_DIR):
            p_ = os.path.join(STORY_CKPT_DIR, name)
            if os.path.isdir(p_) and now - os.path.getmtime(p_) > max_age_days * 86400:
                shutil.rmtree(p_, ignore_errors=True)
    except Exception as e:
        print(f"[STORY] purge failed: {e}")


def run_story_job(video_id, local_path, font_path, settings):
    style = str(settings.get('story_style', '')).strip()
    pilot = str(settings.get('pilot', 'false')) == 'true'
    qa = {"video_id": video_id, "mode": f"story:{style}{':pilot' if pilot else ''}", "scenes": 0,
          "warnings": [], "errors": [], "needs_review": False, "state": "running"}
    job_qa[video_id] = qa
    set_job_state(video_id, state="running", part=0, total=0)
    temp_dir = tempfile.mkdtemp(prefix=f"story_{video_id}_")
    keep_ckpt = True
    store = None
    try:
        if se is None or dv2 is None:
            raise RuntimeError("story_engine.py နဲ့ dialogue_v2.py ကို mainMRS.py နဲ့ folder တူတူ ထည့်ပြီး restart လုပ်ပါ")
        _story_purge_stale()
        pm = probe_media(local_path)
        src_dur = float(pm.get('duration') or 0.0)
        vd = float(pm.get('video_duration') or 0.0)
        if vd > 0 and src_dur > 0 and abs(src_dur - vd) / max(src_dur, vd) > 0.15:
            src_dur = vd
        if not math.isfinite(src_dur) or src_dur < 60:
            raise RuntimeError(f"ဗီဒီယို duration မမှန်ပါ ({src_dur:.1f}s)")
        try:
            tmin = float(settings.get('target_minutes') or 0)
        except (TypeError, ValueError):
            tmin = 0.0
        target = (tmin if tmin > 0 else 20.0) * 60.0
        if target > src_dur * 0.9:
            target = src_dur * 0.9
            qa['warnings'].append("target length was capped to 90% of the source length")
        voices = {"NARRATOR": settings.get('narrator_voice_id') or "my-MM-NilarNeural",
                  "MALE": settings.get('male_voice_id') or "my-MM-ThihaNeural",
                  "FEMALE": settings.get('female_voice_id') or "my-MM-NilarNeural"}
        voice_pool = {"male": ["male_deep", "male_hero", "male_urgent", "elder_male", "villain", "teen_boy", "child_boy", "comic", "male_calm"],
                      "female": ["female_strong", "female_young", "elder_female", "teen_girl", "female_sad", "child_girl", "female_soft", "energetic"]}
        est = se.estimate(src_dur, min(target, se.PILOT_SEC) if pilot else target, pilot)
        log_status(video_id, f"🎙️ Story Studio — {se.STYLES[style]['label']} · target {target / 60:.0f} မိနစ်{' (PILOT)' if pilot else ''} · source {src_dur / 60:.0f} မိနစ်")
        log_status(video_id, f"📋 ခန့်မှန်း: analysis {est['windows']} ကြိမ် · အခန်း {est['chapters']} · Gemini ခေါ်ယူမှု ~{est['calls']} (cache ရှိရင် လျော့မယ်)")
        logo_sig = ""
        if settings.get('logo_path') and os.path.exists(settings['logo_path']):
            logo_sig = f"{os.path.getsize(settings['logo_path'])}:{int(os.path.getmtime(settings['logo_path']))}"
        sig_src = json.dumps([dv2.cache_key(local_path), style, round(target), pilot, settings.get('pov'), voices,
                              settings.get('resolution'), settings.get('aspect_ratio'), settings.get('v_rate'),
                              settings.get('v_pitch'), _story_overlay_cfg(settings), logo_sig], sort_keys=True, default=str)
        ckpt_root = os.path.join(STORY_CKPT_DIR, hashlib.sha1(sig_src.encode("utf-8")).hexdigest()[:16])
        store = _StoryStore(video_id, local_path, settings, ckpt_root, font_path)
        resumed = [d for d in os.listdir(ckpt_root) if d.startswith("ch_")] if os.path.isdir(ckpt_root) else []
        if resumed:
            log_status(video_id, f"♻️ အရင် run ရဲ့ checkpoint အခန်း {len(resumed)} ခု တွေ့ပါသည် — ပြီးသားကို ကျော်ပြီး ဆက်လုပ်ပါမည်")
        deps = _dv2_make_deps(video_id, settings)
        result = se.run_story(local_path, src_dur, style, target, voices, deps, DV2_CACHE_DIR,
                              os.path.join(temp_dir, "plan"), pilot=pilot,
                              pov=str(settings.get('pov') or 'auto'), voice_pool=voice_pool, store=store)
        metas, sqa, chapters = result["scenes"], result["qa"], result["chapters"]
        qa.update(sqa)
        qa['scenes'] = len(metas)
        log_status(video_id, f"🎞️ အခန်း {len(chapters)} ခုကို ပေါင်းစပ်နေပါသည် (re-encode မလုပ်ပါ)...")
        set_job_state(video_id, state="running", part=len(chapters), total=len(chapters))
        rr = _story_assemble(video_id, store, [c["idx"] for c in chapters], metas, settings, font_path,
                             os.path.join(temp_dir, "assemble"))
        media = probe_media(rr["final"])
        err_ms = abs(media['duration'] - rr['audio_sec']) * 1000
        stream_ms = abs(media.get('video_duration', 0.0) - media.get('audio_duration', 0.0)) * 1000
        qa.update({"narration_duration_sec": round(rr['audio_sec'], 3), "final_duration_sec": round(media['duration'], 3),
                   "audio_video_duration_error_ms": round(err_ms, 2), "stream_duration_error_ms": round(stream_ms, 2),
                   "footage_shortfall_beats": rr["shortfall_beats"], "output_size": rr["out_size"],
                   "subtitles_burned": rr["burned"]})
        problems = []
        if not media['has_video'] or not media['has_audio']:
            problems.append("final MP4 is missing a video or audio stream")
        if err_ms > 250 or stream_ms > 150:
            problems.append(f"A/V duration mismatch {err_ms:.0f}ms / streams {stream_ms:.0f}ms")
        if problems:
            qa['errors'].extend(problems)
            raise RuntimeError("; ".join(problems))
        offs, acc = [], 0.0
        for m in metas:
            offs.append(acc)
            acc += m["_dur"]
        starts = [offs[c["first_scene"]] for c in chapters]
        ch_txt = se.chapters_text(chapters, starts)
        with open(os.path.join(DOWNLOAD_DIR, f"{video_id}_chapters.txt"), "w", encoding="utf-8") as f:
            f.write(ch_txt or "(အခန်း ၃ ခုအောက်ဆိုရင် YouTube chapter မထုတ်နိုင်ပါ)")
        meta = result["meta"]
        with open(os.path.join(DOWNLOAD_DIR, f"{video_id}_youtube.txt"), "w", encoding="utf-8") as f:
            f.write("TITLE OPTIONS\n" + "\n".join(f"- {t}" for t in meta["titles"]))
            f.write("\n\nDESCRIPTION\n" + meta["description"])
            if ch_txt:
                f.write("\n\n" + ch_txt)
            f.write("\n\nTAGS\n" + ", ".join(meta["tags"]) + "\n")
        qa['needs_review'] = bool(qa['warnings'] or qa.get('story_warnings'))
        qa['state'] = "done"
        job_qa[video_id] = qa
        keep_ckpt = False
        set_job_state(video_id, state="done", part=len(chapters), total=len(chapters))
        log_status(video_id, f"✅ ပြီးပါပြီ — {rr['audio_sec'] / 60:.1f} မိနစ် (target {target / 60:.1f}m, "
                             f"{qa.get('story_length_ratio', 0) * 100:.0f}%) · အခန်း {len(chapters)} · A/V error {err_ms:.0f}ms")
    except Exception as e:
        qa['state'] = "error"; qa['needs_review'] = True
        if str(e) not in qa['errors']:
            qa['errors'].append(str(e))
        job_qa[video_id] = qa
        set_job_state(video_id, state="error", part=0, total=0)
        extra = " — ပြီးသားအခန်းတွေ သိမ်းထားလို့ ဗီဒီယိုတူကို ပြန်တင်ပြီး ဆက်လုပ်နိုင်ပါတယ်" if keep_ckpt and store is not None else ""
        log_status(video_id, f"❌ Story Studio error: {e}{extra}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        if not keep_ckpt and store is not None:
            shutil.rmtree(store.root, ignore_errors=True)
        try:
            if local_path and os.path.exists(local_path):
                os.remove(local_path)
        except Exception as ce:
            print(f"[STORY] cleanup failed: {ce}")


@app.get("/api/story/styles")
async def story_styles():
    if se is None:
        return {"styles": [], "error": "story_engine.py not loaded"}
    return {"styles": se.list_styles()}


@app.post("/api/story/start")
async def story_start(
    background_tasks: BackgroundTasks,
    video_file:       Optional[UploadFile] = File(None),
    uploaded_id:      Optional[str] = Form(None),
    _:                str = Depends(require_token),
    story_style:      str = Form("fast_recap"),
    target_minutes:   str = Form("20"),
    pilot:            str = Form("false"),
    pov:              str = Form("auto"),
    narrator_voice_id: str = Form("narrator_fast_f"),
    male_voice_id:    str = Form("my-MM-ThihaNeural"),
    female_voice_id:  str = Form("my-MM-NilarNeural"),
    v_rate:           str = Form("1.3"),
    v_pitch:          str = Form("0"),
    aspect_ratio:     str = Form("16:9"),
    resolution:       str = Form("1280:720"),
    sub_mode:         str = Form("burn"),
    sub_size:         str = Form("24"),
    sub_color:        str = Form("Yellow"),
    sub_position:     str = Form("20"),
    selected_font:    str = Form(""),
):
    if se is None or dv2 is None or story_style not in se.STYLES:
        return JSONResponse({"status": "error", "message": "Story Studio မရနိုင်ပါ — story_engine.py နဲ့ dialogue_v2.py ကို mainMRS.py နဲ့ folder တူတူ ထည့်ပြီး restart လုပ်ပါ။"}, status_code=400)
    video_id = "story_" + str(int(time.time() * 1000))
    task_logs[video_id] = "စတင်ပြင်ဆင်နေပါသည်..."
    set_job_state(video_id, state="queued", part=0, total=0)
    file_path = os.path.join(DOWNLOAD_DIR, f"{video_id}_input.mp4")
    if uploaded_id:
        safe_id = re.sub(r'[^a-zA-Z0-9_.-]', '', uploaded_id)[:80]
        staged_path = os.path.join(UPLOAD_STAGING_DIR, safe_id)
        if not os.path.exists(staged_path):
            return JSONResponse({"status": "error", "message": "Uploaded file not found - please re-upload."}, status_code=400)
        shutil.move(staged_path, file_path)
    elif video_file:
        with open(file_path, "wb") as buf:
            shutil.copyfileobj(video_file.file, buf)
    else:
        return JSONResponse({"status": "error", "message": "No video provided."}, status_code=400)
    font_path = None
    if selected_font:
        base = os.path.basename(selected_font)
        if base.lower().endswith(('.ttf', '.otf')) and os.path.exists(os.path.join(FONTS_DIR, base)):
            font_path = os.path.join(FONTS_DIR, base)
    settings = {
        'story_style': story_style, 'target_minutes': target_minutes, 'pilot': 'true' if _form_bool(pilot) else 'false',
        'pov': pov, 'voice_model_type': 'edge', 'voice_id': narrator_voice_id,
        'narrator_voice_id': narrator_voice_id, 'male_voice_id': male_voice_id, 'female_voice_id': female_voice_id,
        'v_rate': v_rate, 'v_pitch': v_pitch, 'aspect_ratio': _safe_aspect_ratio(aspect_ratio), 'resolution': resolution,
        'sub_mode': sub_mode if sub_mode in ('srt', 'burn', 'none') else 'burn', 'sub_size': sub_size, 'sub_color': sub_color,
        'sub_position': sub_position, 'blur_enabled': 'false', 'bypass_mirror': 'false', 'bypass_color': 'false',
        'bypass_rotation': 'false', 'bypass_noise': 'false',
    }
    background_tasks.add_task(run_story_job, video_id, file_path, font_path, settings)
    return {"status": "started", "video_id": video_id}


# ==========================================
# 4e. STORY PROJECTS — one big movie, many parts, one continuous story (kept >= 3 days)
# ==========================================
def _story_proj_root():
    env = os.environ.get("STORY_PROJECT_DIR", "").strip()
    if env:
        return env, True
    if os.path.isdir("/data") and os.access("/data", os.W_OK):      # Hugging Face persistent storage mount
        return "/data/story_projects", True
    return os.path.join(DATA_DIR, "story_projects"), False


STORY_PROJ_DIR, STORY_PROJ_PERSISTENT = _story_proj_root()
os.makedirs(STORY_PROJ_DIR, exist_ok=True)
STORY_RETENTION_DAYS = 3
_story_run_lock = threading.Lock()
_story_proj_lock = threading.Lock()


def _proj_dir(pid):
    pid = re.sub(r'[^a-zA-Z0-9_]', '', str(pid))[:40]
    return os.path.join(STORY_PROJ_DIR, pid)


def _proj_load(pid):
    try:
        with open(os.path.join(_proj_dir(pid), "project.json"), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _proj_save(proj):
    with _story_proj_lock:
        d = _proj_dir(proj["id"])
        os.makedirs(d, exist_ok=True)
        tmp = os.path.join(d, "project.json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(proj, f, ensure_ascii=False)
        os.replace(tmp, os.path.join(d, "project.json"))


def _proj_touch(proj):
    """Retention is 3 days from the LAST activity, so a project in use never expires under the user."""
    proj["last_activity"] = time.time()
    proj["expires_at"] = proj["last_activity"] + STORY_RETENTION_DAYS * 86400


def _proj_purge_expired():
    now = time.time()
    try:
        for name in os.listdir(STORY_PROJ_DIR):
            pj = _proj_load(name)
            if pj and pj.get("expires_at", now + 1) < now and not pj.get("running"):
                shutil.rmtree(_proj_dir(name), ignore_errors=True)
    except Exception as e:
        print(f"[STORY] project purge failed: {e}")


def _proj_recover_stale():
    """A power cut / restart kills a running part without letting it clean up, leaving the project stuck on
    'running'.  When no job holds the run lock, nothing can really be running - so reopen those parts for restart
    (their finished chapters are still on disk and will be reused)."""
    if _story_run_lock.locked():
        return 0
    fixed = 0
    try:
        names = os.listdir(STORY_PROJ_DIR)
    except OSError:
        return 0
    for name in names:
        pj = _proj_load(name)
        if not pj:
            continue
        changed = False
        if pj.get("running"):
            pj["running"] = False
            changed = True
        for p_ in pj.get("parts", []):
            if p_.get("status") == "running":
                p_["status"] = "failed"
                p_["error"] = "ထုတ်နေစဉ် ရပ်သွားခဲ့ပါတယ် (စက်ပိတ်/restart) — 'ပြန်စ/ဆက်လုပ်' ကိုနှိပ်ပါ၊ ပြီးသားအခန်းတွေကို ပြန်သုံးမယ်"
                changed = True
                fixed += 1
        if changed:
            _proj_save(pj)
    return fixed


def _proj_public(pj):
    out = {k: v for k, v in pj.items() if k not in ("memory", "pending_memory", "settings")}
    out["hours_left"] = round(max(0.0, pj.get("expires_at", 0) - time.time()) / 3600.0, 1)
    out["persistent_storage"] = STORY_PROJ_PERSISTENT
    out["settings"] = {k: v for k, v in pj.get("settings", {}).items() if k not in ("logo_path",)}
    out["has_logo"] = bool(pj.get("settings", {}).get("logo_path"))
    out["logo_name"] = os.path.basename(pj.get("settings", {}).get("logo_path") or "")
    out["story_so_far"] = (pj.get("memory") or {}).get("summary", "")
    return out


def _proj_part_range(pj, part):
    """Start = where the committed previous part ended (exact).  End = a natural pause near start + part length."""
    if part.get("start") is not None and part.get("end") is not None:
        return part["start"], part["end"]
    start = float(pj["next_start"])
    plen = float(pj["part_min"]) * 60.0
    src_dur = float(pj["src_dur"])
    if src_dur - start <= plen * 1.4:
        return start, src_dur
    end = dv2.find_break(os.path.join(_proj_dir(pj["id"]), "source.mp4"), start + plen, src_dur)
    return start, max(start + 60.0, end)


def _proj_target(pj, start, end):
    try:
        pt = float(pj.get("part_target_min") or 0)
    except (TypeError, ValueError):
        pt = 0.0
    if pt > 0:
        return min(pt * 60.0, (end - start) * 0.9)
    ratio = float(pj.get("ratio") or 40)
    return max(45.0, (end - start) * ratio / 100.0)


def run_story_part_job(pid, part_no):
    pj = _proj_load(pid)
    video_id = f"proj_{pid}_p{int(part_no):03d}"
    qa = {"video_id": video_id, "mode": f"story_part:{pj['settings'].get('story_style') if pj else ''}", "warnings": [],
          "errors": [], "needs_review": False, "state": "running"}
    job_qa[video_id] = qa
    task_logs[video_id] = "စတင်ပြင်ဆင်နေပါသည်..."
    set_job_state(video_id, state="running", part=0, total=0)
    if not _story_run_lock.acquire(blocking=False):
        qa['state'] = "error"; qa['errors'].append("နောက်ထပ် Story job တစ်ခု run နေပါသည်")
        set_job_state(video_id, state="error", part=0, total=0)
        log_status(video_id, "❌ နောက်ထပ် Story job တစ်ခု run နေလို့ ခဏစောင့်ပါ")
        return
    temp_dir = tempfile.mkdtemp(prefix=f"storypart_{pid}_")
    part = None
    try:
        if pj is None:
            raise RuntimeError("project မတွေ့ပါ (သက်တမ်းကုန်သွားနိုင်ပါတယ်)")
        if se is None or dv2 is None:
            raise RuntimeError("story_engine.py / dialogue_v2.py မတွေ့ပါ")
        part = next(p_ for p_ in pj["parts"] if p_["no"] == part_no)
        part["status"], part["error"] = "running", ""
        pj["running"] = True
        _proj_touch(pj); _proj_save(pj)
        src = os.path.join(_proj_dir(pid), "source.mp4")
        settings = dict(pj["settings"])
        font_path = None
        sel = settings.get("selected_font")
        if sel and os.path.exists(os.path.join(FONTS_DIR, os.path.basename(sel))):
            font_path = os.path.join(FONTS_DIR, os.path.basename(sel))
        start, end = _proj_part_range(pj, part)
        part["start"], part["end"] = start, end
        _proj_save(pj)
        target = _proj_target(pj, start, end)
        total_est = max(1, int(math.ceil(float(pj["src_dur"]) / (float(pj["part_min"]) * 60.0))))
        log_status(video_id, f"🎬 Part {part_no}/~{total_est}: source {start / 60:.1f}–{end / 60:.1f} မိနစ် → recap target {target / 60:.1f} မိနစ်")
        deps = _dv2_make_deps(video_id, settings)
        overview = None
        if pj.get("prescan"):
            overview = se.prescan_overview(deps, src, float(pj["src_dur"]), DV2_CACHE_DIR)
            if overview is None:
                qa['warnings'].append("pre-scan overview မရပါ (စကားပြောမရှိ/မအောင်မြင်) — look-ahead မပါဘဲ ဆက်လုပ်ပါသည်")
        voices = {"NARRATOR": settings.get('narrator_voice_id') or "my-MM-NilarNeural",
                  "MALE": settings.get('male_voice_id') or "my-MM-ThihaNeural",
                  "FEMALE": settings.get('female_voice_id') or "my-MM-NilarNeural"}
        voice_pool = {"male": ["male_deep", "male_hero", "male_urgent", "elder_male", "villain", "teen_boy", "child_boy", "comic", "male_calm"],
                      "female": ["female_strong", "female_young", "elder_female", "teen_girl", "female_sad", "child_girl", "female_soft", "energetic"]}
        style_text = ""
        rs = settings.get("recap_style")
        if rs:
            try:
                style_text = RECAP_STYLES.get(rs, RECAP_STYLES[RECAP_STYLE_DEFAULT])['voice'].format(target_lang="Burmese")
            except Exception:
                style_text = ""
        part_dir = os.path.join(_proj_dir(pid), "parts", f"p{part_no:03d}")
        os.makedirs(part_dir, exist_ok=True)
        logo_sig = ""
        if settings.get('logo_path') and os.path.exists(settings['logo_path']):
            logo_sig = f"{os.path.getsize(settings['logo_path'])}:{int(os.path.getmtime(settings['logo_path']))}"
        ck_sig = hashlib.sha1(json.dumps([start, end, round(target), settings.get('story_style'), voices, settings.get('resolution'),
                                          settings.get('aspect_ratio'), _story_overlay_cfg(settings), logo_sig, settings.get('recap_style')],
                                         sort_keys=True, default=str).encode()).hexdigest()[:12]
        store = _StoryStore(video_id, src, settings, os.path.join(part_dir, f"ckpt_{ck_sig}"), font_path)
        for old in os.listdir(part_dir):                       # stale checkpoints from other settings
            if old.startswith("ckpt_") and old != f"ckpt_{ck_sig}":
                shutil.rmtree(os.path.join(part_dir, old), ignore_errors=True)
        result = se.run_part(src, float(pj["src_dur"]), settings.get('story_style'), voices, deps, DV2_CACHE_DIR,
                             os.path.join(temp_dir, "plan"), start, end, part_no, total_est, target, pj.get("memory") or {},
                             voice_pool=voice_pool, store=store, recap_style_text=style_text, overview=overview,
                             pov=str(settings.get('pov') or 'auto'), lines_path=os.path.join(part_dir, f"lines_{ck_sig}.json"))
        metas, chapters = result["scenes"], result["chapters"]
        qa.update(result["qa"])
        out_prefix = os.path.join(part_dir, "out")
        rr = _story_assemble(video_id, store, [c["idx"] for c in chapters], metas, settings, font_path,
                             os.path.join(temp_dir, "assemble"), out_prefix=out_prefix)
        media = probe_media(rr["final"])
        err_ms = abs(media['duration'] - rr['audio_sec']) * 1000
        stream_ms = abs(media.get('video_duration', 0.0) - media.get('audio_duration', 0.0)) * 1000
        qa.update({"narration_duration_sec": round(rr['audio_sec'], 3), "final_duration_sec": round(media['duration'], 3),
                   "audio_video_duration_error_ms": round(err_ms, 2), "stream_duration_error_ms": round(stream_ms, 2),
                   "footage_shortfall_beats": rr["shortfall_beats"], "output_size": rr["out_size"],
                   "subtitles_burned": rr["burned"]})
        problems = []
        if not media['has_video'] or not media['has_audio']:
            problems.append("final MP4 is missing a video or audio stream")
        if err_ms > 250 or stream_ms > 150:
            problems.append(f"A/V duration mismatch {err_ms:.0f}ms / streams {stream_ms:.0f}ms")
        if problems:
            qa['errors'].extend(problems)
            raise RuntimeError("; ".join(problems))
        offs, acc = [], 0.0
        for mm in metas:
            offs.append(acc); acc += mm["_dur"]
        ch_list = [{"title": c["title"], "t": offs[c["first_scene"]]} for c in chapters]
        with open(out_prefix + "_chapters.json", "w", encoding="utf-8") as f:
            json.dump(ch_list, f, ensure_ascii=False)
        pj = _proj_load(pid) or pj
        part = next(p_ for p_ in pj["parts"] if p_["no"] == part_no)
        part.update({"status": "done", "title": result["title"], "dur": round(rr["audio_sec"], 2), "error": "",
                     "final": f"parts/p{part_no:03d}/out_final.mp4", "files": {
                         "final": f"parts/p{part_no:03d}/out_final.mp4", "audio": f"parts/p{part_no:03d}/out_tts.mp3",
                         "srt": f"parts/p{part_no:03d}/out_subs.srt", "script": f"parts/p{part_no:03d}/out_script.txt"},
                     "qa": {k: v for k, v in qa.items() if k.startswith("story_") or k.endswith("_ms") or k in ("warnings",)}})
        pj["pending_memory"] = result["memory"]
        pj["pending_part"] = part_no
        pj["last_part_is_last"] = bool(result["is_last"])
        pj["running"] = False
        _proj_touch(pj); _proj_save(pj)                        # persist BEFORE reporting success
        qa['needs_review'] = bool(qa['warnings'] or qa.get('story_warnings'))
        qa['state'] = "done"
        job_qa[video_id] = qa
        set_job_state(video_id, state="done", part=len(chapters), total=len(chapters))
        log_status(video_id, f"✅ Part {part_no} ပြီးပါပြီ — {rr['audio_sec'] / 60:.1f} မိနစ် (target {target / 60:.1f}m, "
                             f"{qa.get('story_length_ratio', 0) * 100:.0f}%) · A/V error {err_ms:.0f}ms · စစ်ကြည့်ပြီး 'အတည်ပြုပြီး ဆက်သွား' နှိပ်ပါ")
    except Exception as e:
        qa['state'] = "error"; qa['needs_review'] = True
        if str(e) not in qa['errors']:
            qa['errors'].append(str(e))
        job_qa[video_id] = qa
        set_job_state(video_id, state="error", part=0, total=0)
        log_status(video_id, f"❌ Part {part_no} error: {e} — ပြီးသားအခန်းတွေ သိမ်းထားလို့ 'ပြန်လုပ်' နှိပ်ရင် ဆက်လုပ်ပါမယ်")
        try:
            pj2 = _proj_load(pid)
            if pj2:
                pt = next((p_ for p_ in pj2["parts"] if p_["no"] == part_no), None)
                if pt:
                    pt["status"], pt["error"] = "failed", str(e)[:300]
                    pj2["running"] = False
                    _proj_save(pj2)
        except Exception:
            pass
    finally:
        _story_run_lock.release()
        shutil.rmtree(temp_dir, ignore_errors=True)
        try:
            pj3 = _proj_load(pid)
            if pj3:
                pj3["running"] = False
                _proj_touch(pj3)
                _proj_save(pj3)
        except Exception:
            pass


def _proj_parse_srt(path):
    try:
        raw = open(path, encoding="utf-8").read().strip()
    except Exception:
        return []
    rows = re.findall(r'\d+\n(\d{2}):(\d{2}):(\d{2}),(\d{3}) --> (\d{2}):(\d{2}):(\d{2}),(\d{3})\n([\s\S]*?)(?=\n\n|$)', raw)
    out = []
    for r in rows:
        a = int(r[0]) * 3600 + int(r[1]) * 60 + int(r[2]) + int(r[3]) / 1000.0
        b = int(r[4]) * 3600 + int(r[5]) * 60 + int(r[6]) + int(r[7]) / 1000.0
        out.append((a, b, r[8].strip()))
    return out


def run_story_join_job(pid):
    pj = _proj_load(pid)
    video_id = f"proj_{pid}_join"
    task_logs[video_id] = "စတင်ပြင်ဆင်နေပါသည်..."
    set_job_state(video_id, state="running", part=0, total=0)
    qa = {"video_id": video_id, "mode": "story_join", "errors": [], "warnings": [], "state": "running"}
    job_qa[video_id] = qa
    if not _story_run_lock.acquire(blocking=False):
        set_job_state(video_id, state="error", part=0, total=0)
        log_status(video_id, "❌ နောက်ထပ် Story job တစ်ခု run နေပါသည်")
        return
    try:
        if pj is None:
            raise RuntimeError("project မတွေ့ပါ")
        parts = [p_ for p_ in pj["parts"] if p_["status"] == "approved"]
        if not parts:
            raise RuntimeError("အတည်ပြုပြီး အပိုင်း မရှိသေးပါ")
        root = _proj_dir(pid)
        jdir = os.path.join(root, "joined"); os.makedirs(jdir, exist_ok=True)
        lst = os.path.join(jdir, "list.txt")
        with open(lst, "w", encoding="utf-8") as f:
            for p_ in parts:
                f.write("file '" + os.path.join(root, p_["final"]).replace("'", "'\\''") + "'\n")
        final = os.path.join(jdir, "joined_final.mp4")
        log_status(video_id, f"🔗 Part {len(parts)} ခုကို re-encode မလုပ်ဘဲ ပေါင်းနေပါသည်...")
        run_cmd(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", "-movflags", "+faststart", final],
                "Story join", video_id)
        off, srt_rows, chapters = 0.0, [], []
        for p_ in parts:
            pdir = os.path.dirname(os.path.join(root, p_["final"]))
            for a, b, t in _proj_parse_srt(os.path.join(pdir, "out_subs.srt")):
                srt_rows.append((a + off, b + off, t))
            try:
                for c in json.load(open(os.path.join(pdir, "out_chapters.json"), encoding="utf-8")):
                    chapters.append({"title": c["title"], "t": c["t"] + off})
            except Exception:
                chapters.append({"title": p_.get("title") or f"Part {p_['no']}", "t": off})
            off += float(p_.get("dur") or 0.0)
        with open(os.path.join(jdir, "joined_subs.srt"), "w", encoding="utf-8") as f:
            f.write("\n".join(f"{i + 1}\n{format_timestamp(a)} --> {format_timestamp(b)}\n{t}\n" for i, (a, b, t) in enumerate(srt_rows)))
        ch_txt = se.chapters_text([{"title": c["title"]} for c in chapters], [c["t"] for c in chapters])
        with open(os.path.join(jdir, "joined_chapters.txt"), "w", encoding="utf-8") as f:
            f.write(ch_txt or "(အခန်း ၃ ခုအောက်ဆိုရင် YouTube chapter မထုတ်နိုင်ပါ)")
        deps = _dv2_make_deps(video_id, pj["settings"])
        mem = pj.get("memory") or {}
        meta = se.make_meta(deps, {"logline": mem.get("summary", "")[:300], "genre": "", "tone": "", "themes": []},
                            [{"title": c["title"]} for c in chapters], pj["settings"].get("story_style", ""))
        with open(os.path.join(jdir, "joined_youtube.txt"), "w", encoding="utf-8") as f:
            f.write("TITLE OPTIONS\n" + "\n".join(f"- {t}" for t in meta["titles"]) + "\n\nDESCRIPTION\n" + meta["description"])
            if ch_txt:
                f.write("\n\n" + ch_txt)
            f.write("\n\nTAGS\n" + ", ".join(meta["tags"]) + "\n")
        media = probe_media(final)
        qa.update({"final_duration_sec": round(media['duration'], 2), "parts_joined": len(parts), "state": "done"})
        pj = _proj_load(pid) or pj
        pj["joined"] = "joined/joined_final.mp4"
        _proj_touch(pj); _proj_save(pj)
        set_job_state(video_id, state="done", part=len(parts), total=len(parts))
        log_status(video_id, f"✅ ပေါင်းပြီးပါပြီ — {media['duration'] / 60:.1f} မိနစ်၊ part {len(parts)} ခု၊ chapters {len(chapters)}")
    except Exception as e:
        qa['state'] = "error"; qa['errors'].append(str(e))
        set_job_state(video_id, state="error", part=0, total=0)
        log_status(video_id, f"❌ Join error: {e}")
    finally:
        _story_run_lock.release()


def _proj_new_part(pj):
    no = len(pj["parts"]) + 1
    pj["parts"].append({"no": no, "start": None, "end": None, "status": "pending", "title": "", "dur": 0, "error": ""})
    return no


@app.get("/api/story/projects")
async def story_projects_list():
    _proj_recover_stale()
    _proj_purge_expired()
    out = []
    try:
        for name in sorted(os.listdir(STORY_PROJ_DIR)):
            pj = _proj_load(name)
            if pj:
                out.append(_proj_public(pj))
    except Exception:
        pass
    return {"projects": out, "storage": STORY_PROJ_DIR, "persistent": STORY_PROJ_PERSISTENT, "retention_days": STORY_RETENTION_DAYS}


@app.get("/api/story/project/{pid}")
async def story_project_get(pid: str):
    _proj_recover_stale()
    pj = _proj_load(pid)
    if not pj:
        return JSONResponse({"status": "error", "message": "project မတွေ့ပါ"}, status_code=404)
    return _proj_public(pj)


@app.post("/api/story/project/create")
async def story_project_create(
    uploaded_id:      Optional[str] = Form(None),
    video_file:       Optional[UploadFile] = File(None),
    _:                str = Depends(require_token),
    name:             str = Form("Story project"),
    story_style:      str = Form("fast_recap"),
    recap_style:      str = Form(""),
    part_min:         str = Form("10"),
    ratio:            str = Form("40"),
    part_target_min:  str = Form("0"),
    prescan:          str = Form("true"),
    pov:              str = Form("auto"),
    narrator_voice_id: str = Form("narrator_fast_f"),
    male_voice_id:    str = Form("male_calm"),
    female_voice_id:  str = Form("female_soft"),
    v_rate:           str = Form("1.3"),
    v_pitch:          str = Form("0"),
    aspect_ratio:     str = Form("16:9"),
    resolution:       str = Form("1280:720"),
    overlays:         str = Form("{}"),
):
    if se is None or dv2 is None or story_style not in se.STYLES:
        return JSONResponse({"status": "error", "message": "story_engine.py / dialogue_v2.py မတွေ့ပါ"}, status_code=400)
    _proj_purge_expired()
    pid = "p" + str(int(time.time() * 1000))
    root = _proj_dir(pid)
    os.makedirs(root, exist_ok=True)
    src = os.path.join(root, "source.mp4")
    if uploaded_id:
        safe_id = re.sub(r'[^a-zA-Z0-9_.-]', '', uploaded_id)[:80]
        staged = os.path.join(UPLOAD_STAGING_DIR, safe_id)
        if not os.path.exists(staged):
            shutil.rmtree(root, ignore_errors=True)
            return JSONResponse({"status": "error", "message": "Uploaded file not found - please re-upload."}, status_code=400)
        shutil.move(staged, src)
    elif video_file:
        with open(src, "wb") as buf:
            shutil.copyfileobj(video_file.file, buf)
    else:
        shutil.rmtree(root, ignore_errors=True)
        return JSONResponse({"status": "error", "message": "No video provided."}, status_code=400)
    pm = probe_media(src)
    dur = float(pm.get('duration') or 0.0)
    vd = float(pm.get('video_duration') or 0.0)
    if vd > 0 and dur > 0 and abs(dur - vd) / max(dur, vd) > 0.15:
        dur = vd
    if not math.isfinite(dur) or dur < 120:
        shutil.rmtree(root, ignore_errors=True)
        return JSONResponse({"status": "error", "message": f"ဗီဒီယို duration မမှန်ပါ ({dur:.1f}s)"}, status_code=400)
    try:
        pmin = float(dv2._clamp(float(part_min), 5, 40))
    except Exception:
        pmin = 10.0
    settings = {'story_style': story_style, 'recap_style': recap_style, 'pov': pov, 'voice_model_type': 'edge',
                'voice_id': narrator_voice_id, 'narrator_voice_id': narrator_voice_id, 'male_voice_id': male_voice_id,
                'female_voice_id': female_voice_id, 'v_rate': v_rate, 'v_pitch': v_pitch,
                'aspect_ratio': _safe_aspect_ratio(aspect_ratio), 'resolution': resolution, 'overlays': overlays,
                'sub_mode': 'burn', 'blur_enabled': 'false', 'bypass_mirror': 'false', 'bypass_color': 'false',
                'bypass_rotation': 'false', 'bypass_noise': 'false'}
    pj = {"id": pid, "name": (name or "Story project")[:80], "created": time.time(), "src_dur": dur, "part_min": pmin,
          "ratio": float(ratio or 40), "part_target_min": float(part_target_min or 0), "prescan": _form_bool(prescan),
          "next_start": 0.0, "parts": [], "memory": {}, "pending_memory": None, "finished": False, "joined": None,
          "settings": settings, "running": False, "estimated_parts": max(1, int(math.ceil(dur / (pmin * 60.0))))}
    _proj_touch(pj)
    _proj_save(pj)
    return _proj_public(pj)


@app.post("/api/story/project/{pid}/design")
async def story_project_design(pid: str, _: str = Depends(require_token),
                               overlays: str = Form("{}"), sub_mode: str = Form("burn"), sub_color: str = Form("Yellow"),
                               aspect_ratio: str = Form(""), resolution: str = Form(""),
                               logo_file: Optional[UploadFile] = File(None), remove_logo: str = Form("false")):
    pj = _proj_load(pid)
    if not pj:
        return JSONResponse({"status": "error", "message": "project မတွေ့ပါ"}, status_code=404)
    st = pj["settings"]
    new_ar = _safe_aspect_ratio(aspect_ratio) if aspect_ratio else st.get("aspect_ratio")
    new_res = resolution or st.get("resolution")
    if (new_ar != st.get("aspect_ratio") or new_res != st.get("resolution")):
        if pj["parts"]:
            return JSONResponse({"status": "error", "message": "Part ထုတ်ပြီးနောက် ratio/resolution မပြောင်းနိုင်ပါ (part တွေ ပေါင်းဖို့ အရွယ်တူရမယ်)"}, status_code=400)
        st["aspect_ratio"], st["resolution"] = new_ar, new_res
    st["overlays"] = overlays
    st["sub_mode"] = sub_mode if sub_mode in ("srt", "burn", "none") else "burn"
    st["sub_color"] = sub_color
    if _form_bool(remove_logo) and st.get("logo_path"):
        try:
            os.remove(st["logo_path"])
        except OSError:
            pass
        st.pop("logo_path", None)
    if logo_file is not None and getattr(logo_file, "filename", ""):
        ext = os.path.splitext(logo_file.filename)[1].lower()
        ext = ext if ext in (".png", ".jpg", ".jpeg", ".webp") else ".png"
        lp = os.path.join(_proj_dir(pid), "logo" + ext)
        with open(lp, "wb") as buf:
            shutil.copyfileobj(logo_file.file, buf)
        st["logo_path"] = lp
    _proj_touch(pj); _proj_save(pj)
    return _proj_public(pj)


@app.post("/api/story/project/{pid}/preview")
async def story_project_preview(pid: str, _: str = Depends(require_token), t: str = Form("60"), exact: str = Form("true"),
                                overlays: str = Form(""), sub_mode: str = Form(""), sample_text: str = Form("နမူနာ စာတန်း · Sample subtitle"),
                                aspect_ratio: str = Form(""), resolution: str = Form("")):
    pj = _proj_load(pid)
    if not pj:
        return JSONResponse({"status": "error", "message": "project မတွေ့ပါ"}, status_code=404)
    st = dict(pj["settings"])
    if aspect_ratio:
        st["aspect_ratio"] = _safe_aspect_ratio(aspect_ratio)
    if resolution:
        st["resolution"] = resolution
    if overlays:
        st["overlays"] = overlays
    if sub_mode in ("srt", "burn", "none"):
        st["sub_mode"] = sub_mode
    try:
        tt = max(0.0, min(float(t), float(pj["src_dur"]) - 1.0))
    except ValueError:
        tt = 60.0
    name = f"storypv_{pid}_{int(time.time() * 1000)}.jpg"
    out = os.path.join(PREVIEW_DIR, name)
    loop = asyncio.get_running_loop()
    want = _form_bool(exact, True)
    sub_on = _story_overlay_cfg(st)["sub"]["mode"] != "none"
    try:
        W, H = await loop.run_in_executor(None, lambda: _story_render_preview(
            f"pv_{pid}", os.path.join(_proj_dir(pid), "source.mp4"), tt, st, None, out, want,
            sample_text if (want and sub_on) else ""))
    except Exception as e:
        return JSONResponse({"status": "error", "message": f"preview မအောင်မြင်ပါ: {e}"}, status_code=500)
    _proj_touch(pj); _proj_save(pj)
    return {"url": f"/previews/{name}", "width": W, "height": H, "t": tt}


@app.post("/api/story/preview_upload")
async def story_preview_upload(_: str = Depends(require_token), uploaded_id: str = Form(...), t: str = Form("30"),
                               exact: str = Form("true"), overlays: str = Form(""), aspect_ratio: str = Form("16:9"),
                               resolution: str = Form("1280:720"), sample_text: str = Form("နမူနာ စာတန်း · Sample subtitle"),
                               bypass_mirror: str = Form("false"), bypass_color: str = Form("false"),
                               bypass_rotation: str = Form("false"), bypass_noise: str = Form("false"),
                               logo_file: Optional[UploadFile] = File(None)):
    safe_id = re.sub(r'[^a-zA-Z0-9_.-]', '', uploaded_id)[:80]
    src = os.path.join(UPLOAD_STAGING_DIR, safe_id)
    if not os.path.isfile(src):
        return JSONResponse({"status": "error", "message": "Upload မတွေ့ပါ — ပြန် upload လုပ်ပါ"}, status_code=404)
    st = {"aspect_ratio": _safe_aspect_ratio(aspect_ratio), "resolution": resolution, "overlays": overlays,
          "bypass_mirror": bypass_mirror, "bypass_color": bypass_color, "bypass_rotation": bypass_rotation,
          "bypass_noise": bypass_noise}
    stamp = int(time.time() * 1000)
    logo_tmp = None
    if logo_file is not None and getattr(logo_file, "filename", ""):
        logo_tmp = os.path.join(PREVIEW_DIR, f"storypv_logo_{stamp}.png")
        with open(logo_tmp, "wb") as buf:
            shutil.copyfileobj(logo_file.file, buf)
        st["logo_path"] = logo_tmp
    name = f"storypv_up_{stamp}.jpg"
    out = os.path.join(PREVIEW_DIR, name)
    try:
        tt = max(0.0, float(t))
    except ValueError:
        tt = 30.0
    want = _form_bool(exact, True)
    sub_on = _story_overlay_cfg(st)["sub"]["mode"] != "none"
    loop = asyncio.get_running_loop()
    try:
        W, H = await loop.run_in_executor(None, lambda: _story_render_preview(
            f"pvu_{stamp}", src, tt, st, None, out, want, sample_text if (want and sub_on) else ""))
    except Exception as e:
        return JSONResponse({"status": "error", "message": f"preview မအောင်မြင်ပါ: {e}"}, status_code=500)
    finally:
        if logo_tmp:
            try:
                os.remove(logo_tmp)
            except OSError:
                pass
    return {"url": f"/previews/{name}", "width": W, "height": H, "t": tt}


@app.post("/api/story/project/{pid}/run")
async def story_project_run(pid: str, background_tasks: BackgroundTasks, _: str = Depends(require_token)):
    _proj_recover_stale()
    pj = _proj_load(pid)
    if not pj:
        return JSONResponse({"status": "error", "message": "project မတွေ့ပါ"}, status_code=404)
    if _story_run_lock.locked() or pj.get("running"):
        return JSONResponse({"status": "error", "message": "Story job တစ်ခု run နေပါသည်"}, status_code=409)
    if pj.get("finished"):
        return JSONResponse({"status": "error", "message": "ဇာတ်ကားအားလုံး ပြီးပါပြီ — ပေါင်းမယ်/ဖျက်မယ်ကို သုံးပါ"}, status_code=400)
    pending = [p_ for p_ in pj["parts"] if p_["status"] in ("done",)]
    if pending:
        return JSONResponse({"status": "error", "message": "ပြီးထားတဲ့ part ကို အရင် 'အတည်ပြုပြီး ဆက်သွား' (သို့) 'ပြန်လုပ်' လုပ်ပါ"}, status_code=400)
    redo = [p_ for p_ in pj["parts"] if p_["status"] in ("failed", "running", "pending")]
    part_no = redo[0]["no"] if redo else _proj_new_part(pj)
    if not redo:
        # the new part starts exactly where the last approved one ended
        pj["parts"][-1]["start"] = float(pj["next_start"]); pj["parts"][-1]["end"] = None
    _proj_touch(pj); _proj_save(pj)
    background_tasks.add_task(run_story_part_job, pid, part_no)
    return {"status": "started", "video_id": f"proj_{pid}_p{part_no:03d}", "part": part_no}


@app.post("/api/story/project/{pid}/approve")
async def story_project_approve(pid: str, _: str = Depends(require_token)):
    pj = _proj_load(pid)
    if not pj:
        return JSONResponse({"status": "error", "message": "project မတွေ့ပါ"}, status_code=404)
    part = next((p_ for p_ in pj["parts"] if p_["status"] == "done"), None)
    if not part or pj.get("pending_memory") is None:
        return JSONResponse({"status": "error", "message": "အတည်ပြုစရာ part မရှိပါ"}, status_code=400)
    pj["memory"] = pj["pending_memory"]
    pj["pending_memory"] = None
    part["status"] = "approved"
    pj["next_start"] = float(part["end"])
    pj["finished"] = bool(pj.pop("last_part_is_last", False)) or pj["next_start"] >= float(pj["src_dur"]) - 5.0
    pdir = os.path.join(_proj_dir(pid), "parts", f"p{part['no']:03d}")
    for name in os.listdir(pdir) if os.path.isdir(pdir) else []:
        if name.startswith("ckpt_"):
            shutil.rmtree(os.path.join(pdir, name), ignore_errors=True)       # free the big per-chapter files
    _proj_touch(pj); _proj_save(pj)
    return _proj_public(pj)


@app.post("/api/story/project/{pid}/redo")
async def story_project_redo(pid: str, _: str = Depends(require_token)):
    _proj_recover_stale()
    pj = _proj_load(pid)
    if not pj:
        return JSONResponse({"status": "error", "message": "project မတွေ့ပါ"}, status_code=404)
    part = next((p_ for p_ in pj["parts"] if p_["status"] in ("done", "failed")), None)
    if not part:
        return JSONResponse({"status": "error", "message": "ပြန်လုပ်စရာ part မရှိပါ"}, status_code=400)
    part["status"] = "failed"
    part["error"] = "ပြန်လုပ်ရန် (user တောင်းဆို)"
    pj["pending_memory"] = None
    pdir = os.path.join(_proj_dir(pid), "parts", f"p{part['no']:03d}")
    for name in os.listdir(pdir) if os.path.isdir(pdir) else []:
        if name.startswith("lines_") or name.startswith("out_") or name.startswith("ckpt_"):
            p_ = os.path.join(pdir, name)
            shutil.rmtree(p_, ignore_errors=True) if os.path.isdir(p_) else os.remove(p_)
    _proj_touch(pj); _proj_save(pj)
    return _proj_public(pj)


@app.post("/api/story/project/{pid}/join")
async def story_project_join(pid: str, background_tasks: BackgroundTasks, _: str = Depends(require_token)):
    pj = _proj_load(pid)
    if not pj:
        return JSONResponse({"status": "error", "message": "project မတွေ့ပါ"}, status_code=404)
    if _story_run_lock.locked():
        return JSONResponse({"status": "error", "message": "Story job တစ်ခု run နေပါသည်"}, status_code=409)
    background_tasks.add_task(run_story_join_job, pid)
    return {"status": "started", "video_id": f"proj_{pid}_join"}


@app.post("/api/story/project/{pid}/extend")
async def story_project_extend(pid: str, _: str = Depends(require_token)):
    pj = _proj_load(pid)
    if not pj:
        return JSONResponse({"status": "error", "message": "project မတွေ့ပါ"}, status_code=404)
    _proj_touch(pj); _proj_save(pj)
    return _proj_public(pj)


@app.post("/api/story/project/{pid}/delete")
async def story_project_delete(pid: str, _: str = Depends(require_token)):
    pj = _proj_load(pid)
    if pj and pj.get("running"):
        return JSONResponse({"status": "error", "message": "run နေစဉ် မဖျက်နိုင်ပါ"}, status_code=409)
    shutil.rmtree(_proj_dir(pid), ignore_errors=True)
    return {"status": "deleted"}


@app.get("/api/story/project/{pid}/file")
async def story_project_file(pid: str, name: str, token: str = ""):
    if _APP_TOKEN and token != _APP_TOKEN:
        return JSONResponse({"status": "error", "message": "token လိုအပ်ပါသည်"}, status_code=401)
    root = os.path.abspath(_proj_dir(pid))
    path = os.path.abspath(os.path.join(root, name))
    if not path.startswith(root + os.sep) or not os.path.isfile(path) or os.path.basename(path) in ("project.json", "source.mp4"):
        return JSONResponse({"status": "error", "message": "file မတွေ့ပါ"}, status_code=404)
    return FileResponse(path, filename=os.path.basename(path))


try:
    _n_fixed = _proj_recover_stale()
    if _n_fixed:
        print(f"[STORY] {_n_fixed} part(s) interrupted by a restart were reopened for resume")
except Exception as _e_rec:
    print(f"[STORY] stale-part recovery failed: {_e_rec}")


@app.post("/api/process")
async def start_processing(
    background_tasks: BackgroundTasks,
    video_file:       Optional[UploadFile] = File(None),
    uploaded_id:      Optional[str] = Form(None),
    _:                str = Depends(require_token),
    logo_file:        Optional[UploadFile] = File(None),
    logo_pos:         str = Form("TOP LEFT"),
    logo_text:        str = Form(""),
    voice_model_type: str = Form("edge"),
    voice_id:         str = Form("my-MM-ThihaNeural"),
    target_lang:      str = Form("Burmese"),
    aspect_ratio:     str = Form("16:9"),
    analysis_engine:  str = Form("ranked_local_v2"),
    auto_analysis_chunk_min: str = Form("0"),
    sync_mode:       str = Form("strict"),
    fallback_policy: str = Form("freeze"),
    ai_budget:       str = Form("balanced"),
    max_audio_stretch: str = Form("1.12"),
    min_beat_sec:     str = Form("35"),
    make_shorts:      str = Form("false"),
    num_shorts:       str = Form("3"),
    make_teaser:      str = Form("false"),
    num_flashes:      str = Form("6"),
    continue_from_id: str = Form(""),
    split_count:      str = Form("14"),
    merge_parts:      str = Form("false"),
    recap_ratio:      str = Form("40"),
    recap_style:      str = Form("cinematic"),
    clean_output:     str = Form("false"),
    blur_enabled:     str = Form("false"),
    blur_y:           str = Form("80"),
    blur_h:           str = Form("20"),
    blur_strength:    str = Form("25"),
    freeze_enabled:   str = Form("false"),
    freeze_interval:  str = Form("5"),
    freeze_duration:  str = Form("2"),
    zoom_power:       str = Form("1.5"),
    v_rate:           str = Form("1.3"),
    v_pitch:          str = Form("0"),
    resolution:       str = Form("1280:720"),
    sub_enabled:      str = Form("false"),
    sub_size:         str = Form("24"),
    sub_color:        str = Form("Yellow"),
    sub_position:     str = Form("20"),
    sub_language:     str = Form(""),
    selected_font:    str = Form(""),
    bypass_mirror:    str = Form("false"),
    bypass_color:     str = Form("false"),
    bypass_rotation:  str = Form("false"),
    bypass_noise:     str = Form("false"),
    blur_regions:     str = Form("[]"),
    dialogue_mode:   str = Form("false"),
    narrator_voice_id: str = Form("my-MM-NilarNeural"),
    male_voice_id:   str = Form("my-MM-ThihaNeural"),
    female_voice_id: str = Form("my-MM-NilarNeural"),
    dialogue_style:  str = Form(""),
    target_minutes:  str = Form("0"),
    overlays:        str = Form(""),
):
    _dv2_style = str(dialogue_style or "").strip().lower()
    if _dv2_style and (dv2 is None or _dv2_style not in dv2.STYLES):
        return JSONResponse({"status": "error", "message": "Dialogue V2 မရနိုင်ပါ — dialogue_v2.py ကို mainMRS.py နဲ့ folder တူတူ ထည့်ပြီး restart လုပ်ပါ။"}, status_code=400)
    video_id  = "recap_" + str(int(time.time() * 1000))
    task_logs[video_id] = "စတင်ပြင်ဆင်နေပါသည်..."
    set_job_state(video_id, state="queued", part=0, total=0)
    # Prune stale entries so the dict does not grow forever on long-lived Spaces.
    cutoff = time.time() - 86400
    for k in list(job_state):
        if job_state[k].get("state") in ("done", "error", "unknown") and job_state[k].get("_t", 0) < cutoff:
            job_state.pop(k, None)
            task_logs.pop(k, None)
    file_path = os.path.join(DOWNLOAD_DIR, f"{video_id}_input.mp4")

    if uploaded_id:
        # File was sent via /api/upload_chunk + /api/upload_finalize
        safe_id = re.sub(r'[^a-zA-Z0-9_.-]', '', uploaded_id)[:80]
        staged_path = os.path.join(UPLOAD_STAGING_DIR, safe_id)
        if not os.path.exists(staged_path):
            return JSONResponse({"status": "error", "message": "Uploaded file not found - please re-upload."}, status_code=400)
        shutil.move(staged_path, file_path)
    elif video_file:
        with open(file_path, "wb") as buf:
            shutil.copyfileobj(video_file.file, buf)
    else:
        return JSONResponse({"status": "error", "message": "No video provided."}, status_code=400)

    logo_path = None
    if logo_file and logo_file.filename != '':
        logo_path = os.path.join(DOWNLOAD_DIR, f"{video_id}_logo.png")
        with open(logo_path,"wb") as f: f.write(await logo_file.read())

    # P1-1: sanitize selected_font — only allow .ttf/.otf basenames in FONTS_DIR
    if selected_font:
        sel_basename = os.path.basename(selected_font)
        if sel_basename and sel_basename.lower().endswith(('.ttf', '.otf')) and os.path.exists(os.path.join(FONTS_DIR, sel_basename)):
            font_path = os.path.join(FONTS_DIR, sel_basename)
        else:
            font_path = None
    else:
        font_path = None

    _dialogue_enabled = _form_bool(dialogue_mode)
    _aspect_ratio = _safe_aspect_ratio(aspect_ratio)
    # Dialogue is an Auto-derived workflow with multi-voice TTS; force the
    # Gemini Auto analysis path so a stale legacy engine cannot make it look
    # like ordinary narrator-only Auto.
    if _dialogue_enabled:
        analysis_engine = "gemini_auto"
    settings = {
        'logo_pos':logo_pos,'logo_text':logo_text,
        'voice_model_type':voice_model_type,'voice_id':voice_id,
        'target_lang':target_lang,'aspect_ratio':_aspect_ratio,'analysis_engine':analysis_engine,
        'workflow_mode': 'dialogue' if _dialogue_enabled else 'auto',
        'auto_analysis_chunk_min':auto_analysis_chunk_min,
        'sync_mode':sync_mode,'fallback_policy':fallback_policy,'ai_budget':ai_budget,'max_audio_stretch':max_audio_stretch,
        'min_beat_sec':min_beat_sec,'make_shorts':make_shorts,'num_shorts':num_shorts,'make_teaser':make_teaser,'num_flashes':num_flashes,'continue_from_id':continue_from_id,'split_count':split_count,'merge_parts':merge_parts,
        'recap_ratio':recap_ratio,'recap_style':recap_style,'clean_output':clean_output,
        'blur_enabled':blur_enabled,'blur_y':blur_y,'blur_h':blur_h,'blur_strength':blur_strength,
        'freeze_enabled':freeze_enabled,'freeze_interval':freeze_interval,
        'freeze_duration':freeze_duration,'zoom_power':zoom_power,
        'v_rate':v_rate,'v_pitch':v_pitch,'resolution':resolution,
        'sub_enabled':sub_enabled,'sub_size':sub_size,'sub_color':sub_color,'sub_position':sub_position,
        'sub_language':sub_language,'blur_regions':blur_regions,
        'bypass_mirror':bypass_mirror,'bypass_color':bypass_color,
        'bypass_rotation':bypass_rotation,'bypass_noise':bypass_noise,
        'dialogue_mode': 'true' if _dialogue_enabled else 'false',
        'narrator_voice_id': narrator_voice_id,
        'male_voice_id': male_voice_id,
        'female_voice_id': female_voice_id,
        # Keep an explicit export summary for QA/log consumers. The historical
        # field names are retained because the renderer already consumes them.
        'transform_summary': {
            'mirror': _form_bool(bypass_mirror),
            'color_grade': _form_bool(bypass_color),
            'micro_rotation': _form_bool(bypass_rotation),
            'film_noise': _form_bool(bypass_noise),
            'blur': _form_bool(blur_enabled),
            'freeze_frame': _form_bool(freeze_enabled),
            'subtitles': _form_bool(sub_enabled),
            'aspect_ratio': _aspect_ratio,
        },
    }
    if _dialogue_enabled and _dv2_style:
        settings['dialogue_style'] = _dv2_style
        settings['target_minutes'] = target_minutes
        if overlays and overlays.strip() not in ("", "{}"):
            settings['overlays'] = overlays
            settings['logo_path'] = logo_path
        background_tasks.add_task(run_dialogue_v2_job, video_id, file_path, font_path, settings)
        return {"status":"started","video_id":video_id}
    background_tasks.add_task(run_advanced_pipeline, video_id, file_path, logo_path, font_path, settings)
    return {"status":"started","video_id":video_id}

@app.get("/api/logs/{video_id}")
async def get_logs(video_id: str):
    s = job_state.get(video_id, {})
    return {
        "log": task_logs.get(video_id, "စတင်ပြင်ဆင်နေပါသည်..."),
        "state": s.get("state", "unknown"),
        "part": s.get("part", 0),
        "total": s.get("total", 0),
        "progress": s.get("progress", ""),
    }

@app.get("/api/preview/{video_id}")
async def get_preview(video_id: str):
    state = preview_states.get(video_id, {})
    frame_url = ""
    if state.get("frame") and os.path.exists(state["frame"]):
        frame_url = f"/previews/{os.path.basename(state['frame'])}"
    return {
        "scene": state.get("scene", 0),
        "total": state.get("total", 0),
        "step": state.get("step", ""),
        "frame_url": frame_url,
        "log": task_logs.get(video_id, ""),
    }

@app.get("/api/qa/{video_id}")
async def get_qa(video_id: str):
    """Return the machine-readable sync report for the UI and debugging."""
    return job_qa.get(video_id, {"video_id": video_id, "state": "pending", "needs_review": True, "warnings": ["QA not available yet"]})


@app.get("/api/files")
def list_files():
    if not os.path.exists(DOWNLOAD_DIR): return []
    files = os.listdir(DOWNLOAD_DIR)
    return sorted([f for f in files if any(k in f for k in ["_final.mp4","_tts.mp3","_subs.srt","_script.txt","_short_"])], reverse=True)

@app.get("/api/recap/past_jobs")
def list_past_jobs():
    """Completed jobs that have a saved script - candidates for the
    'Continue from previous part' picker, so a movie split across
    SEPARATE manual uploads (not the same job's auto-split) can still
    read as one continuous story. One entry per base video_id even if
    that job was itself auto-split into several internal sub-parts."""
    if not os.path.exists(DOWNLOAD_DIR):
        return []
    by_base = {}  # base video_id -> (part_num_or_None, fname)
    for fname in os.listdir(DOWNLOAD_DIR):
        m = re.match(r'^(.+?)_script\.txt$', fname)
        if not m:
            continue
        stem = m.group(1)
        pm = re.match(r'^(.+?)_part(\d+)$', stem)
        if pm:
            base, part_num = pm.group(1), int(pm.group(2))
        else:
            base, part_num = re.sub(r'_combined$', '', stem), None
        # Prefer the EARLIEST part (part 1, or the plain file) as the
        # preview source - it shows the story's actual beginning.
        cur = by_base.get(base)
        if cur is None or (part_num is not None and (cur[0] is None or part_num < cur[0])):
            by_base[base] = (part_num, fname)

    out = []
    for vid, (_, fname) in by_base.items():
        try:
            with open(os.path.join(DOWNLOAD_DIR, fname), "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
            preview = re.sub(r'^\[.*?\]\s*', '', first_line)[:60]
        except Exception:
            preview = ""
        try:
            mtime = os.path.getmtime(os.path.join(DOWNLOAD_DIR, fname))
        except Exception:
            mtime = 0
        out.append({"video_id": vid, "preview": preview, "mtime": mtime})
    out.sort(key=lambda x: x["mtime"], reverse=True)
    return out[:30]

# ==========================================
# 5. VOICE CLONE + TTS STUDIO
# ==========================================
voice_tasks = {}
try:
    import voice_clone as vc

    @app.get("/api/voice/profiles")
    def voice_list_profiles():
        return {"profiles": vc.list_profiles()}

    @app.post("/api/voice/create")
    async def voice_create(
        name: str = Form(...),
        provider: str = Form("f5-tts"),
        edge_voice: str = Form("ms-MM-MintheNeural"),
        edge_rate: str = Form("+0%"),
        edge_pitch: str = Form("+0Hz"),
        _: str = Depends(require_token),
        sample_file: Optional[UploadFile] = File(None),
    ):
        if provider == "f5-tts":
            if not sample_file or not sample_file.filename:
                return JSONResponse({"ok": False, "error": "Audio sample file ထည့်ပါ (F5-TTS voice clone အတွက်)"}, status_code=400)
            audio_bytes = await sample_file.read()
            if len(audio_bytes) < 1000:
                return JSONResponse({"ok": False, "error": "Audio sample အနည်းဆုံး 10KB ရှိရမယ်"}, status_code=400)
            result = await asyncio.to_thread(vc.create_f5_voice, name, audio_bytes, sample_file.filename)
        elif provider == "voxcpm2":
            if not sample_file or not sample_file.filename:
                return JSONResponse({"ok": False, "error": "Reference voice sample file လိုအပ်ပါတယ်"}, status_code=400)
            audio_bytes = await sample_file.read()
            result = await asyncio.to_thread(vc.create_voxcpm_voice, name, audio_bytes, sample_file.filename)
        elif provider == "edge-tts":
            result = await asyncio.to_thread(vc.create_edge_voice, name, edge_voice, edge_rate, edge_pitch)
        else:
            return JSONResponse({"ok": False, "error": f"Unknown provider: {provider}"}, status_code=400)
        if not result.get("ok"):
            return JSONResponse(result, status_code=400)
        return result

    @app.post("/api/voice/generate")
    async def voice_generate(
        text: str = Form(...),
        profile_id: int = Form(...),
        async_mode: str = Form(""),
        _: str = Depends(require_token),
    ):
        if not text.strip():
            return JSONResponse({"ok": False, "error": "Text ထည့်ပါ"}, status_code=400)

        def _work(task_id):
            try:
                def _progress(i, total, msg):
                    voice_tasks[task_id] = {"status": "running", "current": i, "total": total, "message": msg}
                audio_bytes, ct = vc.generate_speech_long(text, profile_id, progress=_progress)
                ext = ".wav" if "wav" in ct else ".mp3"
                fname = f"vs_{profile_id}_{int(time.time()*1000)}{ext}"
                out_path = os.path.join(AUDIO_DIR, fname)
                with open(out_path, "wb") as f:
                    f.write(audio_bytes)
                voice_tasks[task_id] = {
                    "status": "done",
                    "current": 1, "total": 1, "message": "ပြီးပါပြီ",
                    "filename": fname, "url": f"/audio/{fname}",
                    "size_kb": round(len(audio_bytes) / 1024),
                }
            except asyncio.TimeoutError:
                voice_tasks[task_id] = {"status": "error", "error": "Voice generation timed out (90s). Space က နှေးနေသည် / အလုပ်များနေသည်"}
            except Exception as e:
                voice_tasks[task_id] = {"status": "error", "error": str(e)[:300]}

        if str(async_mode) == "1":
            task_id = f"t{int(time.time()*1000)}"
            voice_tasks[task_id] = {"status": "running", "current": 0, "total": 0, "message": "စတင်နေပါသည်..."}
            import threading
            threading.Thread(target=_work, args=(task_id,), daemon=True).start()
            while len(voice_tasks) > 50:
                voice_tasks.pop(next(iter(voice_tasks)))
            return {"ok": True, "task_id": task_id, "async": True}
        else:
            task_id = f"t{int(time.time()*1000)}"
            _work(task_id)
            t = voice_tasks[task_id]
            if t.get("status") == "done":
                return {"ok": True, "filename": t["filename"], "url": t["url"], "size_kb": t["size_kb"]}
            return JSONResponse({"ok": False, "error": t.get("error", "Generation failed")}, status_code=500)

    @app.get("/api/voice/status/{task_id}")
    def voice_status(task_id: str):
        t = voice_tasks.get(task_id)
        if not t:
            return JSONResponse({"ok": False, "error": "Task not found"}, status_code=404)
        out = {"ok": True, "status": t.get("status"), "current": t.get("current", 0),
               "total": t.get("total", 0), "message": t.get("message", "")}
        if t.get("status") == "done":
            out.update({"filename": t.get("filename"), "url": t.get("url"), "size_kb": t.get("size_kb")})
        if t.get("status") == "error":
            out["error"] = t.get("error", "Failed")
        return out

    @app.post("/api/voice/design")
    async def voice_design(
        text: str = Form(...),
        control: str = Form(""),
        cfg_value: float = Form(2.0),
        _: str = Depends(require_token),
    ):
        """Generate speech with a designed voice (no reference audio needed)."""
        if not text.strip():
            return JSONResponse({"ok": False, "error": "Text ထည့်ပါ"}, status_code=400)
        try:
            audio = await asyncio.to_thread(vc.generate_voice_design, text, control, cfg_value)
            if not audio:
                return JSONResponse({"ok": False, "error": "VoxCPM2 API failed"}, status_code=500)
            fname = f"vd_{int(time.time()*1000)}.wav"
            out_path = os.path.join(AUDIO_DIR, fname)
            with open(out_path, "wb") as f:
                f.write(audio)
            return {"ok": True, "filename": fname, "url": f"/audio/{fname}",
                    "size_kb": round(len(audio) / 1024)}
        except Exception as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    @app.delete("/api/voice/delete/{profile_id}")
    def voice_delete(profile_id: int, _: str = Depends(require_token)):
        vc.delete_profile(profile_id)
        return {"ok": True}

    @app.post("/api/voice/backup")
    def voice_backup(_: str = Depends(require_token)):
        try:
            blob = vc.build_backup_bytes()
            fname = f"voice_backup_{int(time.time())}.zip"
            out_path = os.path.join(DOWNLOAD_DIR, fname)
            with open(out_path, "wb") as f:
                f.write(blob)
            return {"ok": True, "filename": fname, "url": f"/downloads/{fname}",
                    "size_kb": round(len(blob) / 1024)}
        except Exception as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    @app.post("/api/voice/restore")
    async def voice_restore(backup_file: UploadFile = File(...), _: str = Depends(require_token)):
        data = await backup_file.read()
        if len(data) > 200 * 1024 * 1024:
            return JSONResponse({"ok": False, "error": "Backup zip 200MB ထက် ကြီးလွန်းပါသည်"}, status_code=400)
        res = vc.restore_backup_bytes(data)
        if not res.get("ok"):
            return JSONResponse(res, status_code=400)
        return res

    @app.get("/api/recap_styles")
    def recap_styles():
        return {"default": RECAP_STYLE_DEFAULT, "styles": [
            {"id": k, "label": v["label"], "label_mm": v.get("label_mm", v["label"]),
             "featured": k in RECAP_STYLE_FEATURED}
            for k, v in RECAP_STYLES.items()
        ]}

    @app.get("/api/audio/library")
    def audio_library():
        items = []
        try:
            for fn in sorted(os.listdir(AUDIO_DIR), reverse=True):
                if not fn.endswith(('.mp3', '.wav', '.m4a', '.ogg')):
                    continue
                p = os.path.join(AUDIO_DIR, fn)
                if not os.path.isfile(p):
                    continue
                dur = None
                try:
                    dur = round(float(subprocess.check_output(
                        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                         "-of", "default=noprint_wrappers=1:nokey=1", p]).decode().strip()), 1)
                except Exception:
                    pass
                items.append({
                    "filename": fn, "url": f"/audio/{fn}",
                    "size_kb": round(os.path.getsize(p) / 1024), "duration": dur,
                    "created": time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(p))),
                })
        except Exception as e:
            print(f"[AUDIO LIB] Error: {e}")
        return {"items": items}

    @app.post("/api/audio/upload")
    async def audio_upload(file: UploadFile = File(...), _: str = Depends(require_token)):
        """
        Add a pre-made audio file (e.g. generated on Kaggle/Colab, or any
        other tool) straight into the Audio Library, so it's usable
        anywhere in the app exactly like a live-generated voice track —
        no server-side TTS call needed for this file.
        """
        ext = os.path.splitext(file.filename or "")[1].lower()
        if ext not in (".mp3", ".wav", ".m4a", ".ogg"):
            return JSONResponse({"ok": False, "error": "mp3/wav/m4a/ogg ဖိုင်မျိုးသာ လက်ခံပါတယ်"}, status_code=400)
        audio_bytes = await file.read()
        if len(audio_bytes) < 1000:
            return JSONResponse({"ok": False, "error": "Audio file အနည်းဆုံး 1KB ရှိရမယ်"}, status_code=400)
        safe_base = re.sub(r"[^A-Za-z0-9_.-]", "_", os.path.basename(file.filename))
        fname = f"{int(time.time())}_{safe_base}"
        out_path = os.path.join(AUDIO_DIR, fname)
        with open(out_path, "wb") as f:
            f.write(audio_bytes)
        return {"ok": True, "filename": fname, "url": f"/audio/{fname}"}

    @app.delete("/api/audio/delete/{filename}")
    def audio_delete(filename: str, _: str = Depends(require_token)):
        safe = os.path.basename(filename)
        if not safe or safe.startswith('_concat'):
            return JSONResponse({"ok": False, "error": "Invalid filename"}, status_code=400)
        p = os.path.join(AUDIO_DIR, safe)
        if os.path.isfile(p):
            os.remove(p)
            return {"ok": True}
        return JSONResponse({"ok": False, "error": "File not found"}, status_code=404)

    def _auto_restore_voices():
        import glob as _glob
        candidates = []
        for pat in [os.path.join(BASE_DIR, "voice_backup*.zip"),
                    os.path.join(DATA_DIR, "voice_backup*.zip")]:
            candidates.extend(_glob.glob(pat))
        if not candidates:
            return
        for c in candidates[:1]:
            try:
                with open(c, "rb") as f:
                    blob = f.read()
                res = vc.restore_backup_bytes(blob)
                print(f"[VOICE] Auto-restore from {c}: {res}")
            except Exception as e:
                print(f"[VOICE] Auto-restore failed ({c}): {e}")

    _auto_restore_voices()

    print("[STARTUP] Voice Clone Studio loaded")
except Exception as vc_err:
    print(f"[STARTUP] Voice Clone disabled: {vc_err}")

# ==========================================
# 6. STORY WRITER (novel / audiobook script generator)
# ==========================================
try:
    import story_writer as sw

    @app.get("/api/story/authors")
    def story_authors():
        return {"ok": True, "authors": sw.author_list(), "genres": sw.genre_list()}

    @app.post("/api/story/create")
    async def story_create(
        title: str = Form(...),
        author_id: str = Form(...),
        _: str = Depends(require_token),
        genre_id: str = Form(...),
        lang: str = Form("my"),
        num_chapters: int = Form(20),
        target_words: int = Form(600),
        sample_file: Optional[UploadFile] = File(None),
    ):
        if num_chapters > 200:
            return JSONResponse({"ok": False, "error": "Max 200 chapters"}, status_code=400)
        if title.strip() and len(title.strip()) < 2:
            return JSONResponse({"ok": False, "error": "Title က 2 လုံးထက်ရှည်ရမယ်"}, status_code=400)
        sample_excerpt = ""
        if sample_file and sample_file.filename:
            try:
                data = await sample_file.read()
                sample_excerpt = data.decode("utf-8", errors="ignore")
            except Exception as e:
                print(f"[STORY] sample read failed: {e}")
        try:
            story = sw.create_story(
                title=title, author_id=author_id, genre_id=genre_id,
                lang=lang, num_chapters=num_chapters,
                target_words=target_words, sample_excerpt=sample_excerpt,
            )
        except ValueError as ve:
            return JSONResponse({"ok": False, "error": str(ve)}, status_code=400)
        story_id = sw.start_writing(story)
        return {"ok": True, "story_id": story_id, "async": True}

    @app.get("/api/story/status/{story_id}")
    def story_status(story_id: str):
        s = sw.story_status(story_id)
        if not s:
            return JSONResponse({"ok": False, "error": "Story not found"}, status_code=404)
        return s

    @app.get("/api/story/library")
    def story_library():
        return {"ok": True, "items": sw.story_library()}

    @app.get("/api/story/download")
    def story_download(id: str = "", mode: str = "full"):
        safe_id = os.path.basename(id)
        path = os.path.join(sw.STORIES_DIR, f"{safe_id}_narration.txt") if mode == "narration" \
            else os.path.join(sw.STORIES_DIR, f"{safe_id}.txt")
        if not os.path.isfile(path):
            return JSONResponse({"ok": False, "error": "File not ready"}, status_code=404)
        ext = "_narration.txt" if mode == "narration" else ".txt"
        return FileResponse(path, filename=f"{safe_id}{ext}",
                            media_type="text/plain; charset=utf-8")

    @app.delete("/api/story/{story_id}")
    def story_delete(story_id: str, _: str = Depends(require_token)):
        sw.delete_story(os.path.basename(story_id))
        return {"ok": True}

    sw.resume_all()
    print("[STARTUP] Story Writer loaded")
except Exception as sw_err:
    import traceback as _sw_tb
    print(f"[STARTUP] Story Writer disabled: {sw_err}")
    _sw_tb.print_exc()

# ==========================================
# 7. THUMBNAIL GENERATOR
# ==========================================
try:
    import thumbnail_gen as thumb

    @app.post("/api/thumbnail/generate")
    async def thumbnail_generate(
        video_title: str = Form(...),
        hook_text: str = Form(""),
        language: str = Form("en"),
        video_id: str = Form(""),
        _: str = Depends(require_token),
    ):
        # If this video already has a saved recap script, feed it in so the
        # thumbnail concepts are grounded in the real story, not just the title.
        story_summary = ""
        if video_id:
            for fname in os.listdir(DOWNLOAD_DIR):
                if fname.startswith(video_id) and fname.endswith("_script.txt"):
                    try:
                        with open(os.path.join(DOWNLOAD_DIR, fname), "r", encoding="utf-8") as f:
                            story_summary += f.read() + " "
                    except Exception:
                        pass
        results = await asyncio.to_thread(
            thumb.generate_thumbnails, video_title, hook_text, language, video_id,
            story_summary.strip(),
        )
        return {"ok": True, "thumbnails": [
            {"url": f"/thumbnails/{r['filename']}", "title": r["title"], "prompt": r["prompt"]}
            for r in results
        ]}

    @app.get("/api/thumbnail/list")
    def thumbnail_list():
        files = sorted(os.listdir(THUMB_DIR), reverse=True)[:30]
        return [{"filename": f, "url": f"/thumbnails/{f}"} for f in files if f.endswith(('.jpg','.png'))]

    print("[STARTUP] Thumbnail Generator loaded")
except Exception as th_err:
    print(f"[STARTUP] Thumbnail Generator disabled: {th_err}")

# ==========================================
# 8. MOVIE FINDER EXTENSION (discovery + download + auto-recap)
#    Wrapped in try/except so the recap tool keeps working
#    even if the finder module ever fails to load.
# ==========================================
try:
    import movie_finder
    movie_finder.set_recap_pipeline(run_advanced_pipeline)
    app.include_router(movie_finder.router)
    movie_finder.init_movie_finder()
    print("[STARTUP] Movie Finder extension loaded")
except Exception as mf_err:
    import traceback
    print(f"[STARTUP] Movie Finder disabled: {mf_err}")
    traceback.print_exc()

# ==========================================
# 9. SHORTS/REELS GENERATOR (standalone - operates on a finished recap's
#    own output files, no extra Gemini-video/Whisper cost; see
#    shorts_reels.py's module docstring). Wrapped in try/except so the
#    recap tool keeps working even if this module ever fails to load.
# ==========================================
try:
    import shorts_reels
    app.include_router(shorts_reels.router)
    print("[STARTUP] Shorts/Reels generator loaded")
except Exception as sr_err:
    import traceback
    print(f"[STARTUP] Shorts/Reels generator disabled: {sr_err}")
    traceback.print_exc()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
