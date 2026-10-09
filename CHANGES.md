# Fixes — Recap Master Pro

## 1. STORY tab — root cause found and fixed

`story_writer.py` **did not exist anywhere in your Space's files.**
`main.py` has:

    try:
        import story_writer as sw
        ...
    except Exception as sw_err:
        print(f"[STARTUP] Story Writer disabled: {sw_err}")

Since the file was missing, that import always failed, was silently
caught, and none of the `/api/story/*` endpoints were ever registered.
Every click on the STORY tab was hitting a 404 — the feature was never
actually running.

**Fix:** `story_writer.py` in this zip is a full, working
implementation (author/genre catalog, chapter-by-chapter generation
via Gemini with continuity between chapters, background-thread
progress you can poll, library listing, download as full text or
narration-ready text, delete). Upload it to your Space's root folder
(same level as `main.py`).

Note: I could not read the exact JSON field names your `index.html`
JavaScript expects back from `/api/story/status/...` and
`/api/story/library` (the file is 112 KB and the relevant
`swCreate`/`loadStoryAuthors`/`loadStoryLibrary` functions were past
what I could pull). I matched the field names main.py already uses
elsewhere (`status`, `current`, `total`, `message` — same shape as
`/api/voice/status/{task_id}`). If the STORY tab still doesn't render
correctly after uploading this, tell me and paste the relevant chunk
of `index.html`'s `<script>` (search for `swCreate`) and I'll match
the field names exactly.

## 2. VOICE clone — bug found and fixed

In `voice_clone.py`, `F5_TTS_SPACES` contained:

    "https://BMCVRN-E2-F5-TTS.hf.space"

Hugging Face Space subdomains are **always lowercase** — this URL
could never resolve, so that fallback space silently failed every
time. Fixed to `https://bmcvrn-e2-f5-tts.hf.space`.

I also checked the VoxCPM2 (Burmese-capable) integration line by line
against the live `openbmb/VoxCPM-Demo` app.py — the input order and
`api_name="generate"` your code sends match exactly, so that call
path itself is correctly wired.

**Important usage note (not a bug, but likely why clones "don't
work" in practice):** F5-TTS only supports English/Chinese target
text — if you send Burmese text to an `f5-tts` voice profile it will
fail or produce garbage. **For Burmese cloning always create the
voice with provider `voxcpm2`**, not `f5-tts`. I added a clearer
error message for this case.

Also: both F5-TTS and VoxCPM2 call free public community GPU Spaces
that sleep when idle — the first call after idle time can take
30–90s to "cold start" before it succeeds (the code already retries
for this, just be patient on the first request).

Upload the patched `voice_clone.py` to your Space's root folder,
overwriting the old one.

## 3. Two smaller fixes worth making in `main.py` (from the earlier review)

I didn't regenerate the full 76 KB `main.py` (too large / risk of a
paste error breaking your app), so apply these two small edits
yourself — both are short and safe:

**a) `/api/upload_chunk` and `/api/upload_finalize` have no
`APP_TOKEN` auth check**, unlike every other write endpoint. If you've
set `APP_TOKEN` as a secret to protect your Space, add this parameter
to both functions (same pattern already used on `/api/process`):

    _: str = Depends(require_token),

**b) Disk cleanup only runs once, at Space startup** — `_cleanup_staging()`
and `_cleanup_media_dir()` are called once at import time and never
again, so a long-running Space (no restart) slowly fills its disk.
Add this near the top of `start_processing()` (the `/api/process`
handler), right next to the existing `_prune_old_logs()` call:

    _cleanup_staging()
    _cleanup_media_dir(THUMB_DIR, "thumbnail")
    _cleanup_media_dir(PREVIEW_DIR, "preview")

## What to upload

Just these two files (overwrite existing ones in the Space repo):
- `story_writer.py` (new file)
- `voice_clone.py` (replaces the old one)

Then apply the two small `main.py` edits above by hand (or tell me to
paste your current `main.py` back and I'll edit it directly).
