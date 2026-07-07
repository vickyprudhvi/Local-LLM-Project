# spec.md — Laptop personal AI assistant (v1)

## What this is

A voice assistant that runs on my Windows laptop. It hears me through the mic, answers out loud, can look through the webcam, remembers facts I tell it, and escalates hard questions to the Claude API. Local-first: everyday requests are handled by a local model through Ollama; only accuracy-critical or high-stakes requests go to the cloud.

This spec is for v1 only. Build exactly this. The "Out of scope" section lists things I know about and have deliberately postponed — do not build them even if they seem easy.

## Environment

- Windows 11, Python 3.11+, PowerShell
- Ollama installed and running at http://localhost:11434 with models pulled: qwen3:8b, qwen3:4b, moondream, all-minilm
- Project root: C:\dev\home-ai with a venv
- I am comfortable with Python but treat me as the manual tester for all audio/camera behavior

## Hard constraints

1. Build in the phase order below. STOP at the end of each phase and wait for me to run the manual test before continuing. Do not proceed past a failed checkpoint.
2. Commit to git at every passing checkpoint with the message given in the phase.
3. No secrets in source. API key and model IDs come from .env via python-dotenv.
4. No frameworks beyond the pip list below. No LangChain, no agent frameworks, no async rewrite.
5. Every Ollama and Anthropic call must have a timeout and raise_for_status / error handling that prints a readable message instead of a stack trace mid-conversation.
6. Keep assistant.py under ~120 lines. Logic lives in the modules.

## Dependencies

pip install: requests anthropic chromadb faster-whisper sounddevice soundfile opencv-python numpy pyttsx3 python-dotenv rich webrtcvad-wheels

## Configuration (.env)

```
ANTHROPIC_API_KEY=sk-...        # I will fill this in
CLAUDE_MODEL=claude-sonnet-5
LOCAL_MODEL=qwen3:30b-a3b       # validated on this laptop: ~10 tok/s, correct routing + math
LOCAL_MODEL_FAST=qwen3:8b       # smaller/faster fallback if voice latency feels bad after Phase 3
LOCAL_MODEL_TINY=qwen3:4b       # for the future LLM router / fact extractor (week 2+)
VISION_MODEL=moondream
EMBED_MODEL=all-minilm
WHISPER_MODEL=base.en
```

.gitignore must cover: venv/, memory/, snapshots/, .env, __pycache__/, *.pyc

## File layout

```
home-ai/
  .env  .gitignore  spec.md  system_prompt.txt
  brain.py          # ask_local, ask_claude, load_system_prompt, trim_history
  router.py         # route_intent -> RouteDecision(mode, tool, payload)
  ears.py           # record + transcribe (push-to-talk v1, VAD function stubbed for later)
  voice.py          # speak() via pyttsx3, module-level engine
  memory_store.py   # remember/recall on ChromaDB PersistentClient
  eyes.py           # snapshot, describe_local (moondream), describe_claude
  assistant.py      # main loop: input -> route -> dispatch -> speak
  plant_watcher.py  # standalone scheduled snapshot script
```

## Module requirements

### brain.py
- ask_local(prompt, history, system_prompt) -> (text, metrics dict). POST to /api/chat with LOCAL_MODEL, stream False, keep_alive "10m". Trim history to last 12 turns before sending. Return Ollama's prompt_eval_count / eval_count / eval_duration in metrics and log them with rich to the console at debug level.
- ask_claude(prompt, history, system_prompt, image_path=None) -> text. Uses anthropic SDK, model from CLAUDE_MODEL, max_tokens 2000. If image_path given, attach as base64 image content block BEFORE the text block. Pass trimmed history (last 6 turns) so escalations keep conversational context.
- Note: claude-sonnet-5 has adaptive thinking on by default and rejects non-default temperature/top_p — do not set sampling parameters.

### router.py
- Pure functions, no I/O. route_intent(text) returns RouteDecision.
- Precedence: manual overrides ("ask claude ...", "use local ...") > tool patterns > Claude domain patterns > default local.
- Tool patterns:
  - remember: `^(remember|note|save this)\s*:\s*(.+)$` (payload = captured fact)
  - recall: "what do you remember", "what do you know about me"
  - time: "what time is it", "date today"
  - look: ONLY these phrases: "look at this", "take a look", "what do you see", "check the camera", "take a picture". Do NOT match the bare word "look" (it false-triggers on "look up X").
- Claude domain patterns (case-insensitive word-boundary regex): money/investing/tax/mortgage/insurance terms, medical/legal terms, resume/salary/career terms, "should i", "think hard", "reason carefully".
- Include unit tests in tests/test_router.py covering: "look up mortgage rates" routes to claude (not camera), "remember: X" extracts X, override prefixes win over everything.

### ears.py
- listen_push_to_talk(seconds=6): press Enter, record via sounddevice int16 mono 16kHz, write wav with soundfile, transcribe with faster-whisper (device cpu, compute_type int8, language en). Force-evaluate the segments iterator with list() before joining.
- listen_vad_webrtc(): implement using webrtcvad-wheels, 30ms frames, stop after 900ms silence, 15s max. Convert captured bytes with np.frombuffer(raw, dtype="int16") — soundfile has no buffer_decode. Wire into the menu but mark experimental; push-to-talk is the default.

### voice.py
- speak(text) using pyttsx3 with a module-level engine (init once, not per call). Rate ~185.
- Leave a commented stub for a future speak_piper.

### memory_store.py
- ChromaDB PersistentClient(path="./memory"), collection "facts".
- Embeddings via Ollama /api/embed with EMBED_MODEL — pass explicit embeddings on add and query; do not rely on Chroma's default embedding function.
- remember(text, kind="fact", source="user", confidence=0.85) -> id: UUID4 ids, metadata {kind, source, confidence, created_at ISO UTC}.
- recall(query, n_results=3) -> list of {text, metadata, distance}.
- Memory writes are MANUAL ONLY in v1: the only write path is the "remember:" command through the router. Do not add automatic fact extraction.
- Recall is automatic: assistant.py prepends up to 3 relevant facts to every local/claude prompt, formatted as "Remembered facts that may be relevant:" block. If recall returns nothing, send the prompt unmodified.

### eyes.py
- snapshot(path="snapshots/latest.jpg", camera_index=0) using cv2.CAP_DSHOW on Windows. Create snapshots/ if missing. Raise a clear error if the camera can't be read.
- Resize helper: longest side <= 1280 for local, <= 1600 for Claude, JPEG quality 85, base64 encode.
- describe_local(path, prompt): POST /api/generate to moondream with images=[b64]. Moondream is single-image only — never send it two images in one request.
- describe_claude(path, ...): delegates to brain.ask_claude with the image. Used when the user's request mentions reading text, questions, documents, screens, or when they say "ask claude to look".

### assistant.py
- Loop: mode prompt [t=text, p=push-to-talk, q=quit] -> get user text -> route_intent -> dispatch:
  - tool/remember -> memory_store.remember, confirm out loud
  - tool/recall -> list stored facts
  - tool/time -> local datetime, no model call
  - tool/look -> snapshot + describe_local, speak result; if the utterance also matches the "read text/question" cases, use describe_claude instead
  - claude -> ask_claude with memory-enriched prompt
  - local -> ask_local with memory-enriched prompt
- Append user/assistant turns to history after each exchange. Print routing decision (mode + tool) in dim text each turn so I can audit routing.
- Speak every answer with voice.speak.

### plant_watcher.py
- Standalone script, own process. Every 30 minutes: snapshot to snapshots/plant_YYYY-MM-DD_HH-MM-SS.jpg.
- Once per day: take yesterday's first and last photo, run describe_local on EACH separately, then ask the local text model to compare the two descriptions and summarize changes. (Moondream cannot compare two images in one call.) Print the summary and append it to snapshots/plant_log.txt.

### system_prompt.txt
Short and stable. Personality: concise, practical, direct; says so when uncertain; never invents facts. Memory rules: use recalled facts naturally, don't recite them. Prefers local execution; recommends escalation for money/health/legal/career/accuracy-critical image reading.

## Build phases and checkpoints

Phase 1 — scaffolding + brain.py + text-only assistant.py loop.
Checkpoint: I type a message, get a local model reply, history works across turns. Commit "text loop".

Phase 2 — router.py + tests.
Checkpoint: pytest passes; routing decisions print per turn; "look up mortgage rates" goes to claude. Commit "router".

Phase 3 — ears.py + voice.py wired in.
Checkpoint: push-to-talk -> spoken answer, end to end. I test mic/speaker manually. Commit "voice loop".

Phase 4 — memory_store.py wired in.
Checkpoint: "remember: dentist on July 14" then restart script then "what do you remember" returns it. Commit "memory".

Phase 5 — Claude escalation live.
Checkpoint: "think hard: <question>" hits the API and answers; normal chat stays local; routing line confirms. Commit "claude routing".

Phase 6 — eyes.py + camera commands.
Checkpoint: "what do you see" describes the room; holding printed text up and saying "look at this question, ask claude" gets an accurate spoken answer. Commit "vision".

Phase 7 — plant_watcher.py + system_prompt.txt polish.
Checkpoint: watcher takes a snapshot on launch and the daily-compare path runs when pointed at two existing photos (test with a --compare-now flag). Commit "plant watcher".

## Out of scope for v1 (do not build)

Wake words. Streaming/interruptible audio. VAD as default input. Automatic memory extraction (manual "remember:" only). LLM-based routing (regex only; log decisions so the LLM router can be added in week 2). Piper/Kokoro TTS (stub only). Multi-agent architecture. Gmail/Calendar/home-automation tools. Any GUI. Pi deployment.

## Success criteria for the weekend

By Sunday night: I can talk to it out loud, it answers out loud with reasonable latency, it saves and recalls facts across restarts, everyday chat never touches the API, "think hard" questions and text-reading camera requests reliably do, and every routing decision is visible on screen.
