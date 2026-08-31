# Vaani — phone-callable voice agent for clinic appointment intake

A caller dials a normal phone number, speaks in Hindi/Marathi/Tamil/English
(code-mixed is fine), and books, reschedules, or cancels a clinic appointment —
no keypad, spoken confirmation read back.

Built on Sarvam AI end to end: **Saaras v3** (realtime STT),
**sarvam-105b-conversations** (dialogue LLM), **Bulbul v3** (streaming TTS).
Vertical behavior lives in [config/clinic_intake.yaml](config/clinic_intake.yaml) —
a new vertical is a new config file, not a fork.

## Status

- [x] **Step 1** — Sarvam STT + TTS standalone against local audio files
- [ ] Step 2 — text-only LLM booking loop
- [ ] Step 3 — composed voice loop over the local simulator
- [ ] Step 4 — LiveKit orchestration (VAD, barge-in)
- [ ] Step 5 — Exotel telephony
- [ ] Step 6 — analytics, eval harness, cost model

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env        # then paste your SARVAM_API_KEY
```

`ffmpeg` on PATH is optional but recommended (converts arbitrary audio to the
16kHz mono PCM the streaming API wants).

## Step 1: standalone speech tools

```bash
source .venv/bin/activate

# TTS → wav (REST one-shot)
python -m vaani.tools.tts_say "आपका appointment मंगलवार को तीन बजे है" --lang hi-IN -o out.wav

# TTS over the streaming WebSocket (reports time-to-first-audio)
python -m vaani.tools.tts_say "Appointment confirm हो गया" --lang hi-IN --stream -o out.wav

# STT a local file through the realtime WebSocket, paced like a live call
# (prints partials as they arrive, then the final with latency)
python -m vaani.tools.stt_file out.wav --lang hi-IN
python -m vaani.tools.stt_file out.wav            # --lang auto: language detection

# Batch STT (the post-call analytics path)
python -m vaani.tools.stt_file out.wav --batch

# Self test: TTS a phrase in every configured language, STT it back
python -m vaani.tools.roundtrip
```

## Configuration

Everything vertical-specific — languages, voice, clinic details, dialogue
states, retry limits, tools — is in `config/clinic_intake.yaml`. Point
`VAANI_CONFIG` (or `--config`) at a different file to change verticals.
`fallback_language` handles the Saaras(23 langs)-vs-Bulbul(11 langs) gap:
a caller we understand but can't speak back to gets an apology plus the
conversation in the fallback language (`language_gap_behavior: fallback`),
or a human handoff (`handoff`).

## Placing a test call

Arrives with step 5 (Exotel). Until then, the simulator (step 3) takes a WAV
and returns the full agent turn.
