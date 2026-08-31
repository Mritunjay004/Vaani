"""Step-1 self test: TTS a phrase per configured language, STT it back, compare.

Usage:
  python -m vaani.tools.roundtrip
Exercises both APIs end to end without any recorded audio. A rough sanity
check, not an eval — the real eval harness (deliverable 4) uses recorded
caller utterances.
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from ..config import load_config
from ..speech.audio import chunk_pcm, read_pcm16
from ..speech.stt import stream_transcribe
from ..speech.tts import speak_to_file

PHRASES = {
    "hi-IN": "मुझे अगले मंगलवार को डॉक्टर मेहता के साथ appointment चाहिए",
    "mr-IN": "मला उद्या सकाळी अपॉइंटमेंट बुक करायची आहे",
    "ta-IN": "எனக்கு நாளை மருத்துவரிடம் நேரம் வேண்டும்",
    "en-IN": "I want to book an appointment with Doctor Iyer next Tuesday at three PM",
}


async def run(config_path: str | None) -> int:
    cfg = load_config(config_path)
    out_dir = Path("samples/roundtrip")
    out_dir.mkdir(parents=True, exist_ok=True)
    failures = 0

    for lang in cfg.languages:
        phrase = PHRASES.get(lang)
        if not phrase:
            print(f"[{lang}] no test phrase configured, skipping")
            continue
        wav = out_dir / f"{lang}.wav"
        speak_to_file(phrase, cfg.tts, lang, wav)
        pcm = read_pcm16(wav, cfg.stt.sample_rate)

        async def chunks(p=pcm):
            for c in chunk_pcm(p, cfg.stt.sample_rate, chunk_ms=100):
                yield c
                await asyncio.sleep(0.1)

        result = await stream_transcribe(chunks(), cfg.stt, language_code=lang)
        ok = bool(result.text.strip())
        failures += 0 if ok else 1
        status = "ok" if ok else "FAIL(empty)"
        print(f"[{lang}] {status}")
        print(f"   sent: {phrase}")
        print(f"   got : {result.text}")
    return failures


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    raise SystemExit(asyncio.run(run(args.config)))


if __name__ == "__main__":
    main()
