"""Synthesize speech to a WAV file through Bulbul.

Usage:
  python -m vaani.tools.tts_say "आपका appointment मंगलवार को 3 बजे है" --lang hi-IN -o out.wav
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import time

from ..config import load_config
from ..speech.audio import write_wav
from ..speech.tts import speak_to_file, stream_speak


async def _run_stream(text: str, lang: str, cfg, out: str) -> None:
    # raw PCM over the WS so we can measure time-to-first-audio; wrap as wav locally
    stream_cfg = dataclasses.replace(cfg.tts, output_audio_codec="linear16")

    async def one_chunk():
        yield text

    pcm = b""
    t0 = time.monotonic()
    first_at = None
    async for audio in stream_speak(one_chunk(), stream_cfg, lang):
        if first_at is None:
            first_at = time.monotonic() - t0
        pcm += audio
    write_wav(out, pcm, cfg.tts.sample_rate)
    print(f"wrote {out} ({len(pcm)} bytes PCM, first audio in {first_at:.2f}s)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("text")
    ap.add_argument("--lang", default=None, help="BCP-47 code (default: config fallback_language)")
    ap.add_argument("-o", "--out", default="out.wav")
    ap.add_argument("--stream", action="store_true", help="use the WebSocket streaming path")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    lang = args.lang or cfg.fallback_language
    if lang not in cfg.languages:
        raise SystemExit(f"{lang} not in configured languages {cfg.languages}")
    if args.stream:
        asyncio.run(_run_stream(args.text, lang, cfg, args.out))
    else:
        path = speak_to_file(args.text, cfg.tts, lang, args.out)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
