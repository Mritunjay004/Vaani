"""Transcribe a local audio file through Sarvam STT.

Usage:
  python -m vaani.tools.stt_file samples/hi_booking.wav                 # realtime streaming (live-path parity)
  python -m vaani.tools.stt_file samples/hi_booking.wav --batch         # batch API
  python -m vaani.tools.stt_file samples/hi_booking.wav --lang hi-IN
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time

from ..config import load_config
from ..speech.audio import chunk_pcm, read_pcm16
from ..speech.stt import batch_transcribe, stream_transcribe


async def _run_stream(path: str, lang: str, cfg) -> None:
    pcm = read_pcm16(path, cfg.stt.sample_rate)

    async def chunks():
        # pace chunks at real time so VAD/endpointing behaves like a live call
        for chunk in chunk_pcm(pcm, cfg.stt.sample_rate, chunk_ms=100):
            yield chunk
            await asyncio.sleep(0.1)

    t0 = time.monotonic()
    result = await stream_transcribe(
        chunks(), cfg.stt, language_code=lang,
        on_partial=lambda text: print(f"  partial: {text}", flush=True),
    )
    elapsed = time.monotonic() - t0
    for f in result.finals:
        print(json.dumps({
            "text": f.text,
            "language": f.language,
            "language_confidence": f.language_confidence,
            "final_latency_ms": round(f.latency_ms) if f.latency_ms else None,
        }, ensure_ascii=False))
    print(f"[stream done in {elapsed:.2f}s, {len(result.finals)} final(s)]")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("audio", help="path to audio file (wav preferred)")
    ap.add_argument("--lang", default="auto", help="BCP-47 code or 'auto'")
    ap.add_argument("--batch", action="store_true", help="use batch API instead of realtime WS")
    ap.add_argument("--config", default=None, help="vertical config path")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.batch:
        t = batch_transcribe(args.audio, cfg.stt, None if args.lang == "auto" else args.lang)
        print(json.dumps({"text": t.text, "language": t.language}, ensure_ascii=False))
    else:
        asyncio.run(_run_stream(args.audio, args.lang, cfg))


if __name__ == "__main__":
    main()
