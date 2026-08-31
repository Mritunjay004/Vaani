"""Sarvam Saaras STT: realtime streaming (live path) and batch (analytics path)."""
from __future__ import annotations

import asyncio
import base64
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator

from sarvamai import AsyncSarvamAI, RealtimeAudioInput, RealtimeEnd, SarvamAI

from ..config import SttConfig, api_key


@dataclass
class Transcript:
    text: str
    language: str | None = None
    language_confidence: float | None = None
    is_final: bool = True
    latency_ms: float | None = None  # last-audio-sent -> final-transcript


@dataclass
class SttStreamResult:
    finals: list[Transcript] = field(default_factory=list)
    partials: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(t.text for t in self.finals).strip()


async def stream_transcribe(
    audio_chunks: AsyncIterator[bytes],
    cfg: SttConfig,
    language_code: str = "auto",
    on_partial=None,
) -> SttStreamResult:
    """Stream PCM chunks to the Saaras realtime WS; collect finals.

    audio_chunks yields raw PCM matching cfg.encoding/cfg.sample_rate.
    """
    client = AsyncSarvamAI(api_subscription_key=api_key())
    result = SttStreamResult()
    last_audio_at: float | None = None

    vad = {k: str(v) for k, v in (cfg.vad or {}).items()}
    async with client.speech_to_text_realtime_streaming.connect(
        model=cfg.model,
        language_code=language_code,
        mode=cfg.mode,
        stream_type=cfg.stream_type,
        sample_rate=str(cfg.sample_rate),
        encoding=cfg.encoding,
        **vad,
    ) as ws:

        async def send() -> None:
            nonlocal last_audio_at
            async for chunk in audio_chunks:
                await ws.send_realtime_audio_input(
                    RealtimeAudioInput(audio=base64.b64encode(chunk).decode())
                )
            last_audio_at = time.monotonic()
            await ws.send_realtime_end(RealtimeEnd())

        async def recv() -> None:
            async for msg in ws:
                event = getattr(msg, "event", None)
                if event == "transcript.partial":
                    result.partials.append(msg.text)
                    if on_partial:
                        on_partial(msg.text)
                elif event == "transcript.final":
                    latency = (
                        (time.monotonic() - last_audio_at) * 1000
                        if last_audio_at
                        else None
                    )
                    result.finals.append(
                        Transcript(
                            text=msg.text,
                            language=getattr(msg, "language", None),
                            language_confidence=getattr(msg, "language_confidence", None),
                            latency_ms=latency,
                        )
                    )
                elif event == "error":
                    code = getattr(msg, "code", "?")
                    detail = getattr(msg, "message", "")
                    if getattr(msg, "is_fatal", True):
                        raise RuntimeError(f"STT fatal error {code}: {detail}")
                elif event == "session.end":
                    return

        send_task = asyncio.create_task(send())
        try:
            await recv()
        finally:
            send_task.cancel()
    return result


def batch_transcribe(path: str | Path, cfg: SttConfig, language_code: str | None = None) -> Transcript:
    """Post-call / eval transcription of a recorded file via saaras batch model."""
    client = SarvamAI(api_subscription_key=api_key())
    kwargs = {"language_code": language_code} if language_code else {}
    with open(path, "rb") as f:
        resp = client.speech_to_text.transcribe(
            file=f, model=cfg.batch_model, mode=cfg.mode, **kwargs
        )
    return Transcript(
        text=resp.transcript,
        language=getattr(resp, "language_code", None),
    )
