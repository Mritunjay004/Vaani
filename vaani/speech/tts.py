"""Sarvam Bulbul TTS: WebSocket streaming (live path) and one-shot convert.

Barge-in note: Bulbul's WS has no server-side cancel. The live pipeline cuts
playback by draining its local buffer and closing this socket; a fresh socket
is opened for the next agent utterance. That policy lives in the orchestrator,
not here — this module just streams.
"""
from __future__ import annotations

import base64
from pathlib import Path
from typing import AsyncIterator

from sarvamai import AsyncSarvamAI, SarvamAI

from ..config import TtsConfig, api_key


async def stream_speak(
    text_chunks: AsyncIterator[str],
    cfg: TtsConfig,
    language_code: str,
) -> AsyncIterator[bytes]:
    """Stream text in, yield audio bytes out (codec per cfg.output_audio_codec)."""
    import asyncio

    client = AsyncSarvamAI(api_subscription_key=api_key())
    async with client.text_to_speech_streaming.connect(
        model=cfg.model, send_completion_event="true"
    ) as ws:
        await ws.configure(
            target_language_code=language_code,
            speaker=cfg.speaker,
            pace=cfg.pace,
            speech_sample_rate=cfg.sample_rate,
            output_audio_codec=cfg.output_audio_codec,
            dict_id=cfg.dict_id,
        )

        async def send() -> None:
            async for chunk in text_chunks:
                await ws.convert(chunk)
            await ws.flush()

        send_task = asyncio.create_task(send())
        try:
            while True:
                message = await ws.recv()
                mtype = getattr(message, "type", None)
                if mtype == "audio":
                    yield base64.b64decode(message.data.audio)
                elif mtype == "event" and message.data.event_type == "final":
                    break
                elif mtype == "error":
                    raise RuntimeError(f"TTS error: {getattr(message, 'data', message)}")
        finally:
            send_task.cancel()


def speak_to_file(text: str, cfg: TtsConfig, language_code: str, out_path: str | Path) -> Path:
    """One-shot synthesis to a WAV file (dev/simulator convenience)."""
    from sarvamai.play import save

    client = SarvamAI(api_subscription_key=api_key())
    kwargs: dict = {}
    if cfg.dict_id:
        kwargs["dict_id"] = cfg.dict_id
    audio = client.text_to_speech.convert(
        text=text,
        model=cfg.model,
        speaker=cfg.speaker,
        language_code=language_code,
        pace=cfg.pace,
        **kwargs,
    )
    out_path = Path(out_path)
    save(audio, str(out_path))
    return out_path
