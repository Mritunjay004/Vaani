"""WAV helpers for the local/simulator path. Telephony transports do their own framing."""
from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path


def read_pcm16(path: str | Path, target_rate: int) -> bytes:
    """Return mono s16le PCM at target_rate from any audio file.

    Uses the file as-is when it already matches; otherwise shells out to ffmpeg.
    """
    path = Path(path)
    try:
        with wave.open(str(path), "rb") as w:
            if (
                w.getnchannels() == 1
                and w.getsampwidth() == 2
                and w.getframerate() == target_rate
            ):
                return w.readframes(w.getnframes())
    except (wave.Error, EOFError):
        pass  # not a plain wav — fall through to ffmpeg

    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            f"{path} is not mono/16-bit/{target_rate}Hz WAV and ffmpeg is not installed "
            "to convert it. Install ffmpeg or provide a matching WAV."
        )
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(target_rate),
         "-f", "s16le", "-"],
        check=True,
        capture_output=True,
    )
    return out.stdout


def write_wav(path: str | Path, pcm: bytes, rate: int) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)


def chunk_pcm(pcm: bytes, rate: int, chunk_ms: int = 100):
    """Yield chunk_ms-sized frames, the way a live call delivers audio."""
    step = rate * 2 * chunk_ms // 1000
    for i in range(0, len(pcm), step):
        yield pcm[i : i + step]
