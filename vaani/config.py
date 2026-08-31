"""Load and validate a vertical config. The config is the product; code is plumbing."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import certifi
import yaml
from dotenv import load_dotenv

load_dotenv()  # picks up SARVAM_API_KEY etc. from a local .env

# python.org macOS builds ship no system CA certs, which breaks the raw
# `websockets` connections the SDK's streaming clients make (REST is fine —
# httpx bundles certifi). Point stdlib ssl at certifi unless already set.
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "clinic_intake.yaml"


@dataclass
class SttConfig:
    model: str
    batch_model: str
    mode: str
    stream_type: str
    sample_rate: int
    encoding: str
    vad: dict[str, Any] = field(default_factory=dict)


@dataclass
class TtsConfig:
    model: str
    speaker: str
    pace: float
    sample_rate: int
    output_audio_codec: str
    dict_id: str | None = None


@dataclass
class LlmConfig:
    model: str
    endpoint: str
    temperature: float
    max_tokens: int


@dataclass
class VerticalConfig:
    vertical: str
    languages: list[str]
    fallback_language: str
    language_gap_behavior: str
    stt: SttConfig
    tts: TtsConfig
    llm: LlmConfig
    clinic: dict[str, Any]
    dialogue: dict[str, Any]
    tools: list[str]
    raw: dict[str, Any] = field(default_factory=dict)


def load_config(path: str | Path | None = None) -> VerticalConfig:
    path = Path(path or os.environ.get("VAANI_CONFIG", DEFAULT_CONFIG))
    data = yaml.safe_load(path.read_text())

    cfg = VerticalConfig(
        vertical=data["vertical"],
        languages=list(data["languages"]),
        fallback_language=data["fallback_language"],
        language_gap_behavior=data.get("language_gap_behavior", "fallback"),
        stt=SttConfig(**data["stt"]),
        tts=TtsConfig(**data["tts"]),
        llm=LlmConfig(**data["llm"]),
        clinic=data.get("clinic", {}),
        dialogue=data.get("dialogue", {}),
        tools=list(data.get("tools", [])),
        raw=data,
    )
    if cfg.fallback_language not in cfg.languages:
        raise ValueError(
            f"fallback_language {cfg.fallback_language!r} must be in languages {cfg.languages}"
        )
    return cfg


def api_key() -> str:
    key = os.environ.get("SARVAM_API_KEY", "")
    if not key:
        raise RuntimeError("SARVAM_API_KEY is not set. Put it in .env or export it.")
    return key
