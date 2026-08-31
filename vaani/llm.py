"""sarvam-105b-conversations wrapper: understand (slot extraction) and render (phrasing).

The FSM owns control flow. The LLM does exactly two jobs per turn:
  understand() — pull structured fields out of the caller's utterance,
                 forced through a tool call so output is always valid JSON
  render()     — phrase a speech act naturally in the session language
"""
from __future__ import annotations

import json
from dataclasses import dataclass, fields
from typing import Any

from sarvamai import SarvamAI

from .config import LlmConfig, api_key

_EXTRACT_TOOL = {
    "type": "function",
    "function": {
        "name": "record_understanding",
        "description": "Record what the caller communicated in this utterance.",
        "parameters": {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "enum": ["book", "reschedule", "cancel", "other"],
                    "description": "Only when the caller expresses what they want done.",
                },
                "name": {"type": "string", "description": "Caller's name if stated."},
                "phone_raw": {
                    "type": "string",
                    "description": "Phone number exactly as spoken, verbatim, including digit words.",
                },
                "doctor": {
                    "type": "string",
                    "description": "Doctor name or department mentioned, verbatim.",
                },
                "date_phrase": {
                    "type": "string",
                    "description": "The date expression verbatim in the original language, e.g. 'अगले मंगलवार', 'परसों', '15/9'. Do NOT resolve it to a date.",
                },
                "time_phrase": {
                    "type": "string",
                    "description": "The time expression verbatim, e.g. 'साढ़े तीन बजे', '3 pm'. Do NOT convert it.",
                },
                "choice_index": {
                    "type": "integer",
                    "description": "1-based index if the caller picked one of the options just offered.",
                },
                "decision": {
                    "type": "string",
                    "enum": ["yes", "no", "unclear"],
                    "description": "Only when the caller was asked to confirm something.",
                },
                "wants_human": {"type": "boolean"},
                "wants_restart": {"type": "boolean"},
            },
            "additionalProperties": False,
        },
    },
}


@dataclass
class Understanding:
    intent: str | None = None
    name: str | None = None
    phone_raw: str | None = None
    doctor: str | None = None
    date_phrase: str | None = None
    time_phrase: str | None = None
    choice_index: int | None = None
    decision: str | None = None
    wants_human: bool = False
    wants_restart: bool = False


class Llm:
    def __init__(self, cfg: LlmConfig):
        self.cfg = cfg
        self.client = SarvamAI(api_subscription_key=api_key())

    def understand(self, utterance: str, state_hint: str,
                   history: list[dict[str, str]] | None = None) -> Understanding:
        system = (
            "You extract structured information from a clinic-booking phone call. "
            "Callers speak Hindi, Marathi, Tamil, or English, often code-mixed. "
            "Record ONLY what this utterance actually communicates — omit every "
            "field the caller did not address. Copy date/time/phone expressions "
            "verbatim; never resolve or translate them.\n"
            f"Conversation context: {state_hint}"
        )
        messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        messages += (history or [])[-6:]
        messages.append({"role": "user", "content": utterance})
        resp = self.client.chat.completions(
            model=self.cfg.model,
            messages=messages,
            temperature=0.0,
            max_tokens=self.cfg.max_tokens,
            tools=[_EXTRACT_TOOL],
            tool_choice="required",
        )
        msg = resp.choices[0].message
        data = self._tool_args(msg)
        allowed = {f.name for f in fields(Understanding)}
        return Understanding(**{k: v for k, v in (data or {}).items()
                                if k in allowed and v not in (None, "")})

    @staticmethod
    def _tool_args(msg) -> dict[str, Any] | None:
        """Extract tool-call arguments from either response shape.

        sarvam-105b sometimes emits the forced call as raw JSON in `content`
        ('[{"name": ..., "parameters": {...}}]' plus trailing junk) instead of
        populating `tool_calls` — handle both.
        """
        calls = getattr(msg, "tool_calls", None) or []
        if calls:
            try:
                return json.loads(calls[0].function.arguments)
            except (json.JSONDecodeError, AttributeError):
                return None
        content = (msg.content or "").strip()
        start = content.find("{")
        if start == -1:
            return None
        try:
            obj, _ = json.JSONDecoder().raw_decode(content[start:])
        except json.JSONDecodeError:
            return None
        if isinstance(obj, dict):
            return obj.get("parameters") or obj.get("arguments") or obj
        return None

    def render(self, instruction: str, language_code: str,
               history: list[dict[str, str]] | None = None) -> str:
        system = (
            "You are the voice of a clinic's phone assistant. Reply with ONLY the "
            f"next thing to say aloud, in the language with BCP-47 code {language_code} "
            "(natural code-mixing with English is fine). One or two short spoken "
            "sentences — no lists, no formatting, no translations in brackets. "
            "Keep every date, time, and number VALUE exactly as given in the "
            "instruction, but say weekday and month names in the reply language "
            "(e.g. 'Wednesday, 2 September' becomes 'बुधवार, 2 सितंबर' in Hindi). "
            "The [instruction] decides WHAT to say — follow "
            "it exactly even if the conversation seems to suggest otherwise; "
            "you only choose the wording."
        )
        messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        messages += (history or [])[-6:]
        messages.append({"role": "user", "content": f"[instruction] {instruction}"})
        resp = self.client.chat.completions(
            model=self.cfg.model,
            messages=messages,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()
