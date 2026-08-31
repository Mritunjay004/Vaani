"""Explicit dialogue state machine for clinic intake.

Control flow lives here, not in the model. Per caller turn:
  1. llm.understand() extracts fields (verbatim spans)
  2. normalize resolves dates/times/phones deterministically
  3. this machine updates slots, runs store tools, picks the next state
  4. llm.render() phrases the machine's instruction in the session language

Slots fill opportunistically — a caller who says name, doctor, and date in
one breath skips ahead — but transitions and retries are always the machine's
decision. Two failed attempts on the same slot → HANDOFF. Never loop.
"""
from __future__ import annotations

import datetime as dt
import enum
import time
from dataclasses import dataclass, field
from typing import Any

from ..clinic.store import ClinicStore, SlotTakenError
from ..llm import Llm, Understanding
from ..nlu.normalize import (
    parse_date,
    parse_phone,
    parse_time,
    speakable_date,
    speakable_phone,
    speakable_time,
)


class State(enum.Enum):
    GREET = "GREET"
    IDENTIFY = "IDENTIFY"
    INTENT = "INTENT"
    COLLECT_BOOKING = "COLLECT_BOOKING"
    FIND_APPOINTMENT = "FIND_APPOINTMENT"
    CONFIRM_TARGET = "CONFIRM_TARGET"
    OFFER_SLOTS = "OFFER_SLOTS"
    CONFIRM = "CONFIRM"
    COMMIT = "COMMIT"
    CLOSE = "CLOSE"
    HANDOFF = "HANDOFF"


@dataclass
class Turn:
    role: str
    text: str
    state: str
    at: float = field(default_factory=time.time)
    understanding: dict[str, Any] | None = None


class DialogueSession:
    def __init__(self, cfg, store: ClinicStore, llm: Llm,
                 language_code: str | None = None, caller_phone: str | None = None,
                 today: dt.date | None = None):
        self.cfg = cfg
        self.store = store
        self.llm = llm
        self.language = language_code or cfg.fallback_language
        self.today = today or dt.date.today()
        self.state = State.GREET
        self.done = False
        self.slots: dict[str, Any] = {
            "name": None, "phone": caller_phone, "intent": None,
            "doctor": None, "date": None, "time": None,
        }
        self.target = None          # existing appointment being changed/cancelled
        self.offered: list[dt.time] = []
        self.offered_date: dt.date | None = None
        self.retries: dict[str, int] = {}
        self.asked: str | None = None   # slot named in our last question
        self.max_retries = int(cfg.dialogue.get("max_slot_retries", 2))
        self.history: list[dict[str, str]] = []   # for LLM context
        self.log: list[Turn] = []                  # full call log
        self.result: dict[str, Any] | None = None  # committed outcome

    # ------------------------------------------------------------- helpers

    def _say(self, instruction: str) -> str:
        text = self.llm.render(instruction, self.language, self.history)
        self.history.append({"role": "assistant", "content": text})
        self.log.append(Turn("agent", text, self.state.value))
        return text

    def _goto(self, state: State) -> None:
        self.log.append(Turn("system", f"{self.state.value} -> {state.value}", state.value))
        self.state = state

    def _fail(self, slot: str) -> bool:
        """Count a failed attempt; True after max_retries failures on the slot."""
        self.retries[slot] = self.retries.get(slot, 0) + 1
        return self.retries[slot] >= self.max_retries

    def _doctor_name(self) -> str:
        return self.store.doctor_name(self.slots["doctor"]["id"])

    def _details(self) -> str:
        return (f"patient {self.slots['name']}, {self._doctor_name()}, "
                f"{speakable_date(self.slots['date'], self.today)} at "
                f"{speakable_time(self.slots['time'])}")

    # ------------------------------------------------------------ absorb

    def _absorb(self, u: Understanding) -> list[str]:
        """Fill any slot the utterance addressed; return fields that failed to parse."""
        bad: list[str] = []
        if u.intent and not self.slots["intent"]:
            self.slots["intent"] = u.intent
        if u.name and not self.slots["name"]:
            self.slots["name"] = u.name.strip()
        if u.phone_raw:
            phone = parse_phone(u.phone_raw)
            if phone:
                self.slots["phone"] = phone
            else:
                bad.append("phone")
        if u.doctor:
            doc = self.store.resolve_doctor(u.doctor)
            if doc:
                self.slots["doctor"] = doc
            else:
                bad.append("doctor")
        if u.date_phrase:
            date = parse_date(u.date_phrase, self.language, self.today)
            if date:
                self.slots["date"] = date
            else:
                bad.append("date")
        if u.time_phrase:
            hours = self.cfg.clinic.get("hours", {})
            t = parse_time(u.time_phrase, self.language,
                           int(hours.get("open", "09:00")[:2]),
                           int(hours.get("close", "18:00")[:2]))
            if t:
                self.slots["time"] = t
            else:
                bad.append("time")
        return bad

    # -------------------------------------------------------------- entry

    def open(self) -> str:
        """Agent speaks first."""
        clinic = self.cfg.clinic.get("name", "the clinic")
        return self._say(
            f"Greet the caller warmly on behalf of {clinic} and ask how you can "
            "help with their appointment."
        )

    def step(self, user_text: str) -> str:
        """One caller turn in, one agent reply out."""
        hint = self._state_hint()
        u = self.llm.understand(user_text, hint, self.history)
        self.history.append({"role": "user", "content": user_text})
        self.log.append(Turn("caller", user_text, self.state.value,
                             understanding=u.__dict__))

        if u.wants_human:
            return self._handoff("caller asked for a person")
        if u.wants_restart:
            self.slots.update({"intent": None, "doctor": None, "date": None, "time": None})
            self.target = None
            self.asked = None
            self._goto(State.INTENT)
            return self._say("Acknowledge starting over and ask what they would like to do.")

        bad = self._absorb(u)
        return self._advance(u, bad)

    # ------------------------------------------------------------ core FSM

    def _advance(self, u: Understanding, bad: list[str]) -> str:
        s = self.state

        if s in (State.GREET, State.IDENTIFY, State.INTENT, State.COLLECT_BOOKING,
                 State.FIND_APPOINTMENT):
            return self._collect(u, bad)
        if s == State.CONFIRM_TARGET:
            return self._confirm_target(u)
        if s == State.OFFER_SLOTS:
            return self._pick_slot(u, bad)
        if s == State.CONFIRM:
            return self._confirm(u, bad)
        # CLOSE / HANDOFF: conversation is over
        self.done = True
        return self._say("Politely say goodbye.")

    def _collect(self, u: Understanding, bad: list[str]) -> str:
        """Needs-driven collection: ask for the first missing thing."""
        for slot in ("name", "phone", "intent"):
            if not self.slots[slot]:
                # a failed attempt = we asked for this slot (or the caller tried
                # and it didn't parse) and it is still empty
                if (self.asked == slot or slot in bad) and self._fail(slot):
                    return self._handoff(f"could not get {slot}")
                self.asked = slot
                self._goto(State.IDENTIFY if slot in ("name", "phone") else State.INTENT)
                asks = {
                    "name": "Ask for the caller's name.",
                    "phone": "Ask for their 10-digit mobile number.",
                    "intent": "Ask whether they want to book, reschedule, or cancel an appointment.",
                }
                prefix = self._bad_prefix(bad)
                return self._say(prefix + asks[slot])

        if self.slots["intent"] in ("reschedule", "cancel") and not self.target:
            return self._find_appointment()

        # booking path (also reschedule once target is confirmed)
        self._goto(State.COLLECT_BOOKING)
        for slot, ask in (
            ("doctor", "Ask which doctor or department they want to see. Options: "
                       + ", ".join(f"{d['name']} ({d['department']})" for d in self.store.doctors())),
            ("date", "Ask what day they would like to come in."),
            ("time", "Ask what time of day suits them."),
        ):
            if self.slots["intent"] == "cancel":
                break
            if not self.slots[slot]:
                if (self.asked == slot or slot in bad) and self._fail(slot):
                    return self._handoff(f"could not understand {slot}")
                self.asked = slot
                return self._say(self._bad_prefix(bad) + ask)
        return self._offer_slots()

    def _find_appointment(self) -> str:
        self._goto(State.FIND_APPOINTMENT)
        appts = self.store.find_appointments(self.slots["phone"])
        if not appts:
            if self._fail("find_appointment"):
                return self._handoff("no appointment found for this number")
            self._goto(State.INTENT)
            self.slots["intent"] = None
            return self._say(
                f"Tell them no upcoming appointment was found for "
                f"{speakable_phone(self.slots['phone'])}, and ask if they would "
                "like to book a new one instead."
            )
        self.target = appts[0]
        self._goto(State.CONFIRM_TARGET)
        return self._say(
            f"Confirm this is the appointment they mean: "
            f"{self.store.doctor_name(self.target.doctor_id)}, "
            f"{speakable_date(self.target.date, self.today)} at "
            f"{speakable_time(self.target.time)}. Ask yes or no."
        )

    def _confirm_target(self, u: Understanding) -> str:
        if u.decision == "yes":
            if self.slots["intent"] == "cancel":
                self._goto(State.CONFIRM)
                return self._say(
                    f"Ask them to confirm cancelling the appointment with "
                    f"{self.store.doctor_name(self.target.doctor_id)} on "
                    f"{speakable_date(self.target.date, self.today)} at "
                    f"{speakable_time(self.target.time)}. Yes or no."
                )
            # reschedule: keep the doctor, collect new date/time
            self.slots["doctor"] = next(
                (d for d in self.store.doctors() if d["id"] == self.target.doctor_id),
                None,
            )
            self.slots["date"] = self.slots["time"] = None
            self._goto(State.COLLECT_BOOKING)
            return self._say("Ask what new day they would like instead.")
        if u.decision == "no":
            if self._fail("target"):
                return self._handoff("could not identify the right appointment")
            self.target = None
            self._goto(State.INTENT)
            return self._say("Ask them to describe which appointment they mean.")
        if self._fail("target_decision"):
            return self._handoff("unclear confirmation")
        return self._say("Ask again, clearly: is that the appointment they mean, yes or no?")

    def _offer_slots(self) -> str:
        self._goto(State.OFFER_SLOTS)
        date, want = self.slots["date"], self.slots["time"]
        slots = self.store.get_availability(self.slots["doctor"]["id"], date, near=want)
        if not slots:
            if self._fail("availability"):
                return self._handoff("no availability")
            self.slots["date"] = None
            return self._say(
                f"Say there are no free slots with {self._doctor_name()} on "
                f"{speakable_date(date, self.today)} and ask for another day."
            )
        if want and slots[0] == want:
            self.slots["time"] = want
            return self._to_confirm()
        self.offered, self.offered_date = slots[:2], date
        opts = " or ".join(speakable_time(t) for t in self.offered)
        lead = (f"The exact time they wanted is not free. " if want else "")
        return self._say(
            f"{lead}Offer these available times with {self._doctor_name()} on "
            f"{speakable_date(date, self.today)}: {opts}. Ask which they prefer."
        )

    def _pick_slot(self, u: Understanding, bad: list[str]) -> str:
        if u.choice_index and 1 <= u.choice_index <= len(self.offered):
            self.slots["time"] = self.offered[u.choice_index - 1]
            return self._to_confirm()
        if u.time_phrase and self.slots["time"] in self.offered:
            return self._to_confirm()
        if u.time_phrase and self.slots["time"]:
            return self._offer_slots()      # asked for a different time: re-check
        if u.date_phrase and self.slots["date"] != self.offered_date:
            return self._offer_slots()      # different day
        if u.decision == "no":
            if self._fail("offer"):
                return self._handoff("no offered slot accepted")
            self.slots["date"] = self.slots["time"] = None
            return self._say("Ask what other day or time would work better.")
        if self._fail("offer"):
            return self._handoff("could not agree on a slot")
        opts = " or ".join(speakable_time(t) for t in self.offered)
        return self._say(f"Repeat the choices simply: {opts}. Which one?")

    def _to_confirm(self) -> str:
        self._goto(State.CONFIRM)
        verb = "rescheduled to" if self.slots["intent"] == "reschedule" else "booked for"
        return self._say(
            f"Read back for final confirmation: appointment {verb} {self._details()}, "
            f"mobile number {speakable_phone(self.slots['phone'])}. Ask: is that correct?"
        )

    def _confirm(self, u: Understanding, bad: list[str]) -> str:
        if u.decision == "yes":
            return self._commit()
        if u.decision == "no" or u.date_phrase or u.time_phrase or u.doctor or u.name:
            if self._fail("confirm"):
                return self._handoff("confirmation kept failing")
            changed = not bad and (u.date_phrase or u.time_phrase or u.doctor or u.name)
            if changed and self.slots["intent"] != "cancel":
                return self._offer_slots()  # corrected value already absorbed
            if self.slots["intent"] == "cancel":
                self._goto(State.CLOSE)
                self.done = True
                return self._say("Say okay, the appointment is unchanged, and say goodbye.")
            return self._say("Ask what should be corrected: the doctor, the day, or the time?")
        if self._fail("confirm"):
            return self._handoff("unclear confirmation")
        return self._say("Ask again plainly: yes or no?")

    def _commit(self) -> str:
        self._goto(State.COMMIT)
        s = self.slots
        try:
            if s["intent"] == "cancel":
                appt = self.store.cancel(self.target.id)
                summary = (f"the appointment with {self.store.doctor_name(appt.doctor_id)} on "
                           f"{speakable_date(appt.date, self.today)} is cancelled")
            elif s["intent"] == "reschedule":
                appt = self.store.reschedule(self.target.id, s["date"], s["time"])
                summary = f"the appointment is moved to {self._details()}"
            else:
                appt = self.store.book(s["name"], s["phone"], s["doctor"]["id"],
                                       s["date"], s["time"])
                summary = f"the appointment is confirmed: {self._details()}"
        except SlotTakenError:
            self.slots["time"] = None
            return self._offer_slots()
        self.result = {"action": s["intent"] or "book", "appointment_id": appt.id,
                       "status": appt.status, "date": appt.date.isoformat(),
                       "time": appt.time.isoformat(), "doctor_id": appt.doctor_id}
        self._goto(State.CLOSE)
        self.done = True
        return self._say(f"Tell the caller {summary}. Thank them and say goodbye.")

    # ------------------------------------------------------------- support

    def _handoff(self, reason: str) -> str:
        self.log.append(Turn("system", f"handoff: {reason}", State.HANDOFF.value))
        self._goto(State.HANDOFF)
        self.done = True
        return self._say(
            "Apologize that you could not complete this over the phone, say a "
            "staff member will call them back shortly, and say goodbye."
        )

    def _bad_prefix(self, bad: list[str]) -> str:
        if not bad:
            return ""
        return ("First, politely say you did not catch their "
                + " and ".join(bad) + " correctly. Then: ")

    def _state_hint(self) -> str:
        hints = {
            State.GREET: "The caller was just greeted and asked how we can help.",
            State.IDENTIFY: "We asked for their name or phone number.",
            State.INTENT: "We asked whether they want to book, reschedule, or cancel.",
            State.COLLECT_BOOKING: "We asked for doctor, day, or time preference.",
            State.FIND_APPOINTMENT: "We are locating their existing appointment.",
            State.CONFIRM_TARGET: "We asked them to confirm which appointment (yes/no).",
            State.OFFER_SLOTS: "We offered specific time options; they may pick one "
                               "(choice_index), name a time, or refuse.",
            State.CONFIRM: "We read back the full booking and asked yes/no.",
        }
        return hints.get(self.state, "General conversation.")
