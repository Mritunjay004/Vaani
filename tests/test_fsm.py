"""FSM control-flow tests with a fake LLM — no network.

The fake returns scripted Understanding objects and echoes render
instructions, so these tests pin down transitions, retries, tool calls,
and commit behavior deterministically.
"""
import datetime as dt

import pytest

from vaani.clinic.store import ClinicStore
from vaani.config import load_config
from vaani.dialogue.fsm import DialogueSession, State
from vaani.llm import Understanding

TODAY = dt.date(2026, 8, 31)  # a Monday


class FakeLlm:
    def __init__(self, understandings):
        self.queue = list(understandings)
        self.rendered = []

    def understand(self, utterance, state_hint, history=None):
        return self.queue.pop(0)

    def render(self, instruction, language_code, history=None):
        self.rendered.append(instruction)
        return instruction  # echo so tests can assert on content


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def store(cfg, tmp_path):
    return ClinicStore(cfg.clinic, tmp_path / "test.db")


def make_session(cfg, store, understandings, phone=None):
    llm = FakeLlm(understandings)
    s = DialogueSession(cfg, store, llm, language_code="hi-IN",
                        caller_phone=phone, today=TODAY)
    s.open()
    return s, llm


def test_happy_booking_path(cfg, store):
    session, llm = make_session(cfg, store, [
        Understanding(intent="book", name="Ravi Kumar"),
        Understanding(phone_raw="98 76 54 32 10"),
        Understanding(doctor="Mehta", date_phrase="कल", time_phrase="तीन बजे"),
        Understanding(decision="yes"),
    ])
    session.step("मुझे appointment book करनी है, मेरा नाम रवि कुमार है")
    assert session.state == State.IDENTIFY          # asking for phone
    session.step("अट्ठानवे छिहत्तर...")
    assert session.slots["phone"] == "9876543210"
    reply = session.step("कल तीन बजे डॉक्टर मेहता")
    assert session.state == State.CONFIRM
    assert "3 PM" in reply and "98 76 54 32 10" in reply
    session.step("हाँ")
    assert session.state == State.CLOSE and session.done
    assert session.result["action"] == "book"
    appts = store.find_appointments("9876543210")
    assert len(appts) == 1
    assert appts[0].date == TODAY + dt.timedelta(days=1)
    assert appts[0].time == dt.time(15, 0)


def test_taken_slot_offers_alternatives(cfg, store):
    date = TODAY + dt.timedelta(days=1)
    store.book("Other Patient", "9000000001", "dr_mehta", date, dt.time(15, 0))
    session, llm = make_session(cfg, store, [
        Understanding(intent="book", name="Sita", phone_raw="9876543210",
                      doctor="Mehta", date_phrase="कल", time_phrase="तीन बजे"),
        Understanding(choice_index=1),
        Understanding(decision="yes"),
    ])
    reply = session.step("सब कुछ एक साथ")
    assert session.state == State.OFFER_SLOTS
    assert "not free" in reply
    session.step("पहला वाला")
    assert session.state == State.CONFIRM
    session.step("हाँ")
    assert session.result["action"] == "book"
    assert session.result["time"] != "15:00:00"


def test_cancel_path(cfg, store):
    date = TODAY + dt.timedelta(days=2)
    appt = store.book("Ravi", "9876543210", "dr_mehta", date, dt.time(15, 0))
    session, llm = make_session(cfg, store, [
        Understanding(intent="cancel", name="Ravi", phone_raw="9876543210"),
        Understanding(decision="yes"),   # confirm target
        Understanding(decision="yes"),   # confirm cancel
    ], phone=None)
    session.step("appointment cancel करना है")
    assert session.state == State.CONFIRM_TARGET
    session.step("हाँ")
    assert session.state == State.CONFIRM
    session.step("हाँ")
    assert session.done and session.result["action"] == "cancel"
    assert store.get(appt.id).status == "cancelled"


def test_reschedule_path(cfg, store):
    date = TODAY + dt.timedelta(days=2)
    appt = store.book("Ravi", "9876543210", "dr_mehta", date, dt.time(15, 0))
    session, llm = make_session(cfg, store, [
        Understanding(intent="reschedule", name="Ravi", phone_raw="9876543210"),
        Understanding(decision="yes"),                       # right appointment
        Understanding(date_phrase="अगले शुक्रवार", time_phrase="सुबह दस बजे"),
        Understanding(decision="yes"),                       # confirm
    ])
    session.step("appointment बदलनी है")
    session.step("हाँ")
    assert session.state == State.COLLECT_BOOKING
    session.step("अगले शुक्रवार सुबह दस बजे")
    assert session.state == State.CONFIRM
    session.step("ठीक है")
    moved = store.get(appt.id)
    assert moved.date == TODAY + dt.timedelta(days=4)       # this Friday
    assert moved.time == dt.time(10, 0)


def test_two_failures_on_slot_handoff(cfg, store):
    session, llm = make_session(cfg, store, [
        Understanding(intent="book", name="Ravi", phone_raw="12345"),   # bad
        Understanding(phone_raw="99"),                                   # bad again
    ])
    session.step("book, Ravi, number 12345")
    assert session.state == State.IDENTIFY          # one failure: re-ask
    session.step("99")
    assert session.state == State.HANDOFF and session.done


def test_first_ask_is_not_a_failure(cfg, store):
    session, llm = make_session(cfg, store, [
        Understanding(intent="book"),
        Understanding(name="Ravi"),
        Understanding(phone_raw="9876543210"),
        Understanding(doctor="Mehta"),
        Understanding(date_phrase="कल"),
        Understanding(time_phrase="तीन बजे"),
        Understanding(decision="yes"),
    ])
    # answering each question one at a time must never trip the retry limit
    for text in ["book करनी है", "रवि", "9876543210", "मेहता", "कल", "तीन बजे"]:
        session.step(text)
        assert session.state != State.HANDOFF
    session.step("हाँ")
    assert session.result["action"] == "book"


def test_wants_human_immediate_handoff(cfg, store):
    session, llm = make_session(cfg, store, [Understanding(wants_human=True)])
    session.step("मुझे किसी इंसान से बात करनी है")
    assert session.state == State.HANDOFF and session.done


def test_no_appointment_found_offers_booking(cfg, store):
    session, llm = make_session(cfg, store, [
        Understanding(intent="cancel", name="Ravi", phone_raw="9876543210"),
        Understanding(intent="book", doctor="Iyer", date_phrase="कल",
                      time_phrase="सुबह दस बजे"),
        Understanding(decision="yes"),
    ])
    reply = session.step("cancel करना है")
    assert session.state == State.INTENT
    assert "no upcoming appointment" in reply
    session.step("ठीक है, नई book कर दो")
    assert session.state == State.CONFIRM
    session.step("हाँ")
    assert session.result["action"] == "book"


def test_confirm_no_asks_for_correction_then_fixes(cfg, store):
    session, llm = make_session(cfg, store, [
        Understanding(intent="book", name="Ravi", phone_raw="9876543210",
                      doctor="Mehta", date_phrase="कल", time_phrase="तीन बजे"),
        Understanding(decision="no", time_phrase="सुबह दस बजे"),
        Understanding(decision="yes"),
    ])
    session.step("सब बता दिया")
    assert session.state == State.CONFIRM
    session.step("नहीं, सुबह दस बजे")
    assert session.state == State.CONFIRM      # re-offered with corrected time
    session.step("हाँ")
    assert session.result["time"] == "10:00:00"


class TestResolveDoctor:
    @pytest.mark.parametrize("spoken,expected", [
        ("Dr. Mehta", "dr_mehta"),
        ("डॉक्टर मेहता", "dr_mehta"),
        ("मेहता", "dr_mehta"),
        ("மேத்தா", "dr_mehta"),
        ("pediatrics", "dr_iyer"),
        ("बच्चों के डॉक्टर", "dr_iyer"),
        ("Dr Iyer", "dr_iyer"),
    ])
    def test_scripts_and_aliases(self, store, spoken, expected):
        doc = store.resolve_doctor(spoken)
        assert doc and doc["id"] == expected

    def test_unknown_is_none(self, store):
        assert store.resolve_doctor("Dr. Unknown") is None
