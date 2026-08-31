"""Clinic appointment store — the typed tools the agent calls.

Plain SQL on SQLite for local dev; the queries are Postgres-compatible so the
docker step swaps the connection, not the code. The model never touches this
directly: the FSM executes these functions and speaks the results.
"""
from __future__ import annotations

import datetime as dt
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS appointments (
    id TEXT PRIMARY KEY,
    patient_name TEXT NOT NULL,
    phone TEXT NOT NULL,
    doctor_id TEXT NOT NULL,
    date TEXT NOT NULL,
    time TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'booked',
    created_at TEXT NOT NULL,
    UNIQUE (doctor_id, date, time, status)
);
"""


@dataclass
class Appointment:
    id: str
    patient_name: str
    phone: str
    doctor_id: str
    date: dt.date
    time: dt.time
    status: str


def _row_to_appt(row: sqlite3.Row) -> Appointment:
    return Appointment(
        id=row["id"],
        patient_name=row["patient_name"],
        phone=row["phone"],
        doctor_id=row["doctor_id"],
        date=dt.date.fromisoformat(row["date"]),
        time=dt.time.fromisoformat(row["time"]),
        status=row["status"],
    )


class ClinicStore:
    def __init__(self, clinic_cfg: dict[str, Any], db_path: str | Path = "vaani.db"):
        self.cfg = clinic_cfg
        self.db = sqlite3.connect(str(db_path))
        self.db.row_factory = sqlite3.Row
        self.db.executescript(_SCHEMA)

    # -------------------------------------------------------------- doctors

    def doctors(self) -> list[dict[str, Any]]:
        return list(self.cfg.get("doctors", []))

    _HONORIFICS = ("dr.", "dr", "doctor", "डॉक्टर", "डॉ.", "डॉ", "மருத்துவர்", "டாக்டர்")

    def resolve_doctor(self, name_or_dept: str) -> dict[str, Any] | None:
        """Match caller's words (any script) against names, aliases, departments."""
        needle = name_or_dept.lower()
        for h in self._HONORIFICS:
            needle = needle.replace(h, " ")
        needle = needle.strip()
        if not needle:
            return None
        for doc in self.doctors():
            surname = doc["name"].lower().replace("dr.", "").strip()
            candidates = [surname, doc["department"].lower(),
                          *[a.lower() for a in doc.get("aliases", [])]]
            if any(c in needle or needle in c for c in candidates if c):
                return doc
        return None

    # ---------------------------------------------------------------- slots

    def get_availability(self, doctor_id: str, date: dt.date,
                         near: dt.time | None = None, limit: int = 3) -> list[dt.time]:
        hours = self.cfg.get("hours", {})
        open_t = dt.time.fromisoformat(hours.get("open", "09:00"))
        close_t = dt.time.fromisoformat(hours.get("close", "18:00"))
        step = int(hours.get("slot_minutes", 15))

        booked = {
            row["time"]
            for row in self.db.execute(
                "SELECT time FROM appointments WHERE doctor_id = ? AND date = ? "
                "AND status = 'booked'",
                (doctor_id, date.isoformat()),
            )
        }
        slots: list[dt.time] = []
        cur = dt.datetime.combine(date, open_t)
        end = dt.datetime.combine(date, close_t)
        while cur < end:
            t = cur.time()
            if t.isoformat() not in booked:
                slots.append(t)
            cur += dt.timedelta(minutes=step)
        if near:
            anchor = dt.datetime.combine(date, near)
            slots.sort(key=lambda t: abs(dt.datetime.combine(date, t) - anchor))
        return slots[:limit]

    # ------------------------------------------------------------ appointments

    def book(self, patient_name: str, phone: str, doctor_id: str,
             date: dt.date, time: dt.time) -> Appointment:
        if time not in self.get_availability(doctor_id, date, near=time, limit=1):
            raise SlotTakenError(doctor_id, date, time)
        appt = Appointment(
            id=f"apt_{uuid.uuid4().hex[:8]}",
            patient_name=patient_name, phone=phone, doctor_id=doctor_id,
            date=date, time=time, status="booked",
        )
        self.db.execute(
            "INSERT INTO appointments (id, patient_name, phone, doctor_id, date, time, "
            "status, created_at) VALUES (?, ?, ?, ?, ?, ?, 'booked', ?)",
            (appt.id, patient_name, phone, doctor_id, date.isoformat(),
             time.isoformat(), dt.datetime.now(dt.UTC).isoformat()),
        )
        self.db.commit()
        return appt

    def find_appointments(self, phone: str) -> list[Appointment]:
        rows = self.db.execute(
            "SELECT * FROM appointments WHERE phone = ? AND status = 'booked' "
            "AND date >= ? ORDER BY date, time",
            (phone, dt.date.today().isoformat()),
        ).fetchall()
        return [_row_to_appt(r) for r in rows]

    def reschedule(self, appointment_id: str, date: dt.date, time: dt.time) -> Appointment:
        row = self.db.execute(
            "SELECT * FROM appointments WHERE id = ? AND status = 'booked'",
            (appointment_id,),
        ).fetchone()
        if not row:
            raise NotFoundError(appointment_id)
        if time not in self.get_availability(row["doctor_id"], date, near=time, limit=1):
            raise SlotTakenError(row["doctor_id"], date, time)
        self.db.execute(
            "UPDATE appointments SET date = ?, time = ? WHERE id = ?",
            (date.isoformat(), time.isoformat(), appointment_id),
        )
        self.db.commit()
        return self.get(appointment_id)

    def cancel(self, appointment_id: str) -> Appointment:
        row = self.db.execute(
            "SELECT * FROM appointments WHERE id = ? AND status = 'booked'",
            (appointment_id,),
        ).fetchone()
        if not row:
            raise NotFoundError(appointment_id)
        self.db.execute(
            "UPDATE appointments SET status = 'cancelled' WHERE id = ?",
            (appointment_id,),
        )
        self.db.commit()
        return self.get(appointment_id)

    def get(self, appointment_id: str) -> Appointment:
        row = self.db.execute(
            "SELECT * FROM appointments WHERE id = ?", (appointment_id,)
        ).fetchone()
        if not row:
            raise NotFoundError(appointment_id)
        return _row_to_appt(row)

    def doctor_name(self, doctor_id: str) -> str:
        for doc in self.doctors():
            if doc["id"] == doctor_id:
                return doc["name"]
        return doctor_id


class SlotTakenError(Exception):
    def __init__(self, doctor_id: str, date: dt.date, time: dt.time):
        super().__init__(f"slot taken: {doctor_id} {date} {time}")
        self.doctor_id, self.date, self.time = doctor_id, date, time


class NotFoundError(Exception):
    pass
