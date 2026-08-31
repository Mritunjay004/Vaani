"""Text-only conversation loop (build step 2) — the full agent minus audio.

Usage:
  python -m vaani.tools.chat_cli --lang hi-IN                 # interactive
  python -m vaani.tools.chat_cli --lang hi-IN --script convo.txt   # one caller line per file line
  python -m vaani.tools.chat_cli --db vaani.db --seed         # seed a demo appointment first
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys

from ..clinic.store import ClinicStore
from ..config import load_config
from ..dialogue.fsm import DialogueSession
from ..llm import Llm


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--db", default="vaani.db")
    ap.add_argument("--phone", default=None, help="simulated caller-ID")
    ap.add_argument("--script", default=None, help="file with one caller line per line")
    ap.add_argument("--seed", action="store_true",
                    help="seed a demo appointment for reschedule/cancel testing")
    args = ap.parse_args()

    cfg = load_config(args.config)
    store = ClinicStore(cfg.clinic, args.db)
    if args.seed:
        appt = store.book("Ravi Kumar", "9876543210", "dr_mehta",
                          dt.date.today() + dt.timedelta(days=2), dt.time(15, 0))
        print(f"[seeded {appt.id}: Ravi Kumar / dr_mehta / {appt.date} {appt.time}]")

    session = DialogueSession(cfg, store, Llm(cfg.llm),
                              language_code=args.lang, caller_phone=args.phone)
    print(f"agent> {session.open()}")

    lines = None
    if args.script:
        lines = iter(open(args.script, encoding="utf-8").read().splitlines())

    while not session.done:
        if lines is not None:
            try:
                text = next(l for l in lines if l.strip())
            except StopIteration:
                print("[script exhausted before the conversation finished]")
                break
            print(f"caller> {text}")
        else:
            try:
                text = input("caller> ").strip()
            except EOFError:
                break
            if not text:
                continue
        print(f"agent> {session.step(text)}")

    print(f"\n[final state: {session.state.value}]")
    if session.result:
        print(f"[result: {json.dumps(session.result)}]")
    if "--transitions" in sys.argv or args.script:
        for t in session.log:
            if t.role == "system":
                print(f"  {t.text}")


if __name__ == "__main__":
    main()
