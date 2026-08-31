"""Deterministic normalization of dates, times, and phone numbers.

The LLM extracts raw spans ("अगले मंगलवार", "साढ़े तीन बजे", "98765 43210");
this layer resolves them to date/time/digits. The LLM never does date
arithmetic. Language keys are BCP-47 prefixes: hi, mr, ta, en. Unknown
languages fall through to the union of all tables (code-mixed callers).
"""
from __future__ import annotations

import datetime as dt
import re
import unicodedata

# ---------------------------------------------------------------- numerals


def fold_numerals(text: str) -> str:
    """Convert native-script digits (०३, ௩, …) in any script to ASCII."""
    out = []
    for ch in text:
        d = unicodedata.digit(ch, None)
        out.append(str(d) if d is not None and not ch.isascii() else ch)
    return "".join(out)


# ------------------------------------------------------------- word tables

_NUM_WORDS: dict[str, dict[str, int]] = {
    "hi": {"शून्य": 0, "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पांच": 5, "पाँच": 5,
           "छह": 6, "छे": 6, "सात": 7, "आठ": 8, "नौ": 9, "दस": 10, "ग्यारह": 11,
           "बारह": 12,
           "ek": 1, "do": 2, "teen": 3, "tin": 3, "char": 4, "chaar": 4,
           "paanch": 5, "panch": 5, "chhah": 6, "che": 6, "saat": 7, "aath": 8,
           "nau": 9, "das": 10, "gyarah": 11, "barah": 12},
    "mr": {"शून्य": 0, "एक": 1, "दोन": 2, "तीन": 3, "चार": 4, "पाच": 5, "सहा": 6,
           "सात": 7, "आठ": 8, "नऊ": 9, "दहा": 10, "अकरा": 11, "बारा": 12,
           "don": 2, "saha": 6, "daha": 10, "akra": 11, "bara": 12},
    "ta": {"பூஜ்யம்": 0, "ஒன்று": 1, "ஒரு": 1, "இரண்டு": 2, "மூன்று": 3,
           "நான்கு": 4, "ஐந்து": 5, "ஆறு": 6, "ஏழு": 7, "எட்டு": 8,
           "ஒன்பது": 9, "பத்து": 10, "பதினொன்று": 11, "பன்னிரண்டு": 12,
           "onru": 1, "oru": 1, "irandu": 2, "moonru": 3, "moondru": 3,
           "nangu": 4, "aindhu": 5, "aaru": 6, "ezhu": 7, "ettu": 8,
           "onbadhu": 9, "pathu": 10},
    "en": {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
           "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
           "eleven": 11, "twelve": 12},
}

_TODAY = {"आज", "aaj", "இன்று", "indru", "today"}
_TOMORROW = {"कल", "kal", "उद्या", "udya", "udyaa", "நாளை", "naalai", "nalai",
             "tomorrow"}
_DAY_AFTER = {"परसों", "परसो", "parson", "parso", "परवा", "parva",
              "நாளைமறுநாள்", "day after tomorrow"}
_NEXT_MARKERS = {"अगले", "अगला", "agle", "agla", "पुढच्या", "पुढील", "pudhchya",
                 "அடுத்த", "adutha", "next"}

# weekday index 0=Monday, matching datetime.weekday()
_WEEKDAYS: dict[str, int] = {}
for _i, _names in enumerate([
    ["monday", "सोमवार", "somvar", "somwar", "திங்கள்", "thingal"],
    ["tuesday", "मंगलवार", "मंगळवार", "mangalvar", "mangalwar", "செவ்வாய்", "sevvai"],
    ["wednesday", "बुधवार", "budhvar", "budhwar", "புதன்", "budhan"],
    ["thursday", "गुरुवार", "बृहस्पतिवार", "guruvar", "guruwar", "வியாழன்", "viyazhan"],
    ["friday", "शुक्रवार", "shukravar", "shukrawar", "வெள்ளி", "velli"],
    ["saturday", "शनिवार", "shanivar", "shaniwar", "சனி", "sani"],
    ["sunday", "रविवार", "ravivar", "raviwar", "ஞாயிறு", "gnayiru"],
]):
    for _n in _names:
        _WEEKDAYS[_n] = _i

_MORNING = {"सुबह", "subah", "सकाळी", "sakali", "காலை", "kaalai", "morning", "am"}
_EVENING = {"शाम", "shaam", "sham", "दोपहर", "dopahar", "दुपारी", "dupari",
            "संध्याकाळी", "मதியம்", "மதியம்", "madhiyam", "மாலை", "maalai",
            "रात", "raat", "रात्री", "இரவு", "iravu", "afternoon", "evening",
            "night", "pm"}
_HALF_MARKERS = {"साढ़े", "साढे", "sade", "saade", "साडे", "अरै"}  # +30 min
_QUARTER_PAST = {"सव्वा", "sava", "savva"}   # +15
_QUARTER_TO = {"पौने", "paune", "pavne"}     # -15


def _words(text: str) -> list[str]:
    return re.findall(r"[^\s,.'!?।]+", fold_numerals(text).lower())


def _word_to_num(word: str, lang: str) -> int | None:
    if word.isdigit():
        return int(word)
    table = _NUM_WORDS.get(lang)
    if table and word in table:
        return table[word]
    for t in _NUM_WORDS.values():  # code-mixed fallback
        if word in t:
            return t[word]
    return None


def _lang_key(language_code: str | None) -> str:
    return (language_code or "").split("-")[0].lower()


# ------------------------------------------------------------------ dates


def parse_date(phrase: str, language_code: str | None = None,
               today: dt.date | None = None) -> dt.date | None:
    """Resolve a spoken date phrase to a concrete future date."""
    today = today or dt.date.today()
    lang = _lang_key(language_code)
    folded = fold_numerals(phrase).lower()
    words = _words(phrase)
    joined = "".join(words)

    if _TODAY & set(words):
        return today
    if _DAY_AFTER & set(words) or "நாளை மறுநாள்" in folded or "day after" in folded:
        return today + dt.timedelta(days=2)
    if _TOMORROW & set(words):
        return today + dt.timedelta(days=1)

    # weekday, optionally with a "next" marker; Tamil often suffixes கிழமை
    has_next = bool(_NEXT_MARKERS & set(words))
    for w in words:
        # Tamil weekdays often carry a கிழமை suffix with a sandhi க்
        base = w.removesuffix("கிழமை").removesuffix("kizhamai").removesuffix("க்")
        if base in _WEEKDAYS:
            ahead = (_WEEKDAYS[base] - today.weekday()) % 7
            if ahead == 0:
                ahead = 7
            # "agle mangalvar" said early in the week usually means the
            # *following* week's Tuesday, not tomorrow
            if has_next and ahead <= 2:
                ahead += 7
            return today + dt.timedelta(days=ahead)

    # numeric dd/mm or dd-mm
    m = re.search(r"\b(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?\b", folded)
    if m:
        day, month = int(m.group(1)), int(m.group(2))
        year = int(m.group(3)) if m.group(3) else today.year
        if year < 100:
            year += 2000
        try:
            d = dt.date(year, month, day)
        except ValueError:
            return None
        if d < today and not m.group(3):
            d = dt.date(year + 1, month, day)
        return d

    # "<n> tareekh / तारीख / தேதி" → that day-of-month, this month or next
    if {"तारीख", "tareekh", "tarikh", "தேதி", "thedhi"} & set(words) or "तारखेला" in joined:
        for w in words:
            n = _word_to_num(w, lang)
            if n and 1 <= n <= 31:
                month, year = today.month, today.year
                if n < today.day:
                    month += 1
                    if month == 13:
                        month, year = 1, year + 1
                try:
                    return dt.date(year, month, n)
                except ValueError:
                    return None
    return None


# ------------------------------------------------------------------ times


def parse_time(phrase: str, language_code: str | None = None,
               open_hour: int = 9, close_hour: int = 18) -> dt.time | None:
    """Resolve a spoken time to a 24h clock time.

    Ambiguous hours (no am/pm word) snap into clinic hours: "4 baje" → 16:00
    when the clinic runs 9–18.
    """
    lang = _lang_key(language_code)
    folded = fold_numerals(phrase).lower()
    words = _words(phrase)

    # explicit hh:mm
    m = re.search(r"\b(\d{1,2})[:.](\d{2})\b", folded)
    hour = minute = None
    if m:
        hour, minute = int(m.group(1)), int(m.group(2))
    else:
        minute = 0
        for w in words:
            n = _word_to_num(w, lang)
            if n is not None and 0 <= n <= 23:
                hour = n
                break
        if hour is None:
            return None
        if _HALF_MARKERS & set(words) or "half past" in folded:
            minute = 30
        elif _QUARTER_PAST & set(words) or "quarter past" in folded:
            minute = 15
        elif _QUARTER_TO & set(words) or "quarter to" in folded:
            minute = 45
            hour -= 1
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None

    is_am = bool(_MORNING & set(words))
    is_pm = bool(_EVENING & set(words)) or "p.m" in folded
    if "a.m" in folded:
        is_am, is_pm = True, False
    if hour <= 12:
        if is_pm and hour != 12:
            hour += 12
        elif is_am and hour == 12:
            hour = 0
        elif not is_am and not is_pm and hour < open_hour:
            if open_hour <= hour + 12 <= close_hour:  # snap into clinic hours
                hour += 12
    return dt.time(hour, minute)


# ----------------------------------------------------------------- phones


_PHONE_DIGIT_WORDS = {**_NUM_WORDS["en"], **_NUM_WORDS["hi"]}


def parse_phone(phrase: str) -> str | None:
    """Extract a 10-digit Indian mobile number from spoken/mixed input."""
    digits = []
    for w in _words(phrase):
        if w.isdigit():
            digits.append(w)
        elif w in _PHONE_DIGIT_WORDS and _PHONE_DIGIT_WORDS[w] <= 9:
            digits.append(str(_PHONE_DIGIT_WORDS[w]))
    s = "".join(digits)
    if len(s) == 12 and s.startswith("91"):
        s = s[2:]
    elif len(s) == 11 and s.startswith("0"):
        s = s[1:]
    return s if len(s) == 10 and s[0] in "6789" else None


# ------------------------------------------------------- speakable output


def speakable_date(d: dt.date, today: dt.date | None = None) -> str:
    """English speakable form; the render LLM phrases it in the session language."""
    today = today or dt.date.today()
    delta = (d - today).days
    label = d.strftime("%A, %-d %B")
    if delta == 0:
        return f"today, {label}"
    if delta == 1:
        return f"tomorrow, {label}"
    return label


def speakable_time(t: dt.time) -> str:
    return t.strftime("%-I:%M %p" if t.minute else "%-I %p")


def speakable_phone(phone: str) -> str:
    """Pairs with spaces — how Indian callers say and verify numbers."""
    return " ".join(phone[i:i + 2] for i in range(0, len(phone), 2))
