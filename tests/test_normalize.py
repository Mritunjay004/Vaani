import datetime as dt

import pytest

from vaani.nlu.normalize import (
    fold_numerals,
    parse_date,
    parse_phone,
    parse_time,
    speakable_phone,
    speakable_time,
)

# Monday 2026-08-31 as the reference "today" for date arithmetic
TODAY = dt.date(2026, 8, 31)


def d(days: int) -> dt.date:
    return TODAY + dt.timedelta(days=days)


class TestFoldNumerals:
    def test_devanagari(self):
        assert fold_numerals("३ बजे") == "3 बजे"

    def test_tamil(self):
        assert fold_numerals("௩ மணி") == "3 மணி"

    def test_ascii_untouched(self):
        assert fold_numerals("3 pm") == "3 pm"


class TestParseDate:
    @pytest.mark.parametrize("phrase,lang,expected", [
        ("आज", "hi-IN", d(0)),
        ("कल", "hi-IN", d(1)),
        ("परसों", "hi-IN", d(2)),
        ("parson", "hi-IN", d(2)),
        ("उद्या", "mr-IN", d(1)),
        ("परवा", "mr-IN", d(2)),
        ("நாளை", "ta-IN", d(1)),
        ("tomorrow", "en-IN", d(1)),
        ("day after tomorrow", "en-IN", d(2)),
    ])
    def test_relative_words(self, phrase, lang, expected):
        assert parse_date(phrase, lang, TODAY) == expected

    def test_weekday_upcoming(self):
        # today is Monday; "friday" = this Friday
        assert parse_date("friday", "en-IN", TODAY) == d(4)

    def test_weekday_same_day_rolls_a_week(self):
        assert parse_date("सोमवार", "hi-IN", TODAY) == d(7)

    def test_agle_mangalvar_near_day_means_next_week(self):
        # Monday + "agle mangalvar": caller means the following week, not tomorrow
        assert parse_date("अगले मंगलवार", "hi-IN", TODAY) == d(8)

    def test_next_marker_far_day_stays_this_week(self):
        assert parse_date("अगले शुक्रवार", "hi-IN", TODAY) == d(4)

    def test_marathi_weekday(self):
        assert parse_date("मंगळवार", "mr-IN", TODAY) == d(1)

    def test_tamil_weekday_with_suffix(self):
        assert parse_date("வெள்ளிக்கிழமை", "ta-IN", TODAY) == d(4)

    def test_numeric_date(self):
        assert parse_date("15/9", "en-IN", TODAY) == dt.date(2026, 9, 15)

    def test_numeric_date_past_rolls_year(self):
        assert parse_date("15/3", "en-IN", TODAY) == dt.date(2027, 3, 15)

    def test_tareekh(self):
        assert parse_date("पांच तारीख", "hi-IN", TODAY) == dt.date(2026, 9, 5)

    def test_garbage_is_none(self):
        assert parse_date("डॉक्टर मेहता", "hi-IN", TODAY) is None


class TestParseTime:
    @pytest.mark.parametrize("phrase,lang,expected", [
        ("तीन बजे", "hi-IN", dt.time(15, 0)),          # snaps into clinic hours
        ("teen baje", "hi-IN", dt.time(15, 0)),
        ("सुबह दस बजे", "hi-IN", dt.time(10, 0)),
        ("शाम पांच बजे", "hi-IN", dt.time(17, 0)),
        ("साढ़े तीन बजे", "hi-IN", dt.time(15, 30)),
        ("सव्वा चार", "hi-IN", dt.time(16, 15)),
        ("पौने पांच", "hi-IN", dt.time(16, 45)),
        ("सकाळी अकरा वाजता", "mr-IN", dt.time(11, 0)),
        ("दुपारी चार वाजता", "mr-IN", dt.time(16, 0)),
        ("காலை பத்து மணி", "ta-IN", dt.time(10, 0)),
        ("மாலை ஐந்து மணி", "ta-IN", dt.time(17, 0)),
        ("3 PM", "en-IN", dt.time(15, 0)),
        ("3:45 pm", "en-IN", dt.time(15, 45)),
        ("half past four", "en-IN", dt.time(16, 30)),
        ("10 am", "en-IN", dt.time(10, 0)),
        ("४ बजे", "hi-IN", dt.time(16, 0)),            # native numeral + snap
    ])
    def test_times(self, phrase, lang, expected):
        assert parse_time(phrase, lang) == expected

    def test_no_time_is_none(self):
        assert parse_time("डॉक्टर मेहता", "hi-IN") is None

    def test_morning_hour_stays_am_inside_hours(self):
        assert parse_time("10 बजे", "hi-IN") == dt.time(10, 0)


class TestParsePhone:
    def test_plain(self):
        assert parse_phone("9876543210") == "9876543210"

    def test_pairs(self):
        assert parse_phone("98 76 54 32 10") == "9876543210"

    def test_country_code(self):
        assert parse_phone("+91 98765 43210") == "9876543210"

    def test_leading_zero(self):
        assert parse_phone("09876543210") == "9876543210"

    def test_native_numerals(self):
        assert parse_phone("९८७६५४३२१०") == "9876543210"

    def test_spoken_digit_words(self):
        assert parse_phone("nine eight seven six five four three two one zero") == "9876543210"

    def test_hindi_digit_words(self):
        assert parse_phone("नौ आठ सात छह पांच चार तीन दो एक शून्य") == "9876543210"

    def test_too_short_is_none(self):
        assert parse_phone("12345") is None

    def test_invalid_leading_digit_is_none(self):
        assert parse_phone("1234567890") is None


class TestSpeakable:
    def test_time_whole_hour(self):
        assert speakable_time(dt.time(15, 0)) == "3 PM"

    def test_time_with_minutes(self):
        assert speakable_time(dt.time(15, 30)) == "3:30 PM"

    def test_phone_pairs(self):
        assert speakable_phone("9876543210") == "98 76 54 32 10"
