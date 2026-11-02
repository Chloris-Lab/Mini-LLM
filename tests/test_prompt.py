import re

from mini_llm.prompt import DEFAULT_SYSTEM_PROMPT, build_system_prompt, format_history, format_prompt


def test_format_prompt_trims_question():
    assert format_prompt(" hi ") == "User: hi\nAssistant:"


def test_format_prompt_with_system():
    assert format_prompt(" hi ", system=" S ") == "System: S\nUser: hi\nAssistant:"
    assert format_prompt(" hi ", system=None) == "User: hi\nAssistant:"


def test_build_system_prompt():
    assert build_system_prompt(DEFAULT_SYSTEM_PROMPT, date="2026-10-23").startswith("Today is 2026-10-23.")
    assert build_system_prompt(" S ", date=None) == "S"


def test_current_date_reads_the_clock_on_every_call(monkeypatch):
    from mini_llm import prompt

    days = iter(["2026-11-02", "2026-11-03"])

    class Day:
        def isoformat(self):
            return next(days)

    class Clock:
        @staticmethod
        def today():
            return Day()

    monkeypatch.setattr(prompt, "date", Clock)
    assert prompt.current_date() == "2026-11-02"
    assert prompt.current_date() == "2026-11-03"


def test_date_pairs_answer_the_system_date():
    from mini_llm.system_prompts import date_pairs

    pairs = date_pairs(60)
    days = set()
    for system, question, answer in pairs:
        day = answer[len("Today is ") : -1]
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", day), answer
        assert answer == f"Today is {day}."
        assert system.startswith(f"Today is {day}.")
        days.add(day)
    assert len(days) > 30


def test_date_pairs_are_deterministic():
    from mini_llm.system_prompts import date_pairs

    assert date_pairs(20) == date_pairs(20)
    assert date_pairs(20) != date_pairs(20, seed=4)


def test_format_history():
    h = format_history("S", [("q1", "a1"), ("q2", None)])
    assert h == "System: S\nUser: q1\nAssistant: a1\nUser: q2\nAssistant:"
    assert format_history(None, [("q1", "a1"), ("q2", None)]) == "User: q1\nAssistant: a1\nUser: q2\nAssistant:"


def test_format_history_rejects_bad_shapes():
    import pytest

    with pytest.raises(ValueError):
        format_history(None, [])
    with pytest.raises(ValueError):
        format_history(None, [("q1", "a1"), ("q2", "a2")])
    with pytest.raises(ValueError):
        format_history(None, [("q1", None), ("q2", None)])
    assert format_history(None, [("q1", None)]) == "User: q1\nAssistant:"
