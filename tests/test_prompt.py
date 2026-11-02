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


def test_prompt_builder_mixes_persona_paraphrases():
    from mini_llm.sft import prompt_builder
    from mini_llm.system_prompts import SYSTEM_PARAPHRASES

    build = prompt_builder(True, 5)
    systems = [build() for _ in range(4000)]
    canonical = sum("Answer briefly and directly." in s for s in systems)
    paraphrased = sum(any(p in s for p in SYSTEM_PARAPHRASES) for s in systems)
    dated = sum(s.startswith("Today is ") for s in systems)
    assert 0.78 < canonical / 4000 < 0.90
    assert 0.10 < paraphrased / 4000 < 0.20
    assert 0.85 < dated / 4000 < 0.95
    assert all("markdown list" in s for s in systems)
    assert prompt_builder(True, 5)() == systems[0]
    assert prompt_builder(False, 5)() is None


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
