import re

from mini_llm import math_data

PLAIN = re.compile(r"^(-?\d+) ([+\-x/]) (\d+) = (-?\d+)\.$")
CARRIED = re.compile(
    r"^(-?\d+) ([+\-]) (\d+) = (\d+) ([+\-]) (\d+) = (-?\d+), then (-?\d+) ([+\-]) (\d+) = (-?\d+)\.$"
)
SPLIT = re.compile(r"^(\d+) x (\d+) = (\d+) x (\d+) \+ (\d+) x (\d+) = (\d+) \+ (\d+) = (\d+)\.$")
SIGNS = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "x": lambda a, b: a * b}
OP_TOKEN = re.compile(r" ([+\-x/]) ")


def check_answer(question, answer):
    assert "$" not in answer, (question, answer)
    assert len(answer) <= 120, (question, answer)

    m = PLAIN.match(answer)
    if m:
        a, sign, b, result = int(m[1]), m[2], int(m[3]), int(m[4])
        if sign == "/":
            assert b != 0 and a % b == 0, (question, answer)
            expected = a // b
        else:
            expected = SIGNS[sign](a, b)
        assert result == expected, (question, answer)
        return

    m = CARRIED.match(answer)
    if m:
        a, sign, b = int(m[1]), m[2], int(m[3])
        assert int(m[4]) == a and m[5] == sign
        step, mid = int(m[6]), int(m[7])
        assert int(m[8]) == mid and m[9] == sign
        rest, result = int(m[10]), int(m[11])
        assert step + rest == b and 0 < step < b, (question, answer)
        assert mid == a + step if sign == "+" else a - step, (question, answer)
        assert result == mid + rest if sign == "+" else mid - rest, (question, answer)
        assert result == SIGNS[sign](a, b), (question, answer)
        return

    m = SPLIT.match(answer)
    if m:
        a, b = int(m[1]), int(m[2])
        assert int(m[3]) == a and int(m[5]) == a, (question, answer)
        high, low = int(m[4]), int(m[6])
        assert high + low == b and high > 0 and low > 0, (question, answer)
        assert int(m[7]) == a * high and int(m[8]) == a * low, (question, answer)
        assert int(m[9]) == int(m[7]) + int(m[8]) == a * b, (question, answer)
        return

    raise AssertionError((question, answer))


def labels(pairs):
    return {math_data_label(answer) for _, answer in pairs}


def math_data_label(answer):
    m = OP_TOKEN.search(answer)
    return {"+": "add", "-": "sub", "x": "mul", "/": "div"}[m[1]] if m else "other"


def test_table_pairs_are_exhaustive_and_correct():
    pairs = math_data.table_pairs(0)
    assert len(pairs) == 144 + 132
    for question, answer in pairs:
        check_answer(question, answer)


def test_synthetic_pairs_are_correct_and_deterministic():
    pairs = math_data.synthetic_pairs(300, seed=5)
    assert len(pairs) == 300
    assert pairs == math_data.synthetic_pairs(300, seed=5)
    for question, answer in pairs:
        check_answer(question, answer)


def test_synthetic_pairs_cover_the_four_operations():
    pairs = math_data.synthetic_pairs(600, seed=7)
    assert labels(pairs) == {"add", "sub", "mul", "div"}


def test_division_is_exact():
    checked = 0
    for question, answer in math_data.synthetic_pairs(2000, seed=9):
        m = PLAIN.match(answer)
        if m and m[2] == "/":
            assert int(m[1]) % int(m[3]) == 0, (question, answer)
            checked += 1
    assert checked > 200


def test_multiplication_reaches_two_digit_factors():
    wide = split = 0
    for question, answer in math_data.synthetic_pairs(2000, seed=11):
        m = SPLIT.match(answer)
        if m:
            split += 1
            assert int(m[2]) >= 10
        if math_data_label(answer) == "mul":
            factor = int(answer.split(" x ")[1].split(" = ")[0])
            wide += int(factor >= 10)
    assert wide > 150, wide
    assert split > 50, split


def test_subtraction_can_be_negative():
    negatives = [
        answer
        for _, answer in math_data.synthetic_pairs(3000, seed=13)
        if PLAIN.match(answer) and PLAIN.match(answer)[2] == "-" and PLAIN.match(answer)[4].startswith("-")
    ]
    assert negatives, "no negative subtraction result"


def test_addition_reaches_six_digits():
    longest = max(len(PLAIN.match(answer)[1]) for _, answer in math_data.synthetic_pairs(2000, seed=17) if PLAIN.match(answer))
    assert longest >= 6, longest


def test_tables_stay_a_small_share():
    tables = set(math_data.table_pairs(1))
    pairs = math_data.synthetic_pairs(1000)
    assert len(pairs) == 1000
    assert sum((q, a) in tables for q, a in pairs) <= 150
    assert set(math_data.synthetic_pairs(100, table_share=1.0)) <= tables


def test_skill_pairs_shape():
    pairs = math_data.skill_pairs(50, seed=2)
    assert len(pairs) == 50
    assert pairs == math_data.skill_pairs(50, seed=2)
    for question, answer in pairs:
        assert question and answer

    count = [a for q, a in pairs if a.endswith(" words.")]
    assert count, "expected word-count examples"

    for q, a in pairs:
        if a.endswith(" words."):
            words = re.search(r'"([^"]+)"', q).group(1).split()
            assert int(a.split()[0]) == len(words)


def test_digit_pre_tokenizer_isolates_digits():
    from mini_llm.prep import digit_pre_tokenizer

    pieces = [t for t, _ in digit_pre_tokenizer().pre_tokenize_str("abc 1234 GPT4 3,472")]
    assert "1234" not in pieces and "472" not in pieces
    assert all(len(p) <= 1 or not p[-1].isdigit() or not p[0].isdigit() for p in pieces)
