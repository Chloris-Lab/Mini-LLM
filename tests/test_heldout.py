from mini_llm.heldout import final_number, shape, shape_ok


def test_shape_reads_lists_and_prose():
    assert shape("A short answer.")[0].startswith("prose")
    assert shape("- one\n- two\n- three") == ("bullets x3", 3)
    assert shape("1. one\n2. two") == ("numbered x2", 2)


def test_shape_ignores_blank_lines_and_accepts_star_bullets():
    assert shape("- one\n\n- two") == ("bullets x2", 2)
    assert shape("* one\n* two") == ("bullets x2", 2)


def test_shape_ok_requires_the_same_list_kind():
    assert shape_ok(("bullets x3", 3), "- a\n- b\n- c")
    assert not shape_ok(("bullets x3", 3), "1. a\n2. b\n3. c")
    assert not shape_ok(("bullets x3", 3), "- a\n- b")


def test_shape_ok_accepts_a_prose_answer_as_long_as_the_reference():
    assert shape_ok(("prose, 1 sentence(s)", 1), "One plain sentence.")
    assert shape_ok(("prose, 2 sentence(s)", 2), "One sentence. Two sentences.")
    assert not shape_ok(("prose, 2 sentence(s)", 2), "One. Two. Three.")
    assert not shape_ok(("prose, 1 sentence(s)", 1), "- a\n- b")


def test_final_number_reads_the_last_integer():
    assert final_number("12 + 8 = 20. The answer is 20.") == 20
    assert final_number("-7") == -7
    assert final_number("no digits here") is None
