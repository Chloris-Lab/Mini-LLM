from mini_llm import sft


def message(role, content):
    return {"role": role, "content": content}


def dialogue(turns):
    msgs = [message("system", "be nice")]
    for i in range(turns):
        msgs += [message("user", f"q{i}"), message("assistant", f"a{i}")]
    return {"messages": msgs}


def test_no_robots_keeps_up_to_four_turns(monkeypatch):
    monkeypatch.setattr(sft, "load_dataset", lambda *a, **k: {"train": [dialogue(6)]})
    pairs, conversations = sft.no_robots_pairs()
    assert pairs == [["q0", "a0"]]
    assert len(conversations) == 1
    assert [q for q, _ in conversations[0]] == [f"q{i}" for i in range(sft.MAX_TURNS)]


def test_no_robots_ignores_broken_dialogues(monkeypatch):
    broken = {"messages": [message("assistant", "orphan"), message("user", "q")]}
    single = {"messages": [message("user", "q0"), message("assistant", "a0")]}
    monkeypatch.setattr(sft, "load_dataset", lambda *a, **k: {"train": [broken, single]})
    pairs, conversations = sft.no_robots_pairs()
    assert pairs == [["q0", "a0"]]
    assert conversations == []


def test_oasst_follows_the_thread(monkeypatch):
    rows = []
    parent = None
    for i in range(5):
        rows.append({
            "message_id": f"p{i}", "parent_id": parent, "role": "prompter", "text": f"q{i}",
            "lang": "en", "deleted": False, "review_result": True, "rank": None,
        })
        rows.append({
            "message_id": f"a{i}", "parent_id": f"p{i}", "role": "assistant", "text": f"a{i}",
            "lang": "en", "deleted": False, "review_result": True, "rank": 0,
        })
        parent = f"a{i}"
    monkeypatch.setattr(sft, "load_dataset", lambda *a, **k: rows)
    pairs, conversations = sft.oasst_pairs()
    assert pairs == [["q0", "a0"]]
    assert len(conversations) == 1
    assert [q for q, _ in conversations[0]] == [f"q{i}" for i in range(sft.MAX_TURNS)]


def test_oasst_keeps_a_single_exchange_as_a_pair(monkeypatch):
    rows = [
        {"message_id": "p0", "parent_id": None, "role": "prompter", "text": "q0",
         "lang": "en", "deleted": False, "review_result": True, "rank": None},
        {"message_id": "a0", "parent_id": "p0", "role": "assistant", "text": "a0",
         "lang": "en", "deleted": False, "review_result": True, "rank": 0},
    ]
    monkeypatch.setattr(sft, "load_dataset", lambda *a, **k: rows)
    pairs, conversations = sft.oasst_pairs()
    assert pairs == [["q0", "a0"]]
    assert conversations == []
