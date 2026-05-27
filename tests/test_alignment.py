from app.services.feedback.alignment import align, tokenize


def _kinds(ops):
    return [op.kind for op in ops]


def test_identical_sentences_all_match():
    target = tokenize("the cat is on the mat")
    recognized = tokenize("the cat is on the mat")
    ops = align(target, recognized)
    assert set(_kinds(ops)) == {"match"}
    assert len(ops) == 6


def test_missing_word_detected():
    target = tokenize("please close the door")
    recognized = tokenize("please close door")
    ops = align(target, recognized)
    missing = [op.target for op in ops if op.kind == "missing"]
    assert missing == ["the"]


def test_extra_word_detected():
    target = tokenize("i would like coffee")
    recognized = tokenize("i would really like coffee")
    ops = align(target, recognized)
    extra = [op.recognized for op in ops if op.kind == "extra"]
    assert extra == ["really"]


def test_substitution_detected():
    target = tokenize("she sells sea shells")
    recognized = tokenize("she sells sea shoes")
    ops = align(target, recognized)
    subs = [(op.target, op.recognized) for op in ops if op.kind == "substitution"]
    assert subs == [("shells", "shoes")]


def test_tokenize_strips_punctuation_and_case():
    assert tokenize("Hello, World!") == ["hello", "world"]


def test_empty_recognized_marks_all_missing():
    target = tokenize("a b c")
    ops = align(target, [])
    assert _kinds(ops) == ["missing", "missing", "missing"]
