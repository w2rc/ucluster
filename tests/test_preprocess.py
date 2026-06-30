from ucluster.text_cluster import preprocess_text


def test_lowercases_text():
    assert preprocess_text("Hello World") == "hello world"


def test_separates_punctuation_into_tokens():
    assert preprocess_text("Hello, world!") == "hello , world !"


def test_collapses_runs_of_whitespace():
    assert preprocess_text("spread   out\t\nwords") == "spread out words"


def test_replaces_unencodable_characters_without_raising():
    result = preprocess_text("caf\udce9")
    assert isinstance(result, str)
