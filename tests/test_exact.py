from ucluster import ExactClusterer


def _fit(texts: list[str]) -> ExactClusterer:
    clusterer = ExactClusterer()
    clusterer.fit(texts)
    return clusterer


def test_exact_duplicates_share_a_cluster():
    clusters = _fit(["hi there", "hi there", "bye"]).clusters()
    assert clusters[0] == clusters[1]
    assert clusters[0] != -1
    assert clusters[2] == -1


def test_matching_is_case_insensitive():
    clusters = _fit(["Hello", "hello"]).clusters()
    assert clusters[0] == clusters[1] != -1


def test_distinct_duplicate_groups_get_distinct_ids():
    clusters = _fit(["a", "a", "b", "b"]).clusters()
    assert clusters[0] == clusters[1]
    assert clusters[2] == clusters[3]
    assert clusters[0] != clusters[2]


def test_output_length_and_order_match_input():
    texts = ["a", "b", "a"]
    clusters = _fit(texts).clusters()
    assert len(clusters) == len(texts)
    assert clusters[0] == clusters[2]
    assert clusters[1] == -1


def test_empty_input_yields_empty_clusters():
    assert _fit([]).clusters() == []


def test_probabilities_reflect_membership():
    clusterer = _fit(["a", "a", "b"])
    assert clusterer.probabilities() == [1.0, 1.0, 0.0]
    assert clusterer.outlier_probabilities() == [0.0, 0.0, 1.0]
