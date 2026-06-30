import pytest

from ucluster import FuzzyClusterer


@pytest.fixture
def fitted(fake_encoder, grouped_texts):
    clusterer = FuzzyClusterer(encoder=fake_encoder)
    clusterer.fit(grouped_texts)
    return clusterer


def test_similar_texts_share_a_cluster(fitted, alpha_indices, bravo_indices):
    clusters = fitted.clusters()
    alpha_labels = {clusters[i] for i in alpha_indices}
    bravo_labels = {clusters[i] for i in bravo_indices}

    assert len(alpha_labels) == 1
    assert len(bravo_labels) == 1
    assert -1 not in alpha_labels
    assert -1 not in bravo_labels
    assert alpha_labels != bravo_labels


def test_outliers_are_noise(fitted, outlier_indices):
    clusters = fitted.clusters()
    for i in outlier_indices:
        assert clusters[i] == -1


def test_outputs_match_input_length(fitted, grouped_texts):
    n = len(grouped_texts)
    assert len(fitted.clusters()) == n
    assert len(fitted.probabilities()) == n
    assert len(fitted.outlier_probabilities()) == n


def test_probabilities_are_bounded(fitted):
    assert all(0.0 <= p <= 1.0 for p in fitted.probabilities())


def test_outlier_probabilities_complement_probabilities(fitted):
    probs = fitted.probabilities()
    outliers = fitted.outlier_probabilities()
    assert all(abs(o - (1.0 - p)) < 1e-6 for p, o in zip(probs, outliers, strict=True))


def test_fit_is_deterministic(fake_encoder, grouped_texts):
    first = FuzzyClusterer(encoder=fake_encoder)
    first.fit(grouped_texts)
    second = FuzzyClusterer(encoder=fake_encoder)
    second.fit(grouped_texts)
    assert first.clusters() == second.clusters()
