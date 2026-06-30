"""Opt-in integration test exercising the real sentence-transformer model.

Marked ``slow`` and excluded from default CI runs (``pytest -m "not slow"``).
Run explicitly with ``pytest -m slow``. The first run downloads ~470MB from the
Hugging Face Hub; the test skips (rather than fails) when the model cannot be
loaded, e.g. offline with an empty cache.
"""

from collections import Counter

import pytest
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError

from ucluster import FuzzyClusterer

pytestmark = pytest.mark.slow

DOG_SENTENCES = [
    "The dog ran quickly across the park",
    "A dog sprinted through the park",
    "The dog dashed across the park grass",
    "A quick dog raced over the green park",
]

PASTA_SENTENCES = [
    "I boiled pasta for dinner tonight",
    "She cooked spaghetti for the evening meal",
    "We made pasta for dinner",
    "He prepared spaghetti this evening",
]

OUTLIER = "Quantum entanglement continues to puzzle physicists"


@pytest.fixture(scope="module")
def real_clusterer() -> FuzzyClusterer:
    # Skip only on the specific huggingface_hub failures that mean the model is
    # genuinely unreachable (offline/uncached or a Hub HTTP error). Any other
    # exception should surface as a real test failure.
    try:
        return FuzzyClusterer()
    except (LocalEntryNotFoundError, HfHubHTTPError) as exc:
        pytest.skip(f"sentence-transformer model unavailable: {exc}")


def _dominant_label(clusters: list[int], indices: range) -> tuple[int, int]:
    label, count = Counter(clusters[i] for i in indices).most_common(1)[0]
    return label, count


def test_paraphrase_groups_cluster_apart(real_clusterer: FuzzyClusterer):
    texts = DOG_SENTENCES + PASTA_SENTENCES + [OUTLIER]
    real_clusterer.fit(texts)
    clusters = real_clusterer.clusters()

    dog_label, dog_count = _dominant_label(clusters, range(0, 4))
    pasta_label, pasta_count = _dominant_label(clusters, range(4, 8))

    # Each paraphrase group should mostly land in a single real cluster...
    assert dog_label != -1
    assert pasta_label != -1
    assert dog_count >= 3
    assert pasta_count >= 3
    # ...and the two topics should not share that cluster.
    assert dog_label != pasta_label
