import numpy as np
import pytest
from numpy import ndarray


class FakeEncoder:
    """Deterministic stand-in for SentenceTransformer used in fast unit tests.

    Each text is mapped to a vector by keyword: texts containing "alpha" land
    near one axis, texts containing "bravo" near another, and everything else is
    pushed to a unique far-away position so it reads as noise. A tiny
    index-derived jitter keeps points distinct without merging the groups.
    """

    DIM = 8

    def encode(self, texts: list[str], show_progress_bar: bool = False) -> ndarray:
        vectors = []
        for i, text in enumerate(texts):
            vector = np.zeros(self.DIM, dtype=np.float32)
            lowered = text.lower()
            if "alpha" in lowered:
                vector[0] = 10.0
                vector[4] = i * 0.001
            elif "bravo" in lowered:
                vector[1] = 10.0
                vector[4] = i * 0.001
            else:
                vector[2] = 10.0 + i * 5.0
            vectors.append(vector)
        return np.array(vectors, dtype=np.float32)


@pytest.fixture
def fake_encoder() -> FakeEncoder:
    return FakeEncoder()


@pytest.fixture
def grouped_texts() -> list[str]:
    """Two clear groups of six plus two outliers, in interleaved order.

    The interleaving ensures tests assert on membership rather than on input
    position.
    """
    alpha = [f"alpha message {n}" for n in range(6)]
    bravo = [f"bravo message {n}" for n in range(6)]
    outliers = ["solitary zigzag", "lone quokka"]
    return [
        alpha[0], bravo[0], alpha[1], bravo[1], outliers[0],
        alpha[2], bravo[2], alpha[3], bravo[3], outliers[1],
        alpha[4], bravo[4], alpha[5], bravo[5],
    ]


@pytest.fixture
def alpha_indices() -> list[int]:
    return [0, 2, 5, 7, 10, 12]


@pytest.fixture
def bravo_indices() -> list[int]:
    return [1, 3, 6, 8, 11, 13]


@pytest.fixture
def outlier_indices() -> list[int]:
    return [4, 9]
