import json
from collections import defaultdict
from typing import Protocol

import nltk
from loguru import logger
from numpy import ndarray
from sklearn.cluster import HDBSCAN

logger.disable(__name__)

nltk.download("punkt_tab", quiet=True)


def preprocess_text(text: str) -> str:
    text = text.encode("utf-8", "replace").decode()
    return " ".join(nltk.word_tokenize(text.lower()))


class TextClusterer:
    def fit(self, texts: list[str]) -> None:
        raise NotImplementedError

    def clusters(self) -> list[int]:
        raise NotImplementedError

    def probabilities(self) -> list[float]:
        raise NotImplementedError

    def outlier_probabilities(self) -> list[float]:
        raise NotImplementedError


class Encoder(Protocol):
    def encode(self, texts: list[str], show_progress_bar: bool = ...) -> ndarray: ...


class FuzzyClusterer(TextClusterer):
    DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        encoder: Encoder | None = None,
        min_cluster_size: int = 3,
        min_samples: int = 3,
        alpha: float = 1.0,
        epsilon: float = 0.0,
    ) -> None:
        if encoder is None:
            from sentence_transformers import SentenceTransformer

            encoder = SentenceTransformer(model)
        self._encoder = encoder
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.alpha = alpha
        self.epsilon = epsilon

    def _vectorize(self, texts: list[str]) -> ndarray:
        return self._encoder.encode(texts, show_progress_bar=False)

    def fit(self, texts: list[str]) -> None:
        logger.info("Encoding texts...")
        vectors = self._vectorize(texts)
        logger.info("Clustering texts...")
        self._clusterer = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            alpha=self.alpha,
            cluster_selection_epsilon=self.epsilon,
            copy=True,
        )
        self._predictions = self._clusterer.fit_predict(vectors)
        logger.info("Clustering complete!")

    def clusters(self) -> list[int]:
        return self._predictions.tolist()

    def probabilities(self) -> list[float]:
        return self._clusterer.probabilities_.tolist()

    def outlier_probabilities(self) -> list[float]:
        """Return per-point outlier probabilities in [0, 1].

        BREAKING CHANGE in 1.0.0: previously returned `hdbscan.HDBSCAN.outlier_scores_`
        (the GLOSH algorithm), which gave per-point outlier-ness even for in-cluster
        members and was effectively unbounded. sklearn's HDBSCAN does not implement
        GLOSH, so this now returns `1 - probabilities_` instead — bounded in [0, 1],
        with 1.0 for noise points and ~0 for high-confidence cluster members. The
        numbers are not comparable to the old GLOSH scores.
        """
        return (1.0 - self._clusterer.probabilities_).tolist()


class ExactClusterer(TextClusterer):
    def fit(self, texts: list[str]) -> None:
        dupes = defaultdict(list)
        for idx, text in enumerate(texts):
            dupes[preprocess_text(text)].append(idx)

        clusters: dict[int, int] = {}
        next_id = 0
        for occurrences in dupes.values():
            if len(occurrences) <= 1:
                continue
            for occurrence in occurrences:
                clusters[occurrence] = next_id
            next_id += 1

        self._clusters = [clusters.get(i, -1) for i in range(len(texts))]

    def clusters(self) -> list[int]:
        return self._clusters

    def probabilities(self) -> list[float]:
        return [1.0 if c != -1 else 0.0 for c in self.clusters()]

    def outlier_probabilities(self) -> list[float]:
        return [1.0 if c == -1 else 0.0 for c in self.clusters()]


def _display_clusters(texts: list[str], clusters: list[int], probabilities: list[float]) -> None:
    cluster_assignments: dict[int, list[int]] = defaultdict(list)
    for i, assignment in enumerate(clusters):
        cluster_assignments[assignment].append(i)
    for cluster, indices in cluster_assignments.items():
        print(f"Cluster {cluster} -----")
        for idx in indices:
            print(f"{cluster} {probabilities[idx]}: {texts[idx].strip()}")


if __name__ == "__main__":
    logger.enable(__name__)

    with open("data/gettr_posts_small.jsonl") as infile:
        posts = [json.loads(line) for line in infile.readlines()]

    logger.info("Getting text data from input...")
    text_data = [post.get("txt") or "" for post in posts]

    logger.info("Clustering...")
    cl = FuzzyClusterer()
    cl.fit(text_data)

    logger.info("Writing to file...")
    with open("data/clustered.jsonl", "w") as outfile:
        for post, cluster in zip(posts, cl.clusters(), strict=True):
            post["_cluster"] = str(cluster)
            print(json.dumps(post), file=outfile)

    logger.info("Done writing clusters to the file!")
