# Changelog

## 1.1.0

A maintenance release: an automated test suite, CI, and a dependency refresh.
No breaking API changes.

### Added

- **Test suite.** Unit tests for `preprocess_text`, `ExactClusterer`, and
  `FuzzyClusterer`, plus an opt-in `slow`-marked integration test against the
  real sentence-transformer model. Tests assert clustering structure and
  invariants rather than exact values, so they survive dependency upgrades.
- **CI.** A GitHub Actions workflow runs `ruff`, `ty`, and the test suite on
  Python 3.12 and 3.13.
- **`FuzzyClusterer(encoder=...)`** accepts an injected encoder implementing the
  new `Encoder` protocol. When omitted (the default), the sentence-transformer
  model is built and downloaded exactly as before; the parameter is keyword-only
  so existing positional calls are unaffected.

### Changed

- Dependencies refreshed to their latest releases and floors pinned to match
  (scikit-learn `1.9.0`, sentence-transformers `5.6.0`, plus the transitive
  stack). This clears the Dependabot advisories for `torch` and `idna`.

## 1.0.0

This is a substantial modernization release with breaking API and dependency changes. Pin `ucluster<1.0` if you need the old behavior.

### Removed

- **`TransformerCluster`** has been removed. `FuzzyClusterer` is now the transformer-based clusterer; there is no separate class. Update imports: `from ucluster import TransformerCluster` → `from ucluster import FuzzyClusterer`.
- **`tf-cluster`** VisiData command has been removed (redundant with `fuzzy-cluster`).
- **FastText** is no longer a dependency. The "train word vectors from scratch on each run" pipeline is gone. If you depended on per-corpus FastText training, pin `ucluster<1.0`.

### Changed

- **`FuzzyClusterer`** now uses the [`paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) sentence-transformer model instead of training FastText from scratch. The model is downloaded from Hugging Face Hub on first use (~470MB) and cached. First run now requires network access.
- **`FuzzyClusterer.outlier_probabilities()`** semantics have changed. It previously returned `hdbscan.HDBSCAN.outlier_scores_` (the GLOSH algorithm — per-point outlier-ness even for in-cluster members, effectively unbounded). It now returns `1 - probabilities_` (bounded in `[0, 1]`, with `1.0` for noise points). **The numbers are not comparable to the old GLOSH scores.**
- **`FuzzyClusterer.__init__`** signature changed: the `dims` parameter (FastText embedding dimension) is gone. A new `model` parameter accepts a sentence-transformer model name or path.
- **HDBSCAN** now comes from `sklearn.cluster.HDBSCAN` (upstreamed in scikit-learn 1.3) instead of the standalone `hdbscan` package. Same algorithm, no separate Cython wheel to compile.
- **VisiData** dependency bumped from `^2.11` to `>=3.0`. The plugin works on both 2.x and 3.x runtime, but the lockfile pins 3.x.
- **NLTK punkt resource** is now downloaded as `punkt_tab` (NLTK 3.9+ split). Existing `~/nltk_data/tokenizers/punkt` installs will trigger a one-time re-download.
- **Python** floor raised from `^3.10` (which was equivalent to `>=3.10,<4`) to a plain `>=3.10`. No upper bound.
- **Packaging** migrated from old-style Poetry (`[tool.poetry]`) to PEP 621 (`[project]`) with `hatchling` as the build backend. Use `uv sync` instead of `poetry install`.

### Removed installation pain

- The Mac M-series + OpenBLAS + Conda dance is gone. `uv sync` works out of the box on Apple Silicon and x86 because `scikit-learn`, `torch`, and `sentence-transformers` all ship pre-built wheels.
- `env.yml` (Conda environment) has been deleted.
- `poetry.lock` has been deleted; `uv.lock` takes its place.

### Plugin

- VisiData plugin bumped to `2.0.0` to reflect the removed `tf-cluster` command.
