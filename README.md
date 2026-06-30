# uCluster
uCluster is a simple and fast way to cluster textual content. It is available both as a standalone Python package as well as a [VisiData](https://visidata.org) plugin. This documentation will focus on using uCluster as a VisiData plugin; to see how to use it as a Python package, simply inspect [`ucluster/text_cluster.py`](ucluster/text_cluster.py) — it's a very simple interface.

## Motivation

Suppose you have a dataset of millions of text posts. That's great! But there's no way you're going to be able to sift through each post one-by-one. Sure, you can do an emoji flag analysis to get a sense of national identities; similarly, you can look at n-grams to get a sense of popular phrases. These are both useful approaches, but they don't give you a great sense for *communities* of content. Are there a bunch of posts in the dataset that are essentially saying the same thing?

uCluster uses (relatively simple) NLP techniques to *cluster* datasets of text. It looks for similarities in the posts, and groups related posts — for some rough understanding of "related" — together. Not every post will be assigned to a cluster, and not every cluster will be a perfect grouping. As an exploratory tool, however, uCluster can be quite powerful. If a lot of posts are saying essentially the same thing — even with small variations in phrasing, punctuation, emojis, etc. — uCluster will notice.

## Interpreting Clusters

When you run uCluster on your text dataset, each piece of text will either be assigned a cluster (identified by a number >= 0), no cluster (identified by the number -1). You can think of each cluster as containing "similar" posts — whether that similarity arises from using one particularly rare word (or hashtag), from using a similar sentence structure, or from something else. You can probably get an intuition for "why" each cluster is as it is from just looking at its content.

There is no pre-set number of clusters, as clusters are determined "naturally" by looking at density. For more information about how this clustering works, see the "architecture" section below.

**Note:** uCluster also supports _exact clustering_, in which it doesn't perform any NLP magic and simply creates clusters based on exact matches. For example, if there are multiple posts with the exact same content (case insensitive), they will be assigned to the same cluster. You can access this clusterer through the `exact-cluster` VisiData command, or through the `ExactClusterer` class in [`ucluster/text_cluster.py`](ucluster/text_cluster.py). Most of this document is referring to the more advanced "fuzzy" clusterer. 

## Words of Warning

uCluster is not a magic tool, and it's important to be mindful of its inherent limitations.

1. **Clusters aren't significant by themselves.** The clusters exist only to point you towards potentially related content — there is nothing significant about the clusters themselves. For example, if two users' posts frequently cluster together, that is not a sign of coordinated inauthentic behavior; it's simply a sign that you might want to investigate those users (and those clusters) further.

2. **It handles English best (but it's still somewhat multilingual).** uCluster (and the tools it uses under the hood) makes several assumptions about the input text, which tend to be most accurate for English text, usually accurate for latin-charactered space-separated languages (e.g., Portuguese), kind of accurate for other space-separated languages (e.g., Russian), and least accurate for logograph-based languages (e.g., Chinese). You can use uCluster for multilingual datasets (this is one of its key design goals), but don't be surprised if you end up with a cluster that is characterized not by content, per se, but by language (e.g., a cluster that contains all the Arabic posts in your dataset).

3. **It uses a pre-trained multilingual sentence-transformer model.** uCluster downloads `paraphrase-multilingual-MiniLM-L12-v2` (~470MB) from the Hugging Face Hub on first use and caches it under `~/.cache/huggingface/`. **First run requires network access** — air-gapped environments need to pre-populate the cache or pass a local path via `FuzzyClusterer(model="/path/to/model")`. After the model is cached, embedding millions of posts is CPU/GPU-bound rather than memory-bound. If a GPU is available, sentence-transformers will use it automatically.

4. **It assumes posts are relatively short.** uCluster works best on posts that are relatively short (think tweet-length). That makes it great for datasets from platforms like Twitter, Gab, Gettr, and Parler, where posts tend to be only a few sentences. It's going to be much less effective for classifying, say, Medium posts.

5. **It doesn't perform any meaningful pre-processing.** uCluster doesn't strip HTML tags, links, or @mentions, for example. Be sure to clean your dataset to make it as close to "pure" text content as possible before running it through uCluster. (Don't strip emojis, though — anecdotally, these are very helpful for clustering. They carry a lot of meaning!)

## VisiData Usage

When installed in VisiData, the uCluster plugin adds two commands that operate on columns: `fuzzy-cluster` and `exact-cluster`. Simply select the column that contains the text you would like to cluster, press space, then type `fuzzy-cluster` or `exact-cluster`. uCluster will then create a new column that will contain the cluster IDs. This can take several minutes. (Don't worry — it's done asynchronously, though VisiData still seems to occasionally freeze on giant datasets.)

## Installation

For most people installing uCluster is as easy as running `pip3 install ucluster`, then adding `import ucluster.vd.plugin` to your `~/.visidatarc` file. If you want to install uCluster in a way that allows local development, follow the "Development Installation" steps below.

### Development Installation

Clone uCluster (`git clone git@github.com:stanfordio/ucluster.git`), then run:

```sh
uv sync --extra dev
```

That installs uCluster, its runtime dependencies, and the dev tooling (`pytest`, `ruff`, `ty`) into a managed `.venv`. There's no longer a need for Conda, Homebrew packages, or environment variables — pre-built wheels for `scikit-learn`, `torch`, and `sentence-transformers` cover x86 and Apple Silicon out of the box.

To install uCluster in your global VisiData environment instead, use:

```sh
uv pip install --system -e .
```

Then add `import ucluster.vd.plugin` to your `~/.visidatarc` as described above.

### Testing

Run the test suite with:

```sh
uv run pytest
```

The fuzzy-clustering tests run against a deterministic fake encoder, so the
default run needs no network access. One integration test exercises the real
sentence-transformer model; it is marked `slow` and skipped by default. Run it
explicitly (downloads the model on first use) with:

```sh
uv run pytest -m slow
```

## In The Weeds: Architecture & Design

This section will eventually contain a more detailed explanation for how uCluster works. For now, here's a brief overview.

* First, we encode each post into a 384-dimensional vector using the [`paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) sentence-transformer model. This model was trained on parallel sentences across 50+ languages, so semantically similar posts end up near each other in vector space regardless of language.
* Then we run [HDBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html) (via scikit-learn) on those post vectors to create the final clusters. HDBSCAN is a density-based clustering algorithm that automatically determines the number of clusters from the data.
