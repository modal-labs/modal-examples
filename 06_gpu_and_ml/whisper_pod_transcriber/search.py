import dataclasses
import json
import pathlib
from typing import Any

from . import podcast


@dataclasses.dataclass
class SearchRecord:
    title: str
    text: str


def search_transcripts(
    search_dict_path: pathlib.Path,
    query: str,
    items: list[podcast.EpisodeMetadata],
):
    query_parts = query.lower().strip().split()
    print(f"loading search dictionary from {search_dict_path}")
    with open(search_dict_path, "r") as f:
        search_dict = json.load(f)

    n = len(items)
    scores = []
    for i, sd in enumerate(search_dict):
        score = sum(sd.get(q, 0) for q in query_parts)
        if score == 0:
            continue  # no match whatsoever, don't include
        score += (
            1.0 * (n - i) / n
        )  # give a small boost to more recent episodes (low index)
        scores.append((score, items[i]))
    # Sort descending, best scores first.
    scores.sort(reverse=True, key=lambda x: x[0])
    return scores


def calculate_tfidf_features(
    records: list[SearchRecord],
    max_features: int = 5000,
    max_df: float = 1.0,
    min_df: int = 3,
):
    """
    Compute tfidf features with scikit learn.
    """
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer

    v = TfidfVectorizer(
        input="content",
        encoding="utf-8",
        decode_error="replace",
        strip_accents="unicode",
        lowercase=True,
        analyzer="word",
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z_][a-zA-Z0-9_-]+\b",
        ngram_range=(1, 1),
        max_features=max_features,
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True,
        max_df=max_df,
        min_df=min_df,
    )
    corpus = [(a.title + ". " + a.text) for a in records]
    X = v.fit_transform(corpus)
    X = np.asarray(X.astype(np.float32).todense())
    print("tfidf calculated array of shape ", X.shape)
    return X, v


def calculate_sim_dot_product(X, ntake=40):
    """
    Take `X` (N,D) features and for each index return closest `ntake` indices via dot product.
    """
    from numpy import np

    S = np.dot(X, X.T)
    IX = np.argsort(S, axis=1)[
        :, : -ntake - 1 : -1
    ]  # take last ntake sorted backwards
    return IX.tolist()


def calculate_similarity_with_svm(X, ntake=40):
    """
    Take X (N,D) features and for each index return closest `ntake` indices using exemplar SVM.
    """
    import numpy as np
    import sklearn.svm
    from tqdm import tqdm

    n, d = X.shape
    ntake = min(ntake, n)  # Cannot take more than is available
    IX = np.zeros((n, ntake), dtype=np.int64)
    print(f"training {n} svms for each paper...")
    for i in tqdm(range(n)):
        # set all examples as negative except this one
        y = np.zeros(X.shape[0], dtype=np.float32)
        y[i] = 1
        # train an SVM
        clf = sklearn.svm.LinearSVC(
            class_weight="balanced",
            verbose=False,
            max_iter=10000,
            tol=1e-4,
            C=0.1,
        )
        clf.fit(X, y)
        s = clf.decision_function(X)
        ix = np.argsort(s)[
            : -ntake - 1 : -1
        ]  # take last ntake sorted backwards
        IX[i] = ix
    return IX.tolist()


def build_search_index(records: list[SearchRecord], v):
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    # construct a reverse index for supporting search
    vocab = v.vocabulary_
    idf = v.idf_
    punc = "'!\"#$%&'()*+,./:;<=>?@[\\]^_`{|}~'"  # removed hyphen from string.punctuation
    trans_table = {ord(c): None for c in punc}

    def makedict(s, forceidf=None):
        words = set(s.lower().translate(trans_table).strip().split())
        words = set(
            w for w in words if len(w) > 1 and (w not in ENGLISH_STOP_WORDS)
        )
        idfd = {}
        for w in words:
            if forceidf is None:
                if w in vocab:
                    idfval = idf[vocab[w]]  # we have a computed idf for this
                else:
                    idfval = (
                        1.0  # some word we don't know; assume idf 1.0 (low)
                    )
            else:
                idfval = forceidf
            idfd[w] = idfval
        return idfd

    def merge_dicts(dict_list: list[dict]):
        m: dict[str, Any] = {}
        for d in dict_list:
            for key, val in d.items():
                m[key] = m.get(key, 0) + val
        return m

    search_dict = []
    for p in records:
        dict_title = makedict(p.title, forceidf=10)
        dict_summary = makedict(p.text)
        qdict = merge_dicts([dict_title, dict_summary])
        search_dict.append(qdict)

    return search_dict
