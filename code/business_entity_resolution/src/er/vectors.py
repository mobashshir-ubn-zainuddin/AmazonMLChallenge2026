"""
Dense hashed character-n-gram TF-IDF vectors.

texts -> char_wb 3-grams hashed into 2^20 buckets (≈ exact n-gram identity)
      -> sublinear tf * idf (idf fitted on the S1 index)
      -> signed random projection to `dim` dense dims (unbiased inner products)
      -> L2 normalize, float16

Cosine of two vectors ≈ TF-IDF char-3-gram cosine (projection noise std ≈ 1/sqrt(dim)).
Dense vectors let us do EXACT brute-force top-K on the GPU with plain matmuls,
which is far faster than Python posting lists.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer

N_FEATURES = 2 ** 20


class HashedTfidf:
    def __init__(self, dim: int = 1024, ngram=(3, 3), seed: int = 13):
        self.dim = dim
        self.hv = HashingVectorizer(
            analyzer="char_wb", ngram_range=ngram, n_features=N_FEATURES,
            alternate_sign=False, norm=None, lowercase=False, dtype=np.float32,
        )
        rng = np.random.default_rng(seed)
        cols = rng.integers(0, dim, size=N_FEATURES)
        signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=N_FEATURES)
        self.proj = sp.csr_matrix((signs, (np.arange(N_FEATURES), cols)), shape=(N_FEATURES, dim))
        self.idf = np.ones(N_FEATURES, dtype=np.float32)

    def fit(self, texts, chunk: int = 500_000) -> "HashedTfidf":
        df = np.zeros(N_FEATURES, dtype=np.int64)
        n = 0
        for i in range(0, len(texts), chunk):
            X = self.hv.transform(texts[i:i + chunk])
            df += np.bincount(X.indices, minlength=N_FEATURES)
            n += X.shape[0]
        self.idf = (np.log((n + 1) / (df + 1)) + 1.0).astype(np.float32)
        return self

    def transform(self, texts) -> np.ndarray:
        """Return float16 array (len(texts), dim); empty texts give zero rows."""
        X = self.hv.transform(texts)
        X.data = np.log1p(X.data) * self.idf[X.indices]
        D = (X @ self.proj).toarray()
        norms = np.linalg.norm(D, axis=1, keepdims=True)
        np.divide(D, norms, out=D, where=norms > 0)
        return D.astype(np.float16)
