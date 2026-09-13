"""Exact nearest-neighbour search. Deliberately the simplest thing that works."""
from __future__ import annotations

import numpy as np


class ExactSearch:
    """
    Brute-force cosine similarity over unit-length vectors.

    NO approximate index, NO FAISS, on purpose. At N=1360 this is one matrix
    multiply. Adding an ANN index here would be complexity you cannot justify
    in a review. In week 11 you MEASURE the crossover point where ANN starts
    paying for itself - and being able to say "I measured it and it did not"
    is a stronger answer than having used FAISS because it is standard.

    Complexity: O(N*d) per query, O(N) memory for the matrix.
    At N=1360, d=768, float32 that is about 4 MB.
    """

    def __init__(self, embeddings: np.ndarray, ids: list[int]):
        if embeddings.shape[0] != len(ids):
            raise ValueError(f"{embeddings.shape[0]} vectors but {len(ids)} ids")
        norms = np.linalg.norm(embeddings, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            raise ValueError(
                f"embeddings are not unit length (min={norms.min():.4f}, "
                f"max={norms.max():.4f}). Dot product would not be cosine.")
        self.emb = embeddings.astype(np.float32)
        self.ids = np.asarray(ids)

    def search(self, queries: np.ndarray, k: int = 10):
        """Returns (ranked_ids [Q,k], scores [Q,k])."""
        scores = queries.astype(np.float32) @ self.emb.T        # [Q, N]
        k = min(k, scores.shape[1])
        # argpartition finds the top k without fully sorting all N - O(N)
        # instead of O(N log N). Then sort just those k.
        part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
        ordered = np.take_along_axis(
            part, np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1), axis=1)
        return self.ids[ordered], np.take_along_axis(scores, ordered, axis=1)
