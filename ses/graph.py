import pickle, re
from collections import defaultdict

import numpy as np
import scipy.sparse as sp

from .llm import CACHE, EMBED_MODEL

SPAN = re.compile(r"[A-Z][\w'’.\-]*(?:\s+(?:of|the|de|von|van|and)?\s*[A-Z][\w'’.\-]*)*")
STOP = {"the", "a", "an", "he", "she", "it", "in", "on", "his", "her", "they", "this", "that", "who", "what",
        "which", "when", "where", "how", "is", "are", "was", "were", "did", "does", "do", "both", "after", "before"}
W_MENTION, W_SHARED, W_KNN, W_MIN = 0.5, 0.2, 0.3, 0.1


def norm_title(t):
    return re.sub(r"\s*\(.*?\)\s*", " ", t).strip().lower()


def spans(text):
    out = set()
    for s in SPAN.findall(text):
        words = s.split()
        while words and words[0].lower() in STOP:
            words = words[1:]
        if words:
            out.add(" ".join(words).lower().rstrip(".'’"))
    return out


class Graph:
    def __init__(self, W, ents, title2pids):
        self.W, self.ents, self.title2pids = W.tocsr(), ents, title2pids
        self.nbrs = [(W.indices[W.indptr[i]:W.indptr[i + 1]], W.data[W.indptr[i]:W.indptr[i + 1]]) for i in range(W.shape[0])]


def build(pids, E, k=6, max_df=20, w_mention=W_MENTION, tag=""):
    path = CACHE / f"graph_{EMBED_MODEL}{tag}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    n = len(pids)
    title2pids = defaultdict(list)
    for i, (t, _) in enumerate(pids):
        title2pids[norm_title(t)].append(i)
    ents = [spans(t + ". " + x) for t, x in pids]

    mention = defaultdict(float)
    inv = defaultdict(list)
    for i, es in enumerate(ents):
        for e in es:
            inv[e].append(i)
            for j in title2pids.get(e, ()):
                if j != i:
                    mention[tuple(sorted((i, j)))] = 1.0
    shared = defaultdict(float)
    for e, ids in inv.items():
        if 1 < len(ids) <= max_df:
            for a in range(len(ids)):
                for b in range(a + 1, len(ids)):
                    shared[(ids[a], ids[b])] += 1 / 3

    knn = {}
    for s in range(0, n, 2000):
        sims = E[s:s + 2000] @ E.T
        for r in range(sims.shape[0]):
            sims[r, s + r] = -1
        top = np.argpartition(-sims, k, axis=1)[:, :k]
        for r in range(sims.shape[0]):
            for j in top[r]:
                knn[tuple(sorted((s + r, int(j))))] = float(sims[r, j])
    lo, hi = min(knn.values()), max(knn.values())

    W = defaultdict(float)
    for (a, b), v in mention.items():
        W[(a, b)] += w_mention * v
    for (a, b), v in shared.items():
        W[(a, b)] += W_SHARED * min(v, 1.0)
    for (a, b), v in knn.items():
        W[(a, b)] += W_KNN * (v - lo) / (hi - lo)
    rows, cols, vals = zip(*[(a, b, w) for (a, b), w in W.items() if w >= W_MIN])
    M = sp.coo_matrix((vals, (rows, cols)), shape=(n, n))
    g = Graph(M + M.T, ents, dict(title2pids))
    path.write_bytes(pickle.dumps(g))
    return g
