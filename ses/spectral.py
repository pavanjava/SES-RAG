import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh


def lambda2(A, normalized=True):
    """Second-smallest Laplacian eigenvalue of a dense symmetric weight matrix. 0 if fragmented."""
    n = len(A)
    if n < 2:
        return 0.0
    d = A.sum(1)
    if normalized:
        inv = np.where(d > 0, np.where(d > 0, d, 1.0) ** -0.5, 0.0)
        L = np.diag((d > 0).astype(float)) - inv[:, None] * A * inv[None, :]
    else:
        L = np.diag(d) - A
    return float(max(np.linalg.eigvalsh(L)[1], 0.0))


def spectral_coords(W, k=32, giant=False):
    """Eigenvectors 1..k of the normalised Laplacian of the full chunk graph (SpectRAG-style)."""
    if giant:
        _, lab = connected_components(W, directed=False)
        keep = np.flatnonzero(lab == np.bincount(lab).argmax())
        out = np.zeros((W.shape[0], k))
        out[keep] = spectral_coords(W[keep][:, keep], k)
        return out
    n = W.shape[0]
    d = np.asarray(W.sum(1)).ravel()
    inv = sp.diags(np.where(d > 0, np.where(d > 0, d, 1.0) ** -0.5, 0.0))
    L = sp.identity(n) - inv @ W @ inv
    _, vecs = eigsh(L, k=k + 1, sigma=-0.01, which="LM")
    X = vecs[:, 1:]
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
