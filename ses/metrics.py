import re, string
from collections import Counter

import numpy as np


def norm(s):
    s = s.lower()
    s = "".join(c for c in s if c not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def em(pred, gold):
    return float(norm(pred) == norm(gold))


def f1(pred, gold):
    p, g = norm(pred).split(), norm(gold).split()
    common = sum((Counter(p) & Counter(g)).values())
    if not common:
        return 0.0
    pr, rc = common / len(p), common / len(g)
    return 2 * pr * rc / (pr + rc)


def abstained(pred):
    return "insufficient" in pred.lower()


def paired_boot(a, b, n=5000, seed=0):
    """Mean difference a-b with 95% CI and one-sided p(a<=b)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return d.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5), float((means <= 0).mean())
