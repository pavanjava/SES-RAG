import numpy as np
from ses.spectral import lambda2
from ses.metrics import em, f1, paired_boot


def path(n):
    A = np.zeros((n, n))
    for i in range(n - 1):
        A[i, i + 1] = A[i + 1, i] = 1
    return A


def test_lambda2_path_raw():
    n = 5
    assert abs(lambda2(path(n), normalized=False) - (2 - 2 * np.cos(np.pi / n))) < 1e-9


def test_lambda2_two_nodes_normalized():
    assert abs(lambda2(path(2)) - 2.0) < 1e-9


def test_lambda2_disconnected_is_zero():
    A = np.zeros((4, 4)); A[0, 1] = A[1, 0] = 1; A[2, 3] = A[3, 2] = 1
    assert lambda2(A) < 1e-9 and lambda2(np.zeros((3, 3))) == 0.0


def test_lambda2_single_node():
    assert lambda2(np.zeros((1, 1))) == 0.0


def test_em_f1():
    assert em("The Eiffel Tower", "eiffel tower") == 1.0
    assert abs(f1("big red dog", "red dog") - 0.8) < 1e-9


def test_paired_boot_sign():
    m, lo, hi, p = paired_boot([1] * 50, [0] * 50)
    assert m == 1 and lo > 0 and p == 0
