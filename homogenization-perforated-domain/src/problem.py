"""The data of the problem, as in the 2024 report.

    A(y) = diag(3 + cos(2 pi y1), 3 + cos(2 pi y2))      (period 1 in y = x/eps)
    hole radius r = 1/4 in cell units (eps/4 in the domain)
    f1(x) = cos(2 pi x1 / eps)
    f2(x) = cos(2 pi x1) + eps sin(2 pi x1 / eps)
"""
import numpy as np

R_HOLE = 0.25


def A_cell(y1, y2):
    return 3 + np.cos(2 * np.pi * y1), 3 + np.cos(2 * np.pi * y2)


def A_eps(eps):
    return lambda x1, x2: A_cell(x1 / eps, x2 / eps)


def f1(eps):
    return lambda x1, x2: np.cos(2 * np.pi * x1 / eps)


def f2(eps):
    return lambda x1, x2: np.cos(2 * np.pi * x1) + eps * np.sin(2 * np.pi * x1 / eps)
