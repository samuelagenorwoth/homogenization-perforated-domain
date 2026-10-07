"""Linear (P1) finite elements with variable coefficients.

assemble(nodes, tri, A, f) returns the sparse matrices
    K[i, j] = int grad(phi_i) . A(x) grad(phi_j)     (stiffness, A a 2x2 diagonal matrix)
    M[i, j] = int phi_i phi_j                        (mass)
and the vector b[i] = int f(x) phi_i. The integrals of A and f over each
triangle use a 6-point quadrature rule (exact for polynomials of degree 4),
so the oscillating coefficients are integrated accurately.
"""
import numpy as np
import scipy.sparse as sp

# Dunavant 6-point rule on the reference triangle: barycentric points and weights
_a, _b = 0.445948490915965, 0.091576213509771
_BARY = np.array([[_a, _a, 1 - 2 * _a], [_a, 1 - 2 * _a, _a], [1 - 2 * _a, _a, _a],
                  [_b, _b, 1 - 2 * _b], [_b, 1 - 2 * _b, _b], [1 - 2 * _b, _b, _b]])
_W = np.array([0.223381589678011] * 3 + [0.109951743655322] * 3)


def geometry(nodes, tri):
    """Areas, basis-function gradients and quadrature points of each triangle."""
    p = nodes[tri]
    x, y = p[:, :, 0], p[:, :, 1]
    area = 0.5 * ((x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0]))
    g = np.empty((len(tri), 3, 2))
    for i, j, k in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
        g[:, i, 0] = (y[:, j] - y[:, k]) / (2 * area)
        g[:, i, 1] = (x[:, k] - x[:, j]) / (2 * area)
    qp = np.einsum("qk,tkd->tqd", _BARY, p)           # T x 6 x 2 quadrature points
    return area, g, qp


def assemble(nodes, tri, A=None, f=None):
    """A(x, y) -> (a11, a22) arrays; f(x, y) -> array. Either may be None."""
    area, g, qp = geometry(nodes, tri)
    P = len(nodes)
    rows = np.repeat(tri, 3, axis=1).ravel()
    cols = np.tile(tri, (1, 3)).ravel()
    out = {}
    if A is not None:
        a11, a22 = A(qp[..., 0], qp[..., 1])
        m11 = area * (a11 @ _W)                       # integral of a11 over each triangle
        m22 = area * (a22 @ _W)
        Ke = (m11[:, None, None] * g[:, :, None, 0] * g[:, None, :, 0] +
              m22[:, None, None] * g[:, :, None, 1] * g[:, None, :, 1])
        out["K"] = sp.csr_matrix((Ke.ravel(), (rows, cols)), shape=(P, P))
    Me = area[:, None, None] / 12 * (np.ones((3, 3)) + np.eye(3))
    out["M"] = sp.csr_matrix((Me.ravel(), (rows, cols)), shape=(P, P))
    if f is not None:
        fq = f(qp[..., 0], qp[..., 1])                # T x 6
        be = area[:, None] * ((fq * _W) @ _BARY)      # int f phi_i for the 3 corners
        out["b"] = np.bincount(tri.ravel(), weights=be.ravel(), minlength=P)
    out["area"] = area
    return out
