"""The cell problems and the effective (homogenized) coefficients.

On the perforated cell Y* (the unit square minus the hole) find periodic
w_k, k = 1, 2, with zero average, such that

    int_{Y*} A(y) (e_k + grad w_k) . grad v dy = 0     for all periodic v,

i.e.  -div(A (e_k + grad w_k)) = 0 in Y*, no flux through the hole.
(The report writes u_1 = -W grad u_0, so W = -w.) Then

    D[i, k] = int_{Y*} (A (e_k + grad w_k))_i dy      effective diffusion
    theta   = |Y*|                                     volume fraction
    f1_hom  = int_{Y*} cos(2 pi y1) dy                 limit of chi * f1
(the cell has area |Y| = 1).
"""
import numpy as np
from scipy.sparse.linalg import spsolve
from mesh import cell_mesh
from fem import assemble, geometry
from problem import A_cell, R_HOLE


def periodic_map(nodes):
    """Number the unknowns so that matching nodes on opposite sides share one."""
    tol = 1e-8
    x, y = nodes[:, 0], nodes[:, 1]
    key = np.round(np.column_stack([np.where(abs(x - 1) < tol, 0.0, x),
                                    np.where(abs(y - 1) < tol, 0.0, y)]) / tol).astype(np.int64)
    _, dof = np.unique(key, axis=0, return_inverse=True)
    return dof.ravel()


def solve_cell(r=R_HOLE, h=0.02, A=A_cell):
    nodes, tri = cell_mesh(r, h)
    dof = periodic_map(nodes)
    n = dof.max() + 1
    P = len(nodes)
    import scipy.sparse as sp
    Pm = sp.csr_matrix((np.ones(P), (np.arange(P), dof)), shape=(P, n))   # periodic -> nodes
    K = assemble(nodes, tri, A=A)["K"]
    area, g, qp = geometry(nodes, tri)
    from fem import _W
    a11, a22 = A(qp[..., 0], qp[..., 1])
    m = np.stack([area * (a11 @ _W), area * (a22 @ _W)], axis=1)          # int_T A_kk
    Kp = (Pm.T @ K @ Pm).tocsr()
    D = np.zeros((2, 2))
    W = []
    for k in range(2):
        # right-hand side: - int A e_k . grad(phi_i) = - sum_T (int_T A_kk) g_i[k]
        be = -m[:, k][:, None] * g[:, :, k]
        b = np.bincount(tri.ravel(), weights=be.ravel(), minlength=P)
        bp = Pm.T @ b
        # w is fixed only up to a constant: pin one value, then remove the mean
        w = np.zeros(n)
        w[1:] = spsolve(Kp[1:, 1:].tocsc(), bp[1:])
        wn = Pm @ w
        mean = (area[:, None] * wn[tri]).sum() / 3 / area.sum()
        wn -= mean
        W.append(wn)
        grad_w = np.einsum("ti,tid->td", wn[tri], g)                     # constant per triangle
        for i in range(2):
            D[i, k] = ((i == k) * m[:, i] + m[:, i] * grad_w[:, i]).sum()
    theta = area.sum()
    f1_hom = (area * (np.cos(2 * np.pi * qp[..., 0]) @ _W)).sum()
    return {"D": D, "theta": theta, "f1_hom": f1_hom, "nodes": nodes, "tri": tri,
            "W": W, "unknowns": n}


if __name__ == "__main__":
    print("Check without a hole (exact answer: D = 2*sqrt(2) I = 2.828427 I, theta = 1):")
    for h in [0.05, 0.025, 0.0125]:
        c = solve_cell(r=0.0, h=h)
        print(f"  h = {h:<7} D = [[{c['D'][0,0]:.6f}, {c['D'][0,1]:+.1e}], "
              f"[{c['D'][1,0]:+.1e}, {c['D'][1,1]:.6f}]]   theta = {c['theta']:.6f}")
    print("\nCheck with a constant coefficient A = I and the hole (Rayleigh's formula"
          " for a square array of holes, Perrins et al. 1979):")
    phi = np.pi * R_HOLE**2
    rayleigh = 1 - 2 * phi / (1 + phi - 0.305827 * phi**4)
    for h in [0.05, 0.025, 0.0125]:
        c = solve_cell(h=h, A=lambda y1, y2: (np.ones_like(y1), np.ones_like(y2)))
        print(f"  h = {h:<7} D11 = {c['D'][0,0]:.6f}   (formula: {rayleigh:.6f})")
    print(f"\nWith the hole, radius {R_HOLE} (exact theta = 1 - pi/16 = {1 - np.pi/16:.6f}):")
    for h in [0.05, 0.025, 0.0125]:
        c = solve_cell(h=h)
        print(f"  h = {h:<7} D = [[{c['D'][0,0]:.6f}, {c['D'][0,1]:+.1e}], "
              f"[{c['D'][1,0]:+.1e}, {c['D'][1,1]:.6f}]]   theta = {c['theta']:.6f}"
              f"   f1_hom = {c['f1_hom']:.6f}")
