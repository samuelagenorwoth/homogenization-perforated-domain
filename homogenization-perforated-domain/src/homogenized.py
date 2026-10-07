"""The homogenized (upscaled) problem (P_0) on the unit square:

    -div(D grad u0) + theta u0 = F   in (0,1)^2,    u0 = 0 on the boundary

with D and theta from the cell problems (cell.py) and F the limit of chi * f:
    f2 = cos(2 pi x1) + eps sin(2 pi x1/eps)  ->  F = theta cos(2 pi x1)
    f1 = cos(2 pi x1/eps)                      ->  F = int_{Y*} cos(2 pi y1) dy   (a constant)
"""
import numpy as np
from scipy.sparse.linalg import spsolve
from mesh import perforated_mesh
from fem import assemble
from cell import solve_cell
from eps_problem import outer_boundary

_EFFECTIVE = None


def effective(h_cell=0.0125):
    """D, theta and the f1 average from the cell problems (computed once)."""
    global _EFFECTIVE
    if _EFFECTIVE is None:
        c = solve_cell(h=h_cell)
        _EFFECTIVE = {"D": c["D"], "theta": c["theta"], "f1_hom": c["f1_hom"]}
    return _EFFECTIVE


def solve_hom(source="f2", h=0.01):
    e = effective()
    D, theta = e["D"], e["theta"]
    if source == "f2":
        F = lambda x1, x2: theta * np.cos(2 * np.pi * x1)
    else:
        F = lambda x1, x2: e["f1_hom"] * np.ones_like(x1)
    nodes, tri = perforated_mesh(0, h)                  # N = 0: the square without holes
    sys = assemble(nodes, tri, A=lambda x1, x2: (D[0, 0] * np.ones_like(x1),
                                                 D[1, 1] * np.ones_like(x1)), f=F)
    Amat = (sys["K"] + theta * sys["M"]).tocsr()
    free = np.setdiff1d(np.arange(len(nodes)), outer_boundary(nodes))
    u = np.zeros(len(nodes))
    u[free] = spsolve(Amat[free][:, free].tocsc(), sys["b"][free])
    return {"nodes": nodes, "tri": tri, "u": u, "M": sys["M"]}


if __name__ == "__main__":
    e = effective()
    print(f"D = {e['D'][0,0]:.4f}, theta = {e['theta']:.4f}, f1 average = {e['f1_hom']:.4f}")
    for source in ["f2", "f1"]:
        for h in [0.02, 0.01]:
            s = solve_hom(source, h)
            print(f"source {source}, h = {h}: min u0 = {s['u'].min():+.5f}  max u0 = {s['u'].max():+.5f}")
