"""The original problem (P_eps) on the unit square with N x N holes (eps = 1/N):

    -div(A(x/eps) grad u) + u = f   in the perforated domain
    A(x/eps) grad u . n = 0         on the holes (no flux)
    u = 0                           on the outer boundary

Weak form: int A grad u . grad v + int u v = int f v. The no-flux condition on
the holes needs no work: it is the natural condition of this weak form.
"""
import numpy as np
from scipy.sparse.linalg import spsolve
from mesh import perforated_mesh
from fem import assemble
from problem import A_eps, f1, f2

SOURCES = {"f1": f1, "f2": f2}


def mesh_size(N, per_cell=20):
    """Mesh size: per_cell triangles across one cell (and at most 0.02).

    The mesh must be fine compared with the CELL, not just the domain: with
    only 10 triangles per cell, the error around each hole stays the same for
    every N and the solution misses the homogenized limit by several percent.
    """
    return min(1.0 / (per_cell * N), 0.02)


def outer_boundary(nodes, tol=1e-9):
    x, y = nodes[:, 0], nodes[:, 1]
    return np.flatnonzero((x < tol) | (x > 1 - tol) | (y < tol) | (y > 1 - tol))


def solve_eps(N, source="f2", h=None):
    eps = 1.0 / N
    h = h or mesh_size(N)
    nodes, tri = perforated_mesh(N, h)
    sys = assemble(nodes, tri, A=A_eps(eps), f=SOURCES[source](eps))
    Amat = (sys["K"] + sys["M"]).tocsr()
    free = np.setdiff1d(np.arange(len(nodes)), outer_boundary(nodes))
    u = np.zeros(len(nodes))
    u[free] = spsolve(Amat[free][:, free].tocsc(), sys["b"][free])
    return {"N": N, "nodes": nodes, "tri": tri, "u": u, "M": sys["M"], "h": h}


if __name__ == "__main__":
    import time
    for source in ["f2", "f1"]:
        print(f"source {source}:")
        for N in [2, 4, 8, 16]:
            t = time.time()
            s = solve_eps(N, source)
            print(f"  N = {N:2d}  h = {s['h']:.4f}  nodes = {len(s['nodes']):7d}  "
                  f"min u = {s['u'].min():+.5f}  max u = {s['u'].max():+.5f}  ({time.time()-t:.1f}s)")
