"""Does u_eps converge to u0 as eps -> 0?

For N = 2, 4, ..., 32 (eps = 1/N), solve the original problem on a mesh with
PER_CELL triangles across each cell, evaluate the homogenized solution u0 at
its nodes, and measure the relative L2 difference over the perforated domain:

    error(eps) = || u_eps - u0 ||_{L2(Omega_eps)} / || u0 ||_{L2(Omega_eps)}

Homogenization theory predicts error = O(eps): halving eps halves the error.
"""
from pathlib import Path
import time
import numpy as np
from matplotlib.tri import Triangulation, LinearTriInterpolator
from eps_problem import solve_eps, mesh_size
from homogenized import solve_hom

RESULTS = Path(__file__).resolve().parent.parent / "results"
NS = [2, 4, 8, 16, 32]
PER_CELL = 30


def l2_error(sol, interp):
    u0 = np.asarray(interp(sol["nodes"][:, 0], sol["nodes"][:, 1]))
    e = sol["u"] - u0
    return np.sqrt(e @ (sol["M"] @ e)) / np.sqrt(u0 @ (sol["M"] @ u0))


def main():
    RESULTS.mkdir(exist_ok=True)
    data = {"N": np.array(NS)}
    for source in ["f2", "f1"]:
        hom = solve_hom(source, h=0.005)
        interp = LinearTriInterpolator(Triangulation(hom["nodes"][:, 0], hom["nodes"][:, 1],
                                                     hom["tri"]), hom["u"])
        errors = []
        print(f"source {source}:")
        for N in NS:
            t = time.time()
            sol = solve_eps(N, source, h=mesh_size(N, PER_CELL))
            errors.append(l2_error(sol, interp))
            rate = "" if len(errors) < 2 else f"   rate {np.log2(errors[-2] / errors[-1]):.2f}"
            print(f"  N = {N:2d}  nodes = {len(sol['nodes']):7d}  relative L2 error = {errors[-1]:.4f}"
                  f"{rate}   ({time.time() - t:.0f}s)", flush=True)
        data[f"err_{source}"] = np.array(errors)
        # How much of the error is the mesh? Repeat N = 16 on a finer mesh.
        sol = solve_eps(16, source, h=mesh_size(16, 40))
        data[f"check16_{source}"] = l2_error(sol, interp)
        print(f"  mesh check, N = 16 with 40 triangles per cell: {data[f'check16_{source}']:.4f}")
    np.savez(RESULTS / "convergence.npz", **data)


if __name__ == "__main__":
    main()
