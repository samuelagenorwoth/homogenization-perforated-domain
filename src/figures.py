"""Figures for the README (saved in ../figures).

  cell-problems.png   the two cell solutions w1, w2 and the effective D
  solutions.png       u_eps for N = 4, 8, 16 next to u0, for both sources
  convergence.png     relative L2 error against eps (needs results/convergence.npz)
  homogenization.gif  the holes multiply and u_eps settles towards u0 (for the website)
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.tri import Triangulation
from PIL import Image
from cell import solve_cell
from eps_problem import solve_eps
from homogenized import solve_hom

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"
INK, TEAL, ORANGE = "#15222E", "#0B6E78", "#D9822B"
CMAP = LinearSegmentedColormap.from_list("site", [TEAL, "white", ORANGE])
plt.rcParams.update({"font.size": 11, "text.color": INK,
                     "axes.labelcolor": INK, "axes.titlecolor": INK})


def draw(ax, sol, vmax):
    T = Triangulation(sol["nodes"][:, 0], sol["nodes"][:, 1], sol["tri"])
    im = ax.tripcolor(T, sol["u"], shading="gouraud", cmap=CMAP, vmin=-vmax, vmax=vmax)
    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color=INK, lw=0.8)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def cell_problems():
    c = solve_cell(h=0.0125)
    T = Triangulation(c["nodes"][:, 0], c["nodes"][:, 1], c["tri"])
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.3), constrained_layout=True)
    for k, ax in enumerate(axes):
        w = c["W"][k]; m = abs(w).max()
        im = ax.tripcolor(T, w, shading="gouraud", cmap=CMAP, vmin=-m, vmax=m)
        ax.set_aspect("equal")
        ax.set_title(f"Cell solution $w_{k+1}$ (direction $e_{k+1}$)")
        ax.set_xlabel("$y_1$"); ax.set_ylabel("$y_2$")
        fig.colorbar(im, ax=ax, shrink=0.85)
    D = c["D"]
    fig.suptitle(rf"Cell problems: $\mathbb{{D}} \approx {D[0,0]:.4f}\, I$,  "
                 rf"$\theta \approx {c['theta']:.4f}$")
    fig.savefig(FIGURES / "cell-problems.png", dpi=160)
    plt.close(fig)


def solutions():
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.6), constrained_layout=True)
    names = {"f2": r"$f_2 = \cos(2\pi x_1) + \varepsilon \sin(2\pi x_1/\varepsilon)$",
             "f1": r"$f_1 = \cos(2\pi x_1/\varepsilon)$"}
    for row, src in enumerate(["f2", "f1"]):
        sols = [solve_eps(N, src) for N in [4, 8, 16]] + [solve_hom(src, 0.01)]
        m = max(abs(s["u"]).max() for s in sols)
        for ax, s, title in zip(axes[row], sols,
                                [r"$u_\varepsilon$, N = 4", "N = 8", "N = 16", r"homogenized $u_0$"]):
            im = draw(ax, s, m)
            ax.set_title(title)
        axes[row, 0].set_ylabel(names[src], fontsize=12)
        fig.colorbar(im, ax=axes[row], shrink=0.8)
    fig.suptitle(r"As the holes multiply, $u_\varepsilon$ approaches the homogenized solution $u_0$")
    fig.savefig(FIGURES / "solutions.png", dpi=110)
    plt.close(fig)


def convergence():
    d = np.load(ROOT / "results" / "convergence.npz")
    eps = 1.0 / d["N"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    ax.loglog(eps, d["err_f2"], "o-", color=TEAL, lw=2, label=r"source $f_2$")
    ax.loglog(eps, d["err_f1"], "s-", color=ORANGE, lw=2, label=r"source $f_1$")
    ax.loglog(eps, d["err_f2"][-1] * eps / eps[-1], ":", color=INK, label=r"slope 1: error $\propto \varepsilon$")
    ax.set_xlabel(r"$\varepsilon = 1/N$  (cell size)")
    ax.set_ylabel(r"$\|u_\varepsilon - u_0\|_{L^2} \,/\, \|u_0\|_{L^2}$")
    ax.set_title(r"The difference between $u_\varepsilon$ and $u_0$ shrinks like $\varepsilon$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    for N, e in zip(d["N"], d["err_f2"]):
        ax.annotate(f"N = {N}", (1 / N, e), textcoords="offset points", xytext=(6, -12), fontsize=9)
    fig.savefig(FIGURES / "convergence.png", dpi=170)
    plt.close(fig)


def gif():
    hom = solve_hom("f2", 0.01)
    sols = [solve_eps(N, "f2") for N in [2, 4, 8, 16]]
    m = max(abs(s["u"]).max() for s in sols + [hom])
    frames = []
    for N, s in zip([2, 4, 8, 16], sols):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4.5), dpi=80, constrained_layout=True)
        draw(axes[0], s, m); axes[0].set_title(f"{N} x {N} holes", fontsize=14)
        draw(axes[1], hom, m); axes[1].set_title("homogenized", fontsize=14)
        fig.canvas.draw()
        rgb = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3])
        frames.append(rgb.quantize(colors=64, dither=Image.Dither.NONE))
        plt.close(fig)
    frames[0].save(FIGURES / "homogenization.gif", save_all=True, append_images=frames[1:],
                   duration=1300, loop=0, optimize=True)


def main():
    FIGURES.mkdir(exist_ok=True)
    cell_problems()
    solutions()
    convergence()
    gif()
    print(f"Figures saved in {FIGURES}")


if __name__ == "__main__":
    main()
