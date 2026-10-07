"""Triangle meshes made with gmsh.

  cell_mesh(r, h)          one periodic cell Y = (0,1)^2 with a hole of radius r
                           in the middle (r = 0: no hole), with matching nodes
                           on opposite sides so periodic conditions can be imposed
  perforated_mesh(N, h)    the unit square with N x N holes of radius 1/(4N)
                           (N = 0: the square without holes)
                           (cell size eps = 1/N), mesh size about h
Both return nodes (P x 2) and triangles (T x 3, counter-clockwise).
"""
import numpy as np
import gmsh


def _mesh(build, h):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("m")
    build()
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.MeshSizeMax", h)
    gmsh.option.setNumber("Mesh.MeshSizeMin", h / 2)
    gmsh.model.mesh.generate(2)
    tags, coords, _ = gmsh.model.mesh.getNodes()
    nodes = coords.reshape(-1, 3)[:, :2]
    _, _, conn = gmsh.model.mesh.getElements(2)
    gmsh.finalize()
    index = -np.ones(int(tags.max()) + 1, dtype=int)
    index[tags.astype(int)] = np.arange(len(tags))
    tri = index[np.asarray(conn[0], dtype=int).reshape(-1, 3)]
    used = np.unique(tri)                      # drop nodes not used by any triangle
    renumber = -np.ones(len(nodes), dtype=int)
    renumber[used] = np.arange(len(used))
    nodes, tri = nodes[used], renumber[tri]
    # make every triangle counter-clockwise
    p = nodes[tri]
    det = (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - \
          (p[:, 1, 1] - p[:, 0, 1]) * (p[:, 2, 0] - p[:, 0, 0])
    tri[det < 0] = tri[det < 0][:, [0, 2, 1]]
    return nodes, tri


def cell_mesh(r, h):
    def build():
        occ = gmsh.model.occ
        square = occ.addRectangle(0, 0, 0, 1, 1)
        if r > 0:
            occ.cut([(2, square)], [(2, occ.addDisk(0.5, 0.5, 0, r, r))])
        occ.synchronize()
        # Periodic mesh: the right side copies the left side, the top copies the bottom
        lines = gmsh.model.getEntities(1)
        def find(test):
            for dim, tag in lines:
                x0, y0, _, x1, y1, _ = gmsh.model.getBoundingBox(dim, tag)
                if test(x0, y0, x1, y1):
                    return tag
        e = 1e-5
        left = find(lambda x0, y0, x1, y1: abs(x0) < e and abs(x1) < e)
        right = find(lambda x0, y0, x1, y1: abs(x0 - 1) < e and abs(x1 - 1) < e)
        bottom = find(lambda x0, y0, x1, y1: abs(y0) < e and abs(y1) < e)
        top = find(lambda x0, y0, x1, y1: abs(y0 - 1) < e and abs(y1 - 1) < e)
        shift_x = [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        shift_y = [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1]
        gmsh.model.mesh.setPeriodic(1, [right], [left], shift_x)
        gmsh.model.mesh.setPeriodic(1, [top], [bottom], shift_y)
    return _mesh(build, h)


def perforated_mesh(N, h):
    eps = 1.0 / N if N else 1.0
    def build():
        occ = gmsh.model.occ
        square = occ.addRectangle(0, 0, 0, 1, 1)
        holes = [(2, occ.addDisk((i + 0.5) * eps, (j + 0.5) * eps, 0, eps / 4, eps / 4))
                 for i in range(N) for j in range(N)]
        if holes:
            occ.cut([(2, square)], holes)
    return _mesh(build, h)
