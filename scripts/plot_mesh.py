import sys
import numpy as np
import gmsh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

path = sys.argv[1] if len(sys.argv) > 1 else "results/bracket.msh"
out = sys.argv[2] if len(sys.argv) > 2 else "results/mesh.png"

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.open(path)
ntags, coords, _ = gmsh.model.mesh.getNodes()
xyz = np.zeros((int(np.max(ntags)) + 1, 3))
xyz[np.asarray(ntags, dtype=np.int64)] = np.asarray(coords).reshape(-1, 3)
types, _, nodes = gmsh.model.mesh.getElements(2)
tris = np.vstack([
    np.asarray(nn, dtype=np.int64).reshape(-1, {2: 3, 9: 6}[int(et)])[:, :3]
    for et, nn in zip(types, nodes)
])
gmsh.finalize()

fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection="3d")
ax.add_collection3d(Poly3DCollection(
    xyz[tris], facecolor="#d9e6f2", edgecolor="#335577", linewidths=0.2))
pts = xyz[np.unique(tris)]
lo, hi = pts.min(axis=0), pts.max(axis=0)
ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
ax.set_box_aspect(hi - lo)
ax.view_init(elev=25, azim=-60)
ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]"); ax.set_zlabel("z [mm]")
fig.tight_layout()
fig.savefig(out, dpi=200)
print("wrote", out)