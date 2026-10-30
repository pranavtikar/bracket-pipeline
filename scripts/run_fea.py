import argparse
import gmsh
from bracket.geometry import build_mesh, W
from bracket.fea import solve, E, LOAD_N

ap = argparse.ArgumentParser()
ap.add_argument("--t", type=float, default=6)
ap.add_argument("--H", type=float, default=45)
ap.add_argument("--r_hole", type=float, default=5)
ap.add_argument("--r_fillet", type=float, default=6)
ap.add_argument("--h_global", type=float, default=6)
ap.add_argument("--h_local", type=float, default=1.5)
a = ap.parse_args()

p = dict(t=a.t, H=a.H, r_hole=a.r_hole, r_fillet=a.r_fillet)
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
try:
    gmsh.model.add("bracket")
    build_mesh(p, a.h_global, a.h_local)
    r = solve(p)
finally:
    gmsh.finalize()

# rough sanity check: end-loaded cantilever of length H and cross-section W x t
I = W * p["t"] ** 3 / 12
beam = LOAD_N * p["H"] ** 3 / (3 * E * I)

print(f"DOFs                 {r.ndof}")
print(f"max von Mises (ROI)  {r.max_vm_roi:.1f} MPa")
print(f"max von Mises (all)  {r.max_vm_global:.1f} MPa")
print(f"max ux               {r.max_ux:.4f} mm   (beam theory ~ {beam:.4f} mm)")
print(f"time                 {r.solve_time:.2f} s")