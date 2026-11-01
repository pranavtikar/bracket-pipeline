import argparse
import gmsh
from bracket.geometry import build_mesh
from bracket.quality import quality_metrics

ap = argparse.ArgumentParser()
ap.add_argument("--t", type=float, default=6)
ap.add_argument("--H", type=float, default=45)
ap.add_argument("--r_hole", type=float, default=5)
ap.add_argument("--r_fillet", type=float, default=6)
ap.add_argument("--h_global", type=float, default=6)
ap.add_argument("--h_local", type=float, default=1.5)
ap.add_argument("--out", default="results/bracket.msh")
a = ap.parse_args()

params = dict(t=a.t, H=a.H, r_hole=a.r_hole, r_fillet=a.r_fillet)
gmsh.initialize()
try:
    gmsh.model.add("bracket")
    build_mesh(params, a.h_global, a.h_local)
    for k, v in quality_metrics().items():
        print(f"{k:12s} {v:.4g}")
    gmsh.option.setNumber("Mesh.SaveAll", 1)
    gmsh.write(a.out)
    print("wrote", a.out)
finally:
    gmsh.finalize()