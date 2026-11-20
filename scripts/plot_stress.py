"""Von Mises contour plot of one design: python scripts/plot_stress.py [--warp 100]"""
import argparse

import numpy as np
import gmsh
from scipy.spatial import cKDTree

from bracket.fea import solve
from bracket.geometry import build_mesh
from bracket.viz import boundary_triangles, feature_edges, nodal_average, render

ap = argparse.ArgumentParser()
ap.add_argument("--t", type=float, default=6)
ap.add_argument("--H", type=float, default=45)
ap.add_argument("--r_hole", type=float, default=5)
ap.add_argument("--r_fillet", type=float, default=6)
ap.add_argument("--h_global", type=float, default=3.4)
ap.add_argument("--h_local", type=float, default=0.85)
ap.add_argument("--warp", type=float, default=0.0,
                help="draw the deformed shape with all displacements multiplied by this factor")
ap.add_argument("--renderer", choices=["auto", "pyvista", "matplotlib"], default="auto")
ap.add_argument("--out", default=None)
a = ap.parse_args()
out = a.out or ("results/stress_contour_warp.png" if a.warp else "results/stress_contour.png")

p = dict(t=a.t, H=a.H, r_hole=a.r_hole, r_fillet=a.r_fillet)
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
try:
    gmsh.model.add("bracket")
    build_mesh(p, a.h_global, a.h_local)
    r = solve(p)
finally:
    gmsh.finalize()

mesh, vm = r.mesh, r.vm
nodes = mesh.geometry.x                          # all geometry nodes (corners and edge midpoints)
tets = mesh.geometry.dofmap[:, :4]               # the 4 corner nodes of every tetrahedron
tree = cKDTree(nodes)

# vm lives in DG1: one value per cell vertex. Map every value to its geometry node and average.
_, node_of_dof = tree.query(vm.function_space.tabulate_dof_coordinates())
values = nodal_average(len(nodes), node_of_dof, vm.x.array)
tris = boundary_triangles(tets)

points, outline, marker = nodes, None, np.array(r.loc_max)
if a.warp:
    # displacement of every geometry node (the P2 displacement nodes coincide with the geometry nodes)
    dist, node_of_u = tree.query(r.uh.function_space.tabulate_dof_coordinates())
    if dist.max() > 1e-6 or len(np.unique(node_of_u)) != len(nodes):
        raise RuntimeError("the displacement nodes do not coincide with the geometry nodes")
    disp = np.zeros_like(nodes)
    disp[node_of_u] = r.uh.x.array.reshape(-1, 3)
    points = nodes + a.warp * disp
    outline = (nodes, feature_edges(nodes, tris))
    marker = points[tree.query(r.loc_max)[1]]    # the peak moves with the material
    umax = np.linalg.norm(disp, axis=1).max()

vmax = float(vm.x.array.max())                   # unaveraged maximum, same number as elsewhere
clim = (float(values[np.unique(tris)].min()), vmax)
title = (f"von Mises stress, t = {p['t']:g}, H = {p['H']:g}, r_hole = {p['r_hole']:g}, "
         f"r_fillet = {p['r_fillet']:g} (mm)")
if a.warp:
    title += f"\ndeformation x{a.warp:g} (thin black lines: undeformed outline)"
used = render(points, tris, values, clim, marker, f"Max {vmax:.1f} MPa", title, out,
              a.renderer, outline)

print(f"renderer             {used}")
print(f"DOFs                 {r.ndof}")
print(f"max von Mises        {vmax:.1f} MPa (unaveraged), {values[np.unique(tris)].max():.1f} MPa (nodal average)")
print("peak location (mm)   x=%.1f y=%.1f z=%.1f" % r.loc_max)
if a.warp:
    print(f"max displacement     {umax:.4f} mm, drawn x{a.warp:g} = {a.warp * umax:.2f} mm")
print("wrote", out)
