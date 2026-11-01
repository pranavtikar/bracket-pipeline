"""Von Mises contour plot of one design: python scripts/plot_stress.py [--renderer matplotlib]"""
import argparse

import numpy as np
import gmsh
from scipy.spatial import cKDTree

from bracket.fea import solve
from bracket.geometry import build_mesh
from bracket.viz import boundary_triangles, nodal_average, render

ap = argparse.ArgumentParser()
ap.add_argument("--t", type=float, default=6)
ap.add_argument("--H", type=float, default=45)
ap.add_argument("--r_hole", type=float, default=5)
ap.add_argument("--r_fillet", type=float, default=6)
ap.add_argument("--h_global", type=float, default=3.4)
ap.add_argument("--h_local", type=float, default=0.85)
ap.add_argument("--renderer", choices=["auto", "pyvista", "matplotlib"], default="auto")
ap.add_argument("--out", default="results/stress_contour.png")
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

mesh, vm = r.mesh, r.vm
nodes = mesh.geometry.x                          # all geometry nodes (corners and edge midpoints)
tets = mesh.geometry.dofmap[:, :4]               # the 4 corner nodes of every tetrahedron

# vm lives in DG1: one value per cell vertex. Map every value to its geometry node and average.
dof_xyz = vm.function_space.tabulate_dof_coordinates()
_, node_of_dof = cKDTree(nodes).query(dof_xyz)
values = nodal_average(len(nodes), node_of_dof, vm.x.array)
tris = boundary_triangles(tets)

vmax = float(vm.x.array.max())                   # unaveraged maximum, same number as elsewhere
clim = (float(values[np.unique(tris)].min()), vmax)
title = (f"von Mises stress, t = {p['t']:g}, H = {p['H']:g}, r_hole = {p['r_hole']:g}, "
         f"r_fillet = {p['r_fillet']:g} (mm)")
used = render(nodes, tris, values, clim, r.loc_max, f"Max {vmax:.1f} MPa", title, a.out, a.renderer)

print(f"renderer             {used}")
print(f"DOFs                 {r.ndof}")
print(f"max von Mises        {vmax:.1f} MPa (unaveraged), {values[np.unique(tris)].max():.1f} MPa (nodal average)")
print("peak location (mm)   x=%.1f y=%.1f z=%.1f" % r.loc_max)
print("wrote", a.out)