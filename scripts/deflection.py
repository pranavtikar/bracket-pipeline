"""How much does the leg bend? python scripts/deflection.py"""
import argparse

import numpy as np
import gmsh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bracket.fea import E, LOAD_N, solve
from bracket.geometry import W, build_mesh

ap = argparse.ArgumentParser()
ap.add_argument("--t", type=float, default=6)
ap.add_argument("--H", type=float, default=45)
ap.add_argument("--r_hole", type=float, default=5)
ap.add_argument("--r_fillet", type=float, default=6)
ap.add_argument("--h_global", type=float, default=3.4)
ap.add_argument("--h_local", type=float, default=0.85)
ap.add_argument("--out", default="results/deflection.png")
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

t, H = p["t"], p["H"]
xyz = r.uh.function_space.tabulate_dof_coordinates()      # one row per displacement node
ux = r.uh.x.array.reshape(-1, 3)[:, 0]                    # x displacement of every node

I = W * t ** 3 / 12                                       # second moment of the leg cross-section
tip_beam = LOAD_N * H ** 3 / (3 * E * I)
tilt_beam = LOAD_N * H ** 2 / (2 * E * I)
tip_fe = ux[xyz[:, 2] > t + H - 1e-6].mean()              # mean x displacement of the top face

# profile along the outer face (x = 0), averaged over the width, in 18 height bins
outer = (xyz[:, 0] < 1e-6) & (xyz[:, 2] >= t - 1e-6)
z, u = xyz[outer, 2], ux[outer]
edges = np.linspace(t, t + H, 19)
which = np.clip(np.digitize(z, edges) - 1, 0, 17)
zc = np.array([z[which == k].mean() for k in range(18) if (which == k).any()])
uc = np.array([u[which == k].mean() for k in range(18) if (which == k).any()])
top = zc > t + H - 10
tilt_fe = np.polyfit(zc[top], uc[top], 1)[0]              # slope of the profile near the top

print(f"tip deflection, FE (mean ux of the top face)  {tip_fe:+.4f} mm")
print(f"tip deflection, beam theory F H^3 / (3 E I)   {tip_beam:+.4f} mm   (FE / beam = {tip_fe / tip_beam:.2f})")
print(f"maximum |ux| anywhere                        {r.max_ux:.4f} mm")
print(f"tilt of the leg top, FE                      {tilt_fe:.5f} rad = {np.degrees(tilt_fe):.3f} deg")
print(f"tilt of the leg top, beam theory             {tilt_beam:.5f} rad = {np.degrees(tilt_beam):.3f} deg")
print(f"tip deflection relative to the leg height    {100 * tip_fe / H:.2f} %")

s = np.linspace(0, H, 100)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(LOAD_N * s ** 2 * (3 * H - s) / (6 * E * I), t + s, "-", color="#2e7d32",
        label="beam theory (cantilever)")
ax.plot(uc, zc, "o", color="#c0392b", label="FE (outer face, averaged over width)")
ax.set_xlabel("x displacement [mm]")
ax.set_ylabel("height z [mm]")
ax.set_title(f"Bending of the leg (F = {LOAD_N:g} N in +x)")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(a.out, dpi=150)
print("wrote", a.out)