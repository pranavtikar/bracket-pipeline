"""Dimensioned drawing of the L-bracket: python scripts/plot_geometry.py"""
import argparse
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bracket.fea import LOAD_N
from bracket.geometry import HOLE_FRAC, L, W, is_valid

ap = argparse.ArgumentParser()
ap.add_argument("--t", type=float, default=6)
ap.add_argument("--H", type=float, default=45)
ap.add_argument("--r_hole", type=float, default=5)
ap.add_argument("--r_fillet", type=float, default=6)
ap.add_argument("--out", default="results/geometry.png")
a = ap.parse_args()

p = dict(t=a.t, H=a.H, r_hole=a.r_hole, r_fillet=a.r_fillet)
if not is_valid(p):
    sys.exit(f"invalid geometry: {p}")
t, H, rh, rf = a.t, a.H, a.r_hole, a.r_fillet
zc = t + HOLE_FRAC * H                      # height of the hole centre above the bottom face

VAR, FIX, GREY, RED, FS = "#1f5fbf", "#222222", "#888888", "#c0392b", 12
ARROW = dict(arrowstyle="<->", lw=1.1, shrinkA=0, shrinkB=0)


def hdim(ax, x0, x1, y, label, color, below=False):
    ax.annotate("", xy=(x0, y), xytext=(x1, y), arrowprops=dict(color=color, **ARROW))
    ax.text(0.5 * (x0 + x1), y - 2 if below else y + 2, label, ha="center",
            va="top" if below else "bottom", color=color, fontsize=FS)


def vdim(ax, x, y0, y1, label, color, side="left", rotate=True):
    ax.annotate("", xy=(x, y0), xytext=(x, y1), arrowprops=dict(color=color, **ARROW))
    ax.text(x - 2 if side == "left" else x + 2, 0.5 * (y0 + y1), label,
            ha="right" if side == "left" else "left", va="center",
            rotation=90 if rotate else 0, color=color, fontsize=FS)


def ext(ax, x0, y0, x1, y1):
    ax.plot([x0, x1], [y0, y1], color=GREY, lw=0.8)


def hatch(ax, x0, x1):
    """Clamped support: thick line with hatching below it."""
    ax.plot([x0, x1], [0, 0], color=FIX, lw=2.6)
    for x in np.arange(x0 + 2, x1 + 1, 4):
        ax.plot([x, x - 3], [0, -4], color=FIX, lw=0.8)


def arrow(ax, x0, y0, x1, y1, color, lw=1.2, scale=12):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=scale,
                                shrinkA=0, shrinkB=0))


xl1, xl2 = (-24, L + 24), (-8, W + 28)
yl = (-28, t + H + 24)
fig, axs = plt.subplots(1, 2, figsize=(13, 6.4),
                        gridspec_kw=dict(width_ratios=[xl1[1] - xl1[0], xl2[1] - xl2[0]]))

# ---- side view (x-z) ----
ax = axs[0]
phi = np.radians(np.linspace(270, 180, 40))
arc = np.c_[t + rf + rf * np.cos(phi), t + rf + rf * np.sin(phi)]
outline = np.vstack([[0, 0], [L, 0], [L, t], arc, [t, t + H], [0, t + H], [0, 0]])
ax.fill(outline[:, 0], outline[:, 1], facecolor="#e8eef6", edgecolor=FIX, lw=1.6)
for z in (zc - rh, zc + rh):                # hidden edges of the hole
    ax.plot([0, t], [z, z], "--", color="#555555", lw=1)
hatch(ax, 0, L)
ax.text(L / 2, -9.5, "bottom face fixed (clamped)", ha="center", va="center",
        fontsize=10, color="#333333")
ext(ax, 0, 0, 0, -19); ext(ax, L, 0, L, -19)
hdim(ax, 0, L, -17, f"{L:g}", FIX, below=True)
ext(ax, 0, t, -13, t); ext(ax, 0, t + H, -13, t + H)
vdim(ax, -11, t, t + H, f"H = {H:g}", VAR, side="left")
ext(ax, 0, t + H, 0, t + H + 12); ext(ax, t, t + H, t, t + H + 12)
hdim(ax, 0, t, t + H + 10, f"t = {t:g}", VAR)
ext(ax, L, 0, L + 13, 0); ext(ax, L, t, L + 13, t)
vdim(ax, L + 11, 0, t, f"t = {t:g}", VAR, side="right", rotate=False)
ang = np.radians(225)
ax.annotate(f"R = {rf:g}", xy=(t + rf + rf * np.cos(ang), t + rf + rf * np.sin(ang)),
            xytext=(t + rf + 18, t + rf + 14), color=VAR, fontsize=FS,
            arrowprops=dict(arrowstyle="->", color=VAR, lw=1.1))
arrow(ax, t / 2, t + H, t + 34, t + H, RED, lw=2.8, scale=22)       # load: +x, on the top face
ax.text(t + 4, t + H + 2.5, f"F = {LOAD_N:g} N", color=RED, fontsize=13, fontweight="bold",
        ha="left", va="bottom")
ax.text(t + 4, t + H - 2.5, "(+x, shear on the top face)", color=RED, fontsize=10,
        ha="left", va="top")
ox, oz = L - 24, t + 22                                              # axis triad
arrow(ax, ox, oz, ox + 14, oz, "#333333"); arrow(ax, ox, oz, ox, oz + 14, "#333333")
ax.text(ox + 17, oz, "x", ha="left", va="center", fontsize=12, color="#333333")
ax.text(ox, oz + 17, "z", ha="center", va="bottom", fontsize=12, color="#333333")
ax.set_title("Side view", fontsize=13)
ax.set_xlim(*xl1); ax.set_ylim(*yl); ax.set_aspect("equal"); ax.axis("off")

# ---- front view (y-z), looking at the inner face of the leg ----
ax = axs[1]
ax.fill([0, W, W, 0], [0, 0, t + H, t + H], facecolor="#e8eef6", edgecolor=FIX, lw=1.6)
ax.plot([0, W], [t, t], color=FIX, lw=0.8)                       # top of the base plate
ax.plot([0, W], [t + rf, t + rf], "--", color="#777777", lw=0.8)  # tangent edge of the fillet
ax.add_patch(plt.Circle((W / 2, zc), rh, facecolor="white", edgecolor=FIX, lw=1.6, zorder=3))
hatch(ax, 0, W)
ax.text(W / 2, -9.5, "bottom face fixed", ha="center", va="center", fontsize=10,
        color="#333333")
ext(ax, 0, 0, 0, -19); ext(ax, W, 0, W, -19)
hdim(ax, 0, W, -17, f"{W:g}", FIX, below=True)
a40 = np.radians(40)
ax.annotate(f"r_hole = {rh:g}", xy=(W / 2 + rh * np.cos(a40), zc + rh * np.sin(a40)),
            xytext=(W / 2 + 4, zc + rh + 9), color=VAR, fontsize=FS, zorder=4,
            arrowprops=dict(arrowstyle="->", color=VAR, lw=1.1))
ext(ax, W, t, W + 13, t); ext(ax, W / 2 + rh, zc, W + 13, zc)
vdim(ax, W + 11, t, zc, f"{HOLE_FRAC:g}·H = {HOLE_FRAC * H:g}", VAR, side="right")
ax.text(W / 2, t + rf + 1.5, f"fillet R = {rf:g} (tangent edge)", ha="center", va="bottom",
        fontsize=10, color="#555555")
for yy in (6, 13, 20, 27, 34):                                       # load points towards the viewer
    ax.add_patch(plt.Circle((yy, t + H + 5), 2.2, fill=False, edgecolor=RED, lw=1.8))
    ax.add_patch(plt.Circle((yy, t + H + 5), 0.8, facecolor=RED, edgecolor=RED))
ax.text(W / 2, t + H + 10, f"F = {LOAD_N:g} N\n+x, towards the viewer", color=RED, fontsize=12,
        fontweight="bold", ha="center", va="bottom")
ox, oz = W + 6, t + H - 6                                           # axis triad
arrow(ax, ox, oz, ox + 14, oz, "#333333"); arrow(ax, ox, oz, ox, oz + 14, "#333333")
ax.add_patch(plt.Circle((ox, oz), 2.0, fill=False, edgecolor="#333333", lw=1.2, zorder=5))
ax.add_patch(plt.Circle((ox, oz), 0.6, facecolor="#333333", zorder=5))
ax.text(ox + 17, oz, "y", ha="left", va="center", fontsize=12, color="#333333")
ax.text(ox, oz + 17, "z", ha="center", va="bottom", fontsize=12, color="#333333")
ax.text(ox + 3, oz - 3, "x (out of page)", ha="left", va="top", fontsize=9, color="#333333")
ax.set_title("Front view (looking at the inner face of the leg)", fontsize=13)
ax.set_xlim(*xl2); ax.set_ylim(*yl); ax.set_aspect("equal"); ax.axis("off")

fig.suptitle(f"L-bracket, reference design: t = {t:g}, H = {H:g}, r_hole = {rh:g}, "
             f"r_fillet = {rf:g}  (all dimensions in mm)", fontsize=14)
fig.text(0.5, 0.02, "Blue: the four variable parameters.  Black: fixed base size.  "
         "Red: load (uniform shear traction on the top face of the leg).  Hatching: clamped bottom face.",
         ha="center", fontsize=11, color="#333333")
fig.tight_layout(rect=(0, 0.05, 1, 0.94))
fig.savefig(a.out, dpi=170)
print("wrote", a.out)