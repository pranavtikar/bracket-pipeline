"""Dimensioned drawing of the L-bracket: python scripts/plot_geometry.py"""
import argparse
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

VAR, FIX, GREY, FS = "#1f5fbf", "#222222", "#888888", 12
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


xl1, xl2 = (-24, L + 24), (-8, W + 24)
yl = (-24, t + H + 18)
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
ext(ax, 0, 0, 0, -13); ext(ax, L, 0, L, -13)
hdim(ax, 0, L, -11, f"{L:g}", FIX, below=True)
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
ax.set_title("Side view", fontsize=13)
ax.set_xlim(*xl1); ax.set_ylim(*yl); ax.set_aspect("equal"); ax.axis("off")

# ---- front view (y-z), looking at the inner face of the leg ----
ax = axs[1]
ax.fill([0, W, W, 0], [0, 0, t + H, t + H], facecolor="#e8eef6", edgecolor=FIX, lw=1.6)
ax.plot([0, W], [t, t], color=FIX, lw=0.8)                       # top of the base plate
ax.plot([0, W], [t + rf, t + rf], "--", color="#777777", lw=0.8)  # tangent edge of the fillet
ax.add_patch(plt.Circle((W / 2, zc), rh, facecolor="white", edgecolor=FIX, lw=1.6, zorder=3))
ext(ax, 0, 0, 0, -13); ext(ax, W, 0, W, -13)
hdim(ax, 0, W, -11, f"{W:g}", FIX, below=True)
a40 = np.radians(40)
ax.annotate(f"r_hole = {rh:g}", xy=(W / 2 + rh * np.cos(a40), zc + rh * np.sin(a40)),
            xytext=(W / 2 + 4, zc + rh + 9), color=VAR, fontsize=FS, zorder=4,
            arrowprops=dict(arrowstyle="->", color=VAR, lw=1.1))
ext(ax, W, t, W + 13, t); ext(ax, W / 2 + rh, zc, W + 13, zc)
vdim(ax, W + 11, t, zc, f"{HOLE_FRAC:g}·H = {HOLE_FRAC * H:g}", VAR, side="right")
ax.text(W / 2, t + rf + 1.5, f"fillet R = {rf:g} (tangent edge)", ha="center", va="bottom",
        fontsize=10, color="#555555")
ax.set_title("Front view (looking at the inner face of the leg)", fontsize=13)
ax.set_xlim(*xl2); ax.set_ylim(*yl); ax.set_aspect("equal"); ax.axis("off")

fig.suptitle(f"L-bracket, reference design: t = {t:g}, H = {H:g}, r_hole = {rh:g}, "
             f"r_fillet = {rf:g}  (all dimensions in mm)", fontsize=14)
fig.text(0.5, 0.02, "Blue: the four variable parameters.  Black: fixed base size.  "
         "The bottom face is clamped, the load acts on the top face of the leg.",
         ha="center", fontsize=11, color="#333333")
fig.tight_layout(rect=(0, 0.05, 1, 0.94))
fig.savefig(a.out, dpi=170)
print("wrote", a.out)