"""Standardized HTML convergence report: python -m bracket.report results/convergence.csv"""
import base64
import io
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from jinja2 import Template

TOL = 0.02   # converged if the last two relative changes are below 2 %

TEMPLATE = Template("""<!doctype html>
<html><head><meta charset="utf-8"><title>Mesh convergence report</title>
<style>
body { font-family: Arial, sans-serif; max-width: 900px; margin: 2em auto; color: #222; }
h1, h2 { border-bottom: 1px solid #ccc; padding-bottom: 4px; }
.tbl { border-collapse: collapse; font-size: 0.85em; }
.tbl td, .tbl th { border: 1px solid #ccc; padding: 3px 8px; text-align: right; }
.ok { background: #e6f4ea; padding: 10px; border-left: 5px solid #2e7d32; }
.bad { background: #fdecea; padding: 10px; border-left: 5px solid #c62828; }
img { max-width: 100%; }
</style></head><body>
<h1>Mesh convergence report</h1>
<h2>Geometry parameters</h2>
<p>Thickness t = {{ p.t }} mm, leg height H = {{ p.H }} mm, hole radius = {{ p.r_hole }} mm,
fillet radius = {{ p.r_fillet }} mm. Quantity monitored: maximum von Mises stress in the
region of interest (hole and fillet).</p>
<h2>Verdict</h2>
<div class="{{ 'ok' if v.converged else 'bad' }}">
<b>{{ 'CONVERGED' if v.converged else 'NOT CONVERGED' }}</b>:
last relative change {{ '%.2f' % (100 * v.last_change) }} % (criterion: below {{ '%.1f' % (100 * tol) }} %
for the last {{ v.n_check }} refinement step(s)). Finest mesh: {{ v.ndof }} DOFs,
ROI maximum {{ '%.2f' % v.final }} MPa.
</div>
<h2>Plots</h2>
{% for title, img in figs %}<h3>{{ title }}</h3><img src="data:image/png;base64,{{ img }}">{% endfor %}
<h2>Data</h2>
{{ table }}
</body></html>""")


def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def verdict(df, col="max_vm_roi", tol=TOL):
    v = df[col].to_numpy(dtype=float)
    rel = np.full(len(v), np.nan)
    rel[1:] = np.abs(v[1:] - v[:-1]) / np.abs(v[1:])
    n_check = 2 if len(v) >= 3 else 1
    ok = len(v) >= 2 and bool(np.all(rel[-n_check:] < tol))
    return rel, dict(converged=ok, last_change=float(rel[-1]), n_check=n_check,
                     ndof=f"{int(df['ndof'].iloc[-1]):,}", final=float(v[-1]))


def build_report(csv_path, out_html=None):
    df = pd.read_csv(csv_path).sort_values("ndof").reset_index(drop=True)
    rel, v = verdict(df)
    df["rel_change"] = rel
    figs = []

    fig, ax = plt.subplots(figsize=(7, 4))
    for col, lab in [("max_vm_roi", "ROI max"), ("max_vm_hole", "near hole"),
                     ("max_vm_root", "near fillet"), ("max_vm_global", "global max")]:
        ax.plot(df["ndof"], df[col], "o-", label=lab)
    ax.set_xscale("log"); ax.set_xlabel("DOFs"); ax.set_ylabel("max von Mises [MPa]")
    ax.grid(alpha=0.3); ax.legend()
    figs.append(("Maximum stress vs. mesh size", _png(fig)))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(df["ndof"], df["rel_change"], "o-", label="relative change (ROI max)")
    ax.axhline(TOL, color="r", ls="--", label=f"{TOL:.0%} criterion")
    ax.set_xscale("log"); ax.set_xlabel("DOFs"); ax.set_ylabel("relative change")
    ax.grid(alpha=0.3, which="both"); ax.legend()
    figs.append(("Convergence criterion", _png(fig)))

    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
    axs[0].plot(df["ndof"], df["aspect_max"], "o-", label="max")
    axs[0].plot(df["ndof"], df["aspect_mean"], "s-", label="mean")
    axs[0].set_title("Aspect ratio")
    axs[1].plot(df["ndof"], df["skew_max"], "o-", label="max")
    axs[1].plot(df["ndof"], df["skew_mean"], "s-", label="mean")
    axs[1].set_title("Skewness")
    for ax in axs:
        ax.set_xscale("log"); ax.set_xlabel("DOFs"); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    figs.append(("Mesh quality", _png(fig)))

    cols = ["h_global", "h_local", "n_tets", "ndof", "max_vm_roi", "max_vm_global",
            "rel_change", "aspect_max", "skew_max", "solve_time"]
    table = df[cols].round(4).to_html(index=False, classes="tbl", na_rep="-")
    html = TEMPLATE.render(p=df.iloc[0], v=v, tol=TOL, figs=figs, table=table)
    out_html = out_html or str(csv_path).replace(".csv", "_report.html")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    return out_html


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python -m bracket.report results/convergence.csv [out.html]")
    print("wrote", build_report(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None))