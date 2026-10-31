import sys

import pandas as pd

from bracket.convergence import run_study

target = float(sys.argv[1])
d = pd.read_csv("results/dataset.csv")
r = d.loc[(d["max_vm_roi"] - target).abs().idxmin()]
p = {k: float(r[k]) for k in ["t", "H", "r_hole", "r_fillet"]}
print("design", int(r["id"]), p, "dataset value:", round(float(r["max_vm_roi"]), 2))
run_study(p, [3.4, 3.0, 2.6], ratio=4.0, out_csv="results/recheck.csv")
cols = ["h_global", "ndof", "max_vm_roi", "max_vm_root", "max_vm_hole", "max_vm_global", "skew_max"]
print(pd.read_csv("results/recheck.csv")[cols].round(2).to_string(index=False))