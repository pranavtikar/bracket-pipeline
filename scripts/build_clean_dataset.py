"""Build the cleaned training set from the mesh audit: results/dataset_clean.csv"""
import numpy as np
import pandas as pd

FEATURES = ["t", "H", "r_hole", "r_fillet"]
TOL = 0.03

d = pd.read_csv("results/dataset.csv")
d = d[d["status"] == "ok"].set_index("id")
a = pd.read_csv("results/audit_h3.0.csv")
a = a[a["status"] == "ok"].set_index("id")
assert set(d.index) == set(a.index), "audit does not cover every design"

d["roi_h3.4"] = d["max_vm_roi"]
d["roi_h3.0"] = a["max_vm_roi"]
rel = (d["roi_h3.4"] - d["roi_h3.0"]).abs() / d["roi_h3.0"]
flagged = rel.index[rel > TOL]
print(f"{len(flagged)} design(s) with mesh disagreement > {100 * TOL:.0f} %: {list(flagged)}")

d["label_rule"] = "mean of 2 meshes"
d["max_vm_roi"] = 0.5 * (d["roi_h3.4"] + d["roi_h3.0"])

fix = pd.read_csv("results/label_fix_id167.csv").set_index("h_global")
for i in flagged:
    assert i == 167, "unexpected flagged design: re-solve it on a third mesh first"
    for k in FEATURES:  # the stored re-check must belong to this design
        assert abs(fix[k].iloc[0] - d.loc[i, k]) < 1e-2, f"re-check does not match design {i}"
    third = float(fix.loc[2.6, "max_vm_roi"])
    d.loc[i, "max_vm_roi"] = float(np.median([d.loc[i, "roi_h3.4"], d.loc[i, "roi_h3.0"], third]))
    d.loc[i, "label_rule"] = "median of 3 meshes"
    print(f"design {i}: {d.loc[i, 'roi_h3.4']:.2f} / {d.loc[i, 'roi_h3.0']:.2f} / {third:.2f}"
          f" -> label {d.loc[i, 'max_vm_roi']:.2f} MPa")

print(f"label noise |h3.4 - h3.0| / h3.0: median {100 * rel.median():.2f} %, "
      f"95th percentile {100 * rel.quantile(0.95):.2f} %")
if 167 in d.index:
    n = int((d["skew_max"] >= d.loc[167, "skew_max"]).sum())
    print(f"designs with skew_max >= {d.loc[167, 'skew_max']:.3f} (the faulty design's value): {n}")
d.reset_index().to_csv("results/dataset_clean.csv", index=False)
print("wrote results/dataset_clean.csv with", len(d), "designs")