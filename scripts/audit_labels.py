"""Re-solve every dataset design on a different mesh to detect faulty labels (resumable)."""
import argparse
import gc
import os

import gmsh
import pandas as pd

from bracket.fea import solve
from bracket.geometry import build_mesh
from bracket.quality import quality_metrics

FEATURES = ["t", "H", "r_hole", "r_fillet"]

ap = argparse.ArgumentParser()
ap.add_argument("--h", type=float, default=3.0)
ap.add_argument("--ratio", type=float, default=4.0)
ap.add_argument("--only", default=None, help="CSV with an 'id' column: audit only these ids")
ap.add_argument("--data", default="results/dataset.csv")
ap.add_argument("--out", default=None)
a = ap.parse_args()

out = a.out or f"results/audit_h{a.h}.csv"
df = pd.read_csv(a.data)
df = df[df["status"] == "ok"]
if a.only:
    df = df[df["id"].isin(pd.read_csv(a.only)["id"])]
done = set(pd.read_csv(out)["id"]) if os.path.exists(out) else set()
print(f"{len(df)} designs to audit, {len(done)} already done -> {out}")

for n, (_, r) in enumerate(df.iterrows(), 1):
    if int(r["id"]) in done:
        continue
    p = {k: float(r[k]) for k in FEATURES}
    row = dict(id=int(r["id"]), **p, h_global=a.h)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add("bracket")
        build_mesh(p, a.h, a.h / a.ratio)
        q = quality_metrics()
        res = solve(p)
        row.update(status="ok", max_vm_roi=res.max_vm_roi, max_vm_root=res.max_vm_root,
                   max_vm_hole=res.max_vm_hole, skew_max=q["skew_max"], ndof=res.ndof)
        print(f"[{n}/{len(df)}] id={row['id']}: {res.max_vm_roi:7.2f} MPa (dataset {r['max_vm_roi']:7.2f})")
    except Exception as e:
        row["status"] = f"failed: {type(e).__name__}"
        print(f"[{n}/{len(df)}] id={row['id']}: {row['status']}")
    finally:
        gmsh.finalize()
        gc.collect()
    pd.DataFrame([row]).to_csv(out, mode="a", header=not os.path.exists(out), index=False)