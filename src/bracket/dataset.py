"""FE dataset: Latin-hypercube sampling + resumable batch runner."""
import gc
import os

import numpy as np
import pandas as pd
import gmsh
from scipy.stats import qmc

from bracket.geometry import build_mesh, is_valid
from bracket.quality import quality_metrics
from bracket.fea import solve

BOUNDS = {"t": (4.0, 10.0), "H": (30.0, 60.0),
          "r_hole": (3.0, 8.0), "r_fillet": (3.0, 10.0)}
H_GLOBAL, H_LOCAL = 3.4, 0.85
NAMES = list(BOUNDS)
COLUMNS = ["id", *NAMES, "status", "max_vm_roi", "max_vm_hole", "max_vm_root",
           "max_vm_global", "ndof", "n_tets", "aspect_max", "skew_max", "solve_time"]


def sample_designs(n_candidates=200, seed=42):
    lo = np.array([BOUNDS[k][0] for k in NAMES])
    hi = np.array([BOUNDS[k][1] for k in NAMES])
    sampler = qmc.LatinHypercube(d=len(NAMES), seed=seed)
    X = np.round(qmc.scale(sampler.random(n_candidates), lo, hi), 3)
    designs = [dict(zip(NAMES, map(float, row))) for row in X]
    return [d for d in designs if is_valid(d)]


def run_dataset(out_csv="results/dataset.csv", n_candidates=200, seed=42, limit=None):
    designs = sample_designs(n_candidates, seed)
    if limit:
        designs = designs[:limit]
    done = set(pd.read_csv(out_csv)["id"]) if os.path.exists(out_csv) else set()
    print(f"{len(designs)} valid designs, {len(done)} already done")
    for i, p in enumerate(designs):
        if i in done:
            continue
        row = dict(id=i, **p)
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        try:
            gmsh.model.add("bracket")
            build_mesh(p, H_GLOBAL, H_LOCAL)
            q = quality_metrics()
            r = solve(p)
            row.update(status="ok", max_vm_roi=r.max_vm_roi, max_vm_hole=r.max_vm_hole,
                       max_vm_root=r.max_vm_root, max_vm_global=r.max_vm_global,
                       ndof=r.ndof, n_tets=q["n_tets"], aspect_max=q["aspect_max"],
                       skew_max=q["skew_max"], solve_time=r.solve_time)
            print(f"[{i + 1}/{len(designs)}] ok  ROI max={r.max_vm_roi:7.2f} MPa  "
                  f"DOFs={r.ndof:,}  {r.solve_time:.1f} s")
        except Exception as e:
            row["status"] = f"failed: {type(e).__name__}: {str(e)[:80]}"
            print(f"[{i + 1}/{len(designs)}] {row['status']}")
        finally:
            gmsh.finalize()
            gc.collect()
        pd.DataFrame([row]).reindex(columns=COLUMNS).to_csv(
            out_csv, mode="a", header=not os.path.exists(out_csv), index=False)
    return pd.read_csv(out_csv)