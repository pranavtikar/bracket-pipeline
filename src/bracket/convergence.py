"""Automated mesh-convergence study for one parameter set."""
import time

import pandas as pd
import gmsh

from bracket.geometry import build_mesh
from bracket.quality import quality_metrics
from bracket.fea import solve

DEFAULT_PARAMS = dict(t=6.0, H=45.0, r_hole=5.0, r_fillet=6.0)
DEFAULT_H = [8.0, 6.0, 4.5, 3.4, 2.6, 2.0]   # global element size (mm)
DOFS_PER_TET = 5.6                            # empirical, for the pre-solve size estimate


def run_study(params=None, h_list=None, ratio=4.0, max_dofs=400_000,
              out_csv="results/convergence.csv"):
    """Mesh + solve for decreasing h. Local size = h / ratio. Saves the CSV after each step."""
    params = dict(params or DEFAULT_PARAMS)
    h_list = list(h_list or DEFAULT_H)
    rows = []
    for h in h_list:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        try:
            gmsh.model.add("bracket")
            t0 = time.perf_counter()
            build_mesh(params, h_global=h, h_local=h / ratio)
            mesh_time = time.perf_counter() - t0
            q = quality_metrics()
            est = DOFS_PER_TET * q["n_tets"]
            if est > max_dofs:
                print(f"h={h}: about {est:,.0f} DOFs expected (> {max_dofs:,}), stopping.")
                break
            r = solve(params)
        finally:
            gmsh.finalize()
        rows.append(dict(
            **params, h_global=h, h_local=h / ratio, n_tets=q["n_tets"], ndof=r.ndof,
            max_vm_roi=r.max_vm_roi, max_vm_global=r.max_vm_global,
            max_vm_root=r.max_vm_root, max_vm_hole=r.max_vm_hole, max_ux=r.max_ux,
            aspect_max=q["aspect_max"], aspect_mean=q["aspect_mean"],
            skew_max=q["skew_max"], skew_mean=q["skew_mean"],
            mesh_time=mesh_time, solve_time=r.solve_time,
        ))
        print(f"h={h:<5} DOFs={r.ndof:>8,}  ROI max={r.max_vm_roi:7.2f} MPa  "
              f"global={r.max_vm_global:7.2f}  solve={r.solve_time:5.1f} s")
        del r
        pd.DataFrame(rows).to_csv(out_csv, index=False)
    return pd.DataFrame(rows)