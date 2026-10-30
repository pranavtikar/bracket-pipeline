import argparse
from bracket.convergence import run_study, DEFAULT_PARAMS, DEFAULT_H

ap = argparse.ArgumentParser()
for k, v in DEFAULT_PARAMS.items():
    ap.add_argument(f"--{k}", type=float, default=v)
ap.add_argument("--h", type=float, nargs="+", default=DEFAULT_H)
ap.add_argument("--ratio", type=float, default=4.0)
ap.add_argument("--max_dofs", type=int, default=400_000)
ap.add_argument("--out", default="results/convergence.csv")
a = ap.parse_args()

params = {k: getattr(a, k) for k in DEFAULT_PARAMS}
run_study(params, a.h, a.ratio, a.max_dofs, a.out)
print("wrote", a.out)