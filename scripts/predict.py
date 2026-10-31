import argparse
import time

import joblib
import numpy as np

from bracket.dataset import BOUNDS
from bracket.geometry import is_valid

ap = argparse.ArgumentParser()
for k in BOUNDS:
    ap.add_argument(f"--{k}", type=float, required=True)
ap.add_argument("--model", default="results/surrogate_mlp.joblib")
a = ap.parse_args()

p = {k: getattr(a, k) for k in BOUNDS}
for k, (lo, hi) in BOUNDS.items():
    if not lo <= p[k] <= hi:
        print(f"warning: {k}={p[k]} is outside the training range [{lo}, {hi}], extrapolating")
if not is_valid(p):
    print("warning: this geometry fails the validity check, so it could not be meshed")

model = joblib.load(a.model)
x = np.array([[p[k] for k in BOUNDS]])
model.predict(x)
t0 = time.perf_counter()
y = float(model.predict(x)[0])
dt = time.perf_counter() - t0
print(f"predicted max von Mises (ROI): {y:.1f} MPa   ({1e3 * dt:.2f} ms)")