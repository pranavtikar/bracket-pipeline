import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("results/convergence.csv").sort_values("ndof")
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(df["ndof"], df["max_vm_roi"], "o-", label="ROI maximum")
ax.plot(df["ndof"], df["max_vm_hole"], "s--", label="near hole")
ax.plot(df["ndof"], df["max_vm_root"], "^--", label="near fillet")
ax.set_xscale("log")
ax.set_xlabel("degrees of freedom")
ax.set_ylabel("max von Mises [MPa]")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig("results/convergence.png", dpi=150)
print("wrote results/convergence.png")