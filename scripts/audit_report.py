import numpy as np
import pandas as pd

d = pd.read_csv("results/dataset.csv").set_index("id")
a = pd.read_csv("results/audit_h3.0.csv")
a = a[a["status"] == "ok"].set_index("id")
j = d[["max_vm_roi", "skew_max"]].join(a[["max_vm_roi", "skew_max"]], rsuffix="_b", how="inner")
j["diff_pct"] = 100 * (j["max_vm_roi"] - j["max_vm_roi_b"]) / j["max_vm_roi_b"]
ad = j["diff_pct"].abs()

print(f"{len(j)} designs compared (dataset mesh 3.4 vs second mesh 3.0)")
print(f"median |difference| {ad.median():.2f} %,  95th percentile {np.percentile(ad, 95):.2f} %,  max {ad.max():.2f} %")
for thr in (2, 3, 5, 10):
    print(f"  more than {thr:>2} % apart: {int((ad > thr).sum())} designs")
print("\nlargest differences:")
print(j.loc[ad.sort_values(ascending=False).index[:8]].round(3).to_string())

flag = j[ad > 3].reset_index()[["id"]]
flag.to_csv("results/flagged_ids.csv", index=False)
print(f"\nwrote results/flagged_ids.csv ({len(flag)} designs)")