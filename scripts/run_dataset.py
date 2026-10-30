import argparse
from bracket.dataset import run_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--n_candidates", type=int, default=200)
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--out", default="results/dataset.csv")
a = ap.parse_args()
df = run_dataset(a.out, a.n_candidates, limit=a.limit)
print(df["status"].str.split(":").str[0].value_counts())