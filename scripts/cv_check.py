import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neural_network import MLPRegressor

from bracket.surrogate import wrap, FEATURES, TARGET

df = pd.read_csv("results/dataset_clean.csv")
df = df[df["status"] == "ok"].reset_index(drop=True)
X, y = df[FEATURES].to_numpy(float), df[TARGET].to_numpy(float)

model = wrap(MLPRegressor(hidden_layer_sizes=(64, 64), alpha=1e-2, activation="tanh",
                          solver="lbfgs", max_iter=5000, random_state=0))
errs = []
for seed in range(3):
    pred = cross_val_predict(model, X, y, cv=KFold(5, shuffle=True, random_state=seed))
    errs.append(np.abs(pred - y) / y)
allerr = np.concatenate(errs)
print(f"{len(allerr)} out-of-fold predictions")
print(f"MAPE {100 * allerr.mean():.2f} %   95th percentile {100 * np.percentile(allerr, 95):.2f} %"
      f"   max {100 * allerr.max():.2f} %   within 5 %: {100 * (allerr < 0.05).mean():.1f} %")

df["err_pct"] = 100 * np.mean(errs, axis=0)
print("\nworst designs:")
print(df.nlargest(5, "err_pct")[FEATURES + [TARGET, "err_pct"]].round(2).to_string(index=False))