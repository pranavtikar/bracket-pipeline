"""Neural surrogate for the peak stress. Run: python -m bracket.surrogate"""
import json
import time
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)

FEATURES = ["t", "H", "r_hole", "r_fillet"]
TARGET = "max_vm_roi"


def wrap(reg):
    """log(inputs) -> standardise -> regressor; the stress itself is modelled on a log scale."""
    return TransformedTargetRegressor(
        regressor=make_pipeline(FunctionTransformer(np.log), StandardScaler(), reg),
        func=np.log, inverse_func=np.exp)


def evaluate(model, X, y):
    p = model.predict(X)
    ape = np.abs(p - y) / y
    return dict(mape_pct=100 * float(ape.mean()), r2=float(r2_score(y, p)),
                max_err_pct=100 * float(ape.max()), within5_pct=100 * float((ape < 0.05).mean()))


def main(csv="results/dataset_clean.csv", out_dir="results"):
    df = pd.read_csv(csv)
    df = df[df["status"] == "ok"].dropna(subset=[TARGET]).reset_index(drop=True)
    X, y = df[FEATURES].to_numpy(float), df[TARGET].to_numpy(float)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
    print(f"{len(df)} designs: {len(ytr)} train, {len(yte)} test")

    # hyperparameter search on the TRAINING set only (5-fold CV)
    cv = KFold(5, shuffle=True, random_state=0)
    grid = {"regressor__mlpregressor__hidden_layer_sizes": [(32, 32), (64, 64), (64, 64, 64)],
            "regressor__mlpregressor__alpha": [1e-6, 1e-4, 1e-2]}
    mlp = wrap(MLPRegressor(activation="tanh", solver="lbfgs", max_iter=5000, random_state=0))
    gs = GridSearchCV(mlp, grid, cv=cv, scoring="neg_mean_absolute_percentage_error")
    gs.fit(Xtr, ytr)
    best = gs.best_estimator_
    print("best MLP:", gs.best_params_, f"(CV MAPE {-100 * gs.best_score_:.2f} %)")

    gp = wrap(GaussianProcessRegressor(
        kernel=C(1.0, (1e-2, 1e2)) * RBF(np.ones(4), (1e-2, 1e2)) + WhiteKernel(1e-4, (1e-8, 1e-1)),
        normalize_y=True, n_restarts_optimizer=5, random_state=0))
    lin = wrap(LinearRegression())
    models = {"MLP (tuned)": best, "Gaussian process": gp, "Power law (log-log linear)": lin}

    # the test set is used once, for the final numbers
    rows = {}
    for name, m in models.items():
        if m is not best:
            m.fit(Xtr, ytr)
        rows[name] = evaluate(m, Xte, yte)
    table = pd.DataFrame(rows).T.round(3)
    print("\nHeld-out test set:")
    print(table.to_string())

    # inference time vs. FE solve time
    x1 = Xte[:1]
    best.predict(x1)
    n = 2000
    t0 = time.perf_counter()
    for _ in range(n):
        best.predict(x1)
    t_single = (time.perf_counter() - t0) / n
    big = np.repeat(Xte, 500, axis=0)
    t0 = time.perf_counter()
    best.predict(big)
    t_batch = (time.perf_counter() - t0) / len(big)
    fe_time = float(df["solve_time"].median())
    print(f"\nMLP prediction: {1e3 * t_single:.3f} ms per design, "
          f"{1e6 * t_batch:.1f} us per design in a batch")
    print(f"FE solve (median, mesh import + solve, no meshing): {fe_time:.2f} s "
          f"-> speed-up about {fe_time / t_single:,.0f}x")

    # learning curve: test error vs. number of training designs
    rng = np.random.RandomState(0)
    sizes = [20, 40, 80, 120, len(Xtr)]
    curve = []
    for k in sizes:
        errs = []
        for _ in range(5):
            idx = rng.choice(len(Xtr), k, replace=False)
            m = clone(best).fit(Xtr[idx], ytr[idx])
            errs.append(evaluate(m, Xte, yte)["mape_pct"])
        curve.append(float(np.mean(errs)))
    print("\nlearning curve (train size -> test MAPE %):",
          {k: round(v, 2) for k, v in zip(sizes, curve)})

    # plots
    pred = best.predict(Xte)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    lim = [0.9 * min(yte.min(), pred.min()), 1.1 * max(yte.max(), pred.max())]
    ax.plot(lim, lim, "k-", lw=1)
    ax.plot(lim, [0.95 * v for v in lim], "k--", lw=0.8)
    ax.plot(lim, [1.05 * v for v in lim], "k--", lw=0.8, label="±5 %")
    ax.scatter(yte, pred, s=22)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("FE max von Mises [MPa]"); ax.set_ylabel("surrogate prediction [MPa]")
    ax.set_title(f"Test set: MAPE {rows['MLP (tuned)']['mape_pct']:.2f} %, "
                 f"R² {rows['MLP (tuned)']['r2']:.3f}")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{out_dir}/surrogate_parity.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sizes, curve, "o-")
    ax.axhline(5, color="r", ls="--", label="5 % target")
    ax.set_xlabel("training designs"); ax.set_ylabel("test MAPE [%]")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(f"{out_dir}/surrogate_learning_curve.png", dpi=150); plt.close(fig)

    joblib.dump(best, f"{out_dir}/surrogate_mlp.joblib")
    with open(f"{out_dir}/surrogate_metrics.json", "w") as f:
        json.dump(dict(n_train=len(ytr), n_test=len(yte), best_params=gs.best_params_,
                       cv_mape_pct=-100 * gs.best_score_, test=rows,
                       predict_ms=1e3 * t_single, fe_solve_s=fe_time,
                       learning_curve=dict(zip(map(str, sizes), curve))),
                  f, indent=2, default=str)
    print(f"\nsaved plots, model and metrics to {out_dir}/")


if __name__ == "__main__":
    main()