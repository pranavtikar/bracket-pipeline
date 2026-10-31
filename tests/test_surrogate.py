import json
import pathlib

import joblib
import numpy as np

METRICS = json.loads(pathlib.Path("results/surrogate_metrics.json").read_text())


def test_accuracy_target_on_held_out_designs():
    m = METRICS["test"]["MLP (tuned)"]
    assert m["mape_pct"] <= 5.0
    assert m["r2"] >= 0.95


def test_prediction_matches_fe_on_an_unseen_design():
    model = joblib.load("results/surrogate_mlp.joblib")
    pred = float(model.predict(np.array([[7.3, 52.0, 4.2, 5.5]]))[0])
    assert abs(pred - 82.1) / 82.1 < 0.05   # 82.1 MPa from run_fea.py at h = 3.4 / 0.85