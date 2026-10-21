"""Axis-wise Ridge fitting with explicit label masks and frozen data splits."""

import numpy as np
from traser.io import write


def fit(input_path, output):
    data = np.load(input_path, allow_pickle=False)
    x, y, split = (data["coordinates"], data["labels"], data["split"])
    assert (
        x.ndim == 2
        and x.shape[1] == 64
        and (y.shape == (len(x), 11))
        and (split.shape == (len(x),))
    )
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    fit = split == "fit"
    val = split == "val"
    mean = x[fit].mean(0)
    sd = np.maximum(x[fit].std(0), 1e-08)
    z = (x - mean) / sd
    coef = np.zeros((64, 11))
    intercept = np.zeros(11)
    records = []
    for j in range(11):
        a = fit & np.isfinite(y[:, j])
        v = val & np.isfinite(y[:, j])
        assert a.sum() > 1 and v.sum() > 0
        results = []
        for alpha in [0.01, 0.1, 1, 10, 100]:
            xm = z[a].mean(0)
            ym = y[a, j].mean()
            center = z[a] - xm
            b = np.linalg.solve(center.T @ center + alpha * np.eye(64), center.T @ (y[a, j] - ym))
            bias = ym - xm @ b
            error = float(np.mean((np.clip(z[v] @ b + bias, 0, 1) - y[v, j]) ** 2))
            results.append((error, alpha, b, bias))
        error, alpha, coef[:, j], intercept[j] = min(results, key=lambda t: (t[0], -t[1]))
        records.append(
            {
                "axis": j,
                "alpha": alpha,
                "validation_mse": error,
                "fit_n": int(a.sum()),
                "val_n": int(v.sum()),
            }
        )
    np.savez(output / "model.npz", mean=mean, sd=sd, coef=coef, intercept=intercept)
    write(
        output / "selection.json",
        {
            "records": records,
            "human_validated": False,
            "status": "new explicit input fit, not the frozen paper mapping",
            "missing_labels": "per-axis masks, no imputation",
        },
    )
