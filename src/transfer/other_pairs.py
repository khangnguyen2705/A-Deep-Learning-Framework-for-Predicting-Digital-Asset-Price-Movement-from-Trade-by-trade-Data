import numpy as np
import pandas as pd


def evaluate_transfer(model, intervals, Cm, labels, T, epsilon=0.0):
    from src.datasets.windowing import build_trailing_windows, FEATURE_COLUMNS

    X, y, t_idx = build_trailing_windows(
        intervals, Cm, labels, T=T, m=6,
        offset_fraction_min=1.0, offset_fraction_max=1.0,
        rng=np.random.default_rng(42),
    )
    if len(X) == 0:
        return 0.0, 0.0, np.array([]), np.array([])

    y_pred = model.predict(X, verbose=0)
    y_pred_class = np.argmax(y_pred, axis=1)
    y_true_class = np.argmax(y, axis=1)
    accuracy = np.mean(y_pred_class == y_true_class)
    loss = model.evaluate(X, y, verbose=0)[0]
    return accuracy, loss, y_pred_class, y_true_class
