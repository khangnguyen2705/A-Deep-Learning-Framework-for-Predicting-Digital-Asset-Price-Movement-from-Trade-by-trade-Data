"""Out-of-sample evaluation utilities.

The paper evaluates the best model chronologically on all valid test examples
(Section V.A). No shuffling. Labels use epsilon=0 (no class balancing).
"""
import numpy as np
import pandas as pd


def evaluate_out_of_sample(model, X_test, y_test, t_indices, intervals,
                            epsilon=0.0):
    """Run the model on the test set and return accuracy, loss, and predictions.

    Returns:
        accuracy:       Scalar directional accuracy.
        loss:           Categorical cross-entropy on test set.
        y_pred_class:   (N,) int array of predicted class indices.
        y_true_class:   (N,) int array of true class indices.
        y_pred_proba:   (N, 2) float array of softmax probabilities.
    """
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred_class = np.argmax(y_pred_proba, axis=1)
    y_true_class = np.argmax(y_test, axis=1)

    accuracy = float(np.mean(y_pred_class == y_true_class))
    loss = float(model.evaluate(X_test, y_test, verbose=0)[0])

    return accuracy, loss, y_pred_class, y_true_class, y_pred_proba


def compute_rolling_accuracy(y_true_class, y_pred_class, intervals,
                              t_indices, window_days=1,
                              ms_per_day=86_400_000):
    """Compute per-day rolling directional accuracy.

    Returns a DataFrame with columns: day_start, day_end, accuracy, count.
    """
    results = []
    timestamps = intervals["interval_end_ms"].values[t_indices]
    start_day = timestamps[0]
    end_day = timestamps[-1]
    current = start_day
    while current < end_day:
        day_end = current + ms_per_day
        mask = (timestamps >= current) & (timestamps < day_end)
        if np.sum(mask) > 0:
            acc = float(np.mean(y_pred_class[mask] == y_true_class[mask]))
            results.append({
                "day_start": current,
                "day_end": day_end,
                "accuracy": acc,
                "count": int(np.sum(mask)),
            })
        current = day_end
    return pd.DataFrame(results)
