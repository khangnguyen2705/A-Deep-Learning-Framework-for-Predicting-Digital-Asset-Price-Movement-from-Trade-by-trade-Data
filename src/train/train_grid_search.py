import numpy as np
import tensorflow as tf
from src.model.lstm_classifier import build_lstm_classifier, compile_model


def _make_lr_schedule(initial_lr=0.001, decay=0.0003, decay_epochs=15,
                       min_lr=0.0001):
    """Paper Section IV.A LR schedule, applied per epoch via a callback.

      lr(epoch) = max(initial_lr - decay * floor(epoch / decay_epochs), min_lr)

    Implemented as a `LearningRateScheduler` callback because subclassing
    `LearningRateSchedule` and feeding a fictive (step/100) epoch caused a
    TF eager-execute deadlock in TF 2.21 paired with the LSTM kernel on
    Apple Silicon.
    """
    def schedule(epoch, lr):
        new_lr = initial_lr - decay * (epoch // decay_epochs)
        return float(max(new_lr, min_lr))
    return tf.keras.callbacks.LearningRateScheduler(schedule, verbose=0)


class EarlyStoppingByLoss(tf.keras.callbacks.Callback):
    def __init__(self, patience=20, delta=0.05):
        super().__init__()
        self.patience = patience
        self.delta = delta
        self.best_epoch = 0
        self.best_loss = float("inf")
        self.wait = 0
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("val_loss")
        if current is None:
            return
        if current < self.best_loss:
            self.best_loss = current
            self.best_epoch = epoch
            self.wait = 0
        else:
            self.wait += 1
        if current > self.best_loss * (1 + self.delta):
            self.model.stop_training = True
            self.stopped_epoch = epoch
        if self.wait >= self.patience:
            self.model.stop_training = True
            self.stopped_epoch = epoch


def compute_batch_size(n: int) -> int:
    if n > 50000:
        return 128
    elif n > 20000:
        return 64
    else:
        return 32


def train_model(X_train, y_train, X_val, y_val, T, num_features, N,
                lr=0.001, max_epochs=200, patience=20, delta=0.05):
    model = build_lstm_classifier(T, num_features, N)
    # Pass a plain float LR; the LearningRateScheduler callback updates it
    # per epoch following the paper's piecewise-linear decay.
    model = compile_model(model, lr=lr)

    batch_size = compute_batch_size(len(X_train))

    callbacks = [
        _make_lr_schedule(initial_lr=lr, decay=0.0003,
                          decay_epochs=15, min_lr=0.0001),
        EarlyStoppingByLoss(patience=patience, delta=delta),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=max_epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    return model, history


def grid_search(X_train, y_train, X_val, y_val, T_values, N_values,
                num_features, lr=0.001, max_epochs=200, patience=20, delta=0.05):
    results = []
    best_val_loss = float("inf")
    best_model = None
    best_params = None
    best_history = None

    for T in T_values:
        for N in N_values:
            model, history = train_model(
                X_train, y_train, X_val, y_val,
                T, num_features, N,
                lr=lr, max_epochs=max_epochs,
                patience=patience, delta=delta,
            )
            val_loss = min(history.history["val_loss"])
            val_acc = max(history.history["val_accuracy"])
            results.append({
                "T": T, "N": N,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "epochs": len(history.history["loss"]),
            })
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model
                best_params = (T, N)
                best_history = history

    return best_model, best_params, best_history, results
