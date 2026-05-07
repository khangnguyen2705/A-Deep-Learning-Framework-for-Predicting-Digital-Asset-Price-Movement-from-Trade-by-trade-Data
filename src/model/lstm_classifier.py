from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Softmax, Input
from tensorflow.keras.optimizers import Adam


def build_lstm_classifier(T: int, num_features: int, N: int) -> Sequential:
    """Paper Section IV (Fig. 4) architecture: LSTM(N) → Dropout(0.5) → Dense(2) → Softmax."""
    model = Sequential([
        Input(shape=(T, num_features)),
        LSTM(units=N),
        Dropout(rate=0.5),
        Dense(2),
        Softmax(),
    ])
    return model


def compile_model(model: Sequential, lr: float = 0.001) -> Sequential:
    optimizer = Adam(learning_rate=lr)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"],
        jit_compile=False,
    )
    return model
