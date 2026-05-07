import pandas as pd
import numpy as np
from pathlib import Path


def load_trades(filepath: str) -> pd.DataFrame:
    df = pd.read_parquet(filepath)
    required = {"timestamp_ms", "price", "amount", "maker"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {missing} in {filepath}")
    df = df.sort_values("timestamp_ms").reset_index(drop=True)
    return df


def load_btc_usdt(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    train = load_trades(str(data_dir / "btc_usdt_train.parquet"))
    test = load_trades(str(data_dir / "btc_usdt_test.parquet"))
    return train, test


def load_other_pair(data_dir: str, symbol: str) -> pd.DataFrame:
    path = Path(data_dir) / f"{symbol}_test.parquet"
    return load_trades(str(path))
