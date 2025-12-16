import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# One scaler instance used across the project (fit once, then transform)
scaler = MinMaxScaler()

def fit_scaler(train_df: pd.DataFrame) -> None:
    """
    Fit the scaler only on training data to avoid data leakage.
    """
    scaler.fit(train_df[["temperature", "vibration", "pressure"]])

def preprocess(data) -> pd.DataFrame:
    """
    Convert incoming dict/list-of-dicts to a DataFrame and normalize features.
    """
    df = pd.DataFrame(data)
    df[["temperature", "vibration", "pressure"]] = scaler.transform(
        df[["temperature", "vibration", "pressure"]]
    )
    return df
