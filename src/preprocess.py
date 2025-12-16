import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Global scaler instance (fit once on training data)
scaler = MinMaxScaler()

def fit_scaler(train_df: pd.DataFrame) -> None:
    """
    Fit the scaler on training data only to avoid data leakage.
    """
    scaler.fit(train_df[["temperature", "vibration", "pressure"]])

def preprocess(data) -> pd.DataFrame:
    """
    Convert incoming dict or list of dicts to a DataFrame and normalize features.
    """
    df = pd.DataFrame(data)
    df[["temperature", "vibration", "pressure"]] = scaler.transform(
        df[["temperature", "vibration", "pressure"]]
    )
    return df
