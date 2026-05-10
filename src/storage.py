import os
import pickle
import pandas as pd


def _has_parquet():
    try:
        import pyarrow  # noqa
        return True
    except ImportError:
        try:
            import fastparquet  # noqa
            return True
        except ImportError:
            return False


USE_PARQUET = _has_parquet()
EXT = ".parquet" if USE_PARQUET else ".pkl"


def save_df(df: pd.DataFrame, path_without_ext: str) -> str:
    """Save dataframe to parquet or pickle."""
    if USE_PARQUET:
        path = path_without_ext + ".parquet"
        df.to_parquet(path, index=False)
    else:
        path = path_without_ext + ".pkl"
        df.to_pickle(path)
    return path


def load_df(path_without_ext: str) -> pd.DataFrame:
    """Load dataframe from parquet or pickle (tries both)."""
    parquet_path = path_without_ext + ".parquet"
    pkl_path = path_without_ext + ".pkl"

    if os.path.exists(parquet_path) and USE_PARQUET:
        return pd.read_parquet(parquet_path)
    elif os.path.exists(pkl_path):
        return pd.read_pickle(pkl_path)
    elif os.path.exists(parquet_path):
        return pd.read_parquet(parquet_path)
    else:
        raise FileNotFoundError(
            f"No data file found at {parquet_path} or {pkl_path}. "
            "Run `python -m src.prepare_data` first."
        )
