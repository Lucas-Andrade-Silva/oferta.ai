from pathlib import Path

import pandas as pd

from fii_analytics.config import settings


def cache_path(name: str) -> Path:
    path = Path(settings.cache_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path / name


def save_csv_cache(df: pd.DataFrame, name: str) -> Path:
    path = cache_path(name)
    df.to_csv(path, index=False)
    return path

