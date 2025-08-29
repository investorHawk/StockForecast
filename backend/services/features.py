from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class FeatureConfig:
    feature_mode: str = "auto"  # auto | ohlcv_only | all_numeric
    lags: List[int] = None
    roll_windows: List[int] = None

    def __post_init__(self):
        if self.lags is None:
            self.lags = [1, 5, 10, 21, 63]
        if self.roll_windows is None:
            self.roll_windows = [21, 63]


def _safe_pct_change(s: pd.Series) -> pd.Series:
    return s.replace(0, np.nan).pct_change()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(period, min_periods=period).mean()
    roll_down = down.rolling(period, min_periods=period).mean()
    rs = roll_up / roll_down
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def add_ohlcv_features(df: pd.DataFrame, schema_map: Dict[str, str], cfg: FeatureConfig) -> pd.DataFrame:
    out = df.copy()
    close = schema_map.get("close")
    open_col = schema_map.get("open")
    high = schema_map.get("high")
    low = schema_map.get("low")
    vol = schema_map.get("volume")

    # Returns
    out["ret1"] = out[close].pct_change()
    out["logret1"] = np.log(out[close]).diff()

    for l in cfg.lags:
        out[f"lag_ret_{l}"] = out["logret1"].shift(l)

    for w in cfg.roll_windows:
        out[f"roll_std_{w}"] = out["logret1"].rolling(w, min_periods=max(2, w // 2)).std()
        out[f"roll_mean_{w}"] = out["logret1"].rolling(w, min_periods=max(2, w // 2)).mean()

    if vol and vol in out.columns:
        out["vol_z"] = (out[vol] - out[vol].rolling(63, min_periods=10).mean()) / (
            out[vol].rolling(63, min_periods=10).std()
        )

    if all(c in out.columns for c in [open_col, high, low, close] if c):
        # Parkinson volatility estimator
        if high and low:
            hl = np.log(out[high] / out[low])
            out["vol_parkinson_21"] = (hl ** 2).rolling(21, min_periods=10).mean()
        # RSI
        out["rsi14"] = compute_rsi(out[close], period=14)

    return out


def select_extra_numeric(
    df: pd.DataFrame, schema_map: Dict[str, str], extra_candidates: List[str], mode: str
) -> List[str]:
    if mode == "ohlcv_only":
        return []
    if mode == "all_numeric":
        return [c for c in extra_candidates if c in df.columns]
    # auto
    close = schema_map.get("close")
    y = np.log(df[close]).diff().shift(-1)  # next-day log return for correlation
    selected: List[str] = []
    for c in extra_candidates:
        if c not in df.columns:
            continue
        s = df[c]
        missing_rate = s.isna().mean()
        if missing_rate > 0.3:
            continue
        corr = pd.concat([s, y], axis=1).dropna().corr().iloc[0, 1]
        if np.isnan(corr):
            continue
        if abs(corr) >= 0.05:
            selected.append(c)
    return selected


def build_features(
    raw: pd.DataFrame,
    schema_map: Dict[str, str],
    extra_candidates: List[str],
    feature_mode: str = "auto",
) -> Tuple[pd.DataFrame, List[str]]:
    cfg = FeatureConfig(feature_mode)

    df = raw.copy()
    # Sort by date to ensure correct time order
    date_col = schema_map.get("date")
    if date_col in df.columns:
        df = df.sort_values(by=[date_col]).reset_index(drop=True)

    df = add_ohlcv_features(df, schema_map, cfg)

    # Select extra numeric columns based on mode
    extras = select_extra_numeric(df, schema_map, extra_candidates, feature_mode)

    # Z-score extra columns using expanding stats to avoid leakage
    for c in extras:
        mean_exp = df[c].expanding(min_periods=30).mean()
        std_exp = df[c].expanding(min_periods=30).std()
        df[f"z_{c}"] = (df[c] - mean_exp) / std_exp

    feature_cols = [
        c
        for c in df.columns
        if c
        not in set(
            [schema_map.get(k) for k in ("code", "date", "open", "high", "low", "close", "volume")]
        )
    ]

    # Drop columns with excessive NaNs
    feature_cols = [c for c in feature_cols if df[c].isna().mean() < 0.4]

    df = df[[col for col in [schema_map.get("date"), schema_map.get("close")] if col] + feature_cols]
    return df, feature_cols

