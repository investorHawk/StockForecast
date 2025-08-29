from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from ..core.config import settings
from ..db.inspect import ensure_cache
from ..db.session import get_connection
from .features import build_features
from . import prices as prices_service
from sklearn.linear_model import ElasticNet, ElasticNetCV, QuantileRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.arima.model import ARIMA


@dataclass
class ForecastResult:
    dates: List[str]
    observed: List[float]
    predicted_path: List[float]
    quantiles: Dict[str, List[float]]
    diagnostics: Dict[str, Any]


def _get_raw_df(stock: str) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
    meta = ensure_cache()
    schema_map = meta.get("schema_map", {})
    extra = meta.get("extra_numeric_columns", [])
    code_col = schema_map.get("code")
    if not code_col:
        raise ValueError("Schema detection did not find code column")
    def qident(name: str) -> str:
        return '"' + str(name).replace('"', '""') + '"'
    with get_connection(settings.DB_PATH) as conn:
        cols = list(set([schema_map.get(k) for k in ("date", "open", "high", "low", "close", "volume", "code") if schema_map.get(k)] + extra))
        select_part = ", ".join(qident(c) for c in cols)
        sql = f"SELECT {select_part} FROM prices WHERE {qident(code_col)} = ? ORDER BY {qident(schema_map.get('date'))} ASC"
        df = pd.read_sql_query(sql, conn, params=(stock,))
    return df, schema_map, extra


def _business_day_range(start_date: pd.Timestamp, periods: int) -> List[str]:
    rng = pd.bdate_range(start=start_date, periods=periods + 1, freq="B").tolist()[1:]
    return [d.date().isoformat() for d in rng]


def _weekly_wf_cv_mae(
    raw: pd.DataFrame,
    schema_map: Dict[str, str],
    extras: List[str],
    lookback_days: int,
    horizon_eval: int,
    feature_mode: str,
    seed: int = 42,
) -> Dict[str, Dict[str, Optional[float]]]:
    close_col = schema_map.get("close")
    n = len(raw)
    step = 5  # ~1週
    max_slices = 5
    # 末尾から週次で複数スライスを作る
    slices = []
    end = n - step
    while end - lookback_days >= 10 and len(slices) < max_slices:
        train_end = end
        test_start = train_end
        test_end = min(n, test_start + step)
        slices.append((train_end, test_start, test_end))
        end -= step
    if not slices:
        return {}

    mae_acc = {"baseline": [], "direct": [], "multistep": []}
    for (train_end, test_start, test_end) in slices:
        raw_train = raw.iloc[:train_end].copy()
        actual = raw.iloc[test_start:test_end][close_col].astype(float).tolist()
        horizon = min(horizon_eval, len(actual))
        if horizon <= 0:
            continue
        # baseline
        p50_b, _ = baseline_forecast(raw_train.iloc[-lookback_days:][close_col], horizon)
        e = np.array(p50_b) - np.array(actual[:horizon])
        mae_acc["baseline"].append(float(np.mean(np.abs(e))))
        # direct
        try:
            p50_d, _, _ = direct_forecast_from_raw(
                raw_train, schema_map, extras, lookback_days, horizon, feature_mode, seed
            )
        except Exception:
            # ARIMAバックアップ
            p50_d, _q = arima_forecast(raw_train.iloc[-lookback_days:][close_col], horizon)
        e = np.array(p50_d) - np.array(actual[:horizon])
        mae_acc["direct"].append(float(np.mean(np.abs(e))))
        # multistep
        try:
            p50_m, _, _ = multistep_forecast_from_raw(
                raw_train, schema_map, extras, lookback_days, horizon, feature_mode, seed
            )
            e = np.array(p50_m) - np.array(actual[:horizon])
            mae_acc["multistep"].append(float(np.mean(np.abs(e))))
        except Exception:
            pass

    out: Dict[str, Dict[str, Optional[float]]] = {}
    for k, arr in mae_acc.items():
        out[k] = {"mae": (float(np.mean(arr)) if arr else None)}
    return out


def baseline_forecast(
    close_series: pd.Series, horizon_days: int
) -> Tuple[List[float], Dict[str, List[float]]]:
    # Daily log returns
    logret = np.log(close_series).diff().dropna()
    mu = float(logret.mean())
    sigma = float(logret.std(ddof=1) if len(logret) > 1 else 0.0)

    last_price = float(close_series.iloc[-1])
    p50 = []
    p10 = []
    p90 = []
    for t in range(1, horizon_days + 1):
        mean_cum = mu * t
        std_cum = sigma * (t ** 0.5)
        p50.append(last_price * float(np.exp(mean_cum)))
        # Using normal quantiles
        z10 = -1.2815515655446004
        z90 = 1.2815515655446004
        p10.append(last_price * float(np.exp(mean_cum + z10 * std_cum)))
        p90.append(last_price * float(np.exp(mean_cum + z90 * std_cum)))
    return p50, {"p10": p10, "p50": p50, "p90": p90}


def _quantile_gbr_models(Xs: np.ndarray, y: np.ndarray, seed: int = 42):
    # Train three quantile models at 0.1/0.5/0.9
    q10 = GradientBoostingRegressor(
        loss="quantile", alpha=0.1, n_estimators=400, learning_rate=0.03, max_depth=3, subsample=0.9, random_state=seed
    ).fit(Xs, y)
    q50 = GradientBoostingRegressor(
        loss="quantile", alpha=0.5, n_estimators=400, learning_rate=0.03, max_depth=3, subsample=0.9, random_state=seed
    ).fit(Xs, y)
    q90 = GradientBoostingRegressor(
        loss="quantile", alpha=0.9, n_estimators=400, learning_rate=0.03, max_depth=3, subsample=0.9, random_state=seed
    ).fit(Xs, y)
    return q10, q50, q90


def quantile_direct_gbr_forecast_from_raw(
    raw: pd.DataFrame,
    schema_map: Dict[str, str],
    extras: List[str],
    lookback_days: int,
    horizon_days: int,
    feature_mode: str,
    seed: int = 42,
) -> Tuple[List[float], Dict[str, List[float]], Dict[str, Any]]:
    # Similar to direct, but learn conditional quantiles for cumulative return over horizon
    np.random.seed(seed)
    date_col = schema_map.get("date")
    close_col = schema_map.get("close")
    df = raw.sort_values(by=[date_col]).reset_index(drop=True)
    feat_df, feature_cols = build_features(df, schema_map, extras, feature_mode)

    close = df[close_col].astype(float).values
    last_idx = len(df) - 1
    t_end = last_idx - horizon_days
    t_start = max(0, last_idx - lookback_days - horizon_days + 1)

    rows_X = []
    rows_y = []
    for t in range(t_start, t_end + 1):
        xi = feat_df.iloc[t][feature_cols].astype(float).fillna(0.0)
        yi = float(np.log(close[t + horizon_days]) - np.log(close[t]))
        if not np.isfinite(yi):
            continue
        rows_X.append(xi.values.astype(float))
        rows_y.append(yi)
    if len(rows_X) < 5:
        raise ValueError("Insufficient training samples for quantile direct method")

    X = np.vstack(rows_X)
    y = np.array(rows_y)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    m10, m50, m90 = _quantile_gbr_models(Xs, y, seed)

    x_last = feat_df.iloc[last_idx][feature_cols].astype(float).fillna(0.0).values.reshape(1, -1)
    x_last_s = scaler.transform(x_last)
    y10 = float(m10.predict(x_last_s)[0])
    y50 = float(m50.predict(x_last_s)[0])
    y90 = float(m90.predict(x_last_s)[0])

    last_price = float(close[-1])
    p50 = []
    p10 = []
    p90 = []
    for t in range(1, horizon_days + 1):
        scale = t / float(horizon_days)
        p50.append(last_price * float(np.exp(scale * y50)))
        p10.append(last_price * float(np.exp(scale * y10)))
        p90.append(last_price * float(np.exp(scale * y90)))

    diag = {
        "model": "gbr_quantile_direct",
        "n_train": int(len(y)),
        "yhat": {"p10": y10, "p50": y50, "p90": y90},
    }
    return p50, {"p10": p10, "p50": p50, "p90": p90}, diag


def _conformal_adjust_prices(
    pred_p10: List[float], pred_p50: List[float], pred_p90: List[float],
    calib_p10: List[float], calib_p50: List[float], calib_p90: List[float],
    actual: List[float],
    train_close_series: pd.Series,
) -> Tuple[List[float], List[float], List[float]]:
    # Compute split-conformal adjustments on calibration window
    a_lo = None
    a_hi = None
    if calib_p10 and calib_p90 and actual:
        n = min(len(calib_p10), len(calib_p90), len(actual))
        lo_scores = [calib_p10[i] - actual[i] for i in range(n)]
        hi_scores = [actual[i] - calib_p90[i] for i in range(n)]
        try:
            a_lo = float(np.quantile(lo_scores, 0.9))
            a_hi = float(np.quantile(hi_scores, 0.9))
        except Exception:
            a_lo = None
            a_hi = None

    p10_adj = pred_p10[:]
    p90_adj = pred_p90[:]
    if a_lo is not None:
        p10_adj = [max(0.0, v - a_lo) for v in p10_adj]
    if a_hi is not None:
        p90_adj = [v + a_hi for v in p90_adj]

    # Volatility scaling using rolling std of log returns (recent vs calibration proxy)
    try:
        logret = np.log(train_close_series.astype(float)).diff().dropna()
        s_recent = float(logret.tail(63).std()) if len(logret) >= 10 else float(logret.std())
        s_cal = float(logret.iloc[-126:-63].std()) if len(logret) >= 200 else s_recent
        if s_recent and s_cal:
            r = max(0.5, min(2.0, s_recent / s_cal))
            # scale distances from p50
            p10_adj = [p50 - (p50 - p10_) * r for p50, p10_ in zip(pred_p50, p10_adj)]
            p90_adj = [p50 + (p90_ - p50) * r for p50, p90_ in zip(pred_p50, p90_adj)]
    except Exception:
        pass

    return p10_adj, pred_p50, p90_adj


def arima_forecast(close_series: pd.Series, horizon_days: int) -> Tuple[List[float], Dict[str, List[float]]]:
    cs = close_series.dropna().astype(float)
    if len(cs) < 20:
        return baseline_forecast(cs, horizon_days)
    logret = np.log(cs).diff().dropna()
    if len(logret) < 10:
        return baseline_forecast(cs, horizon_days)
    try:
        model = ARIMA(logret.values, order=(1, 0, 1))
        fit = model.fit()
        fc = fit.get_forecast(steps=horizon_days)
        mean = fc.predicted_mean
        se = fc.se_mean
        z10 = -1.2815515655446004
        z90 = 1.2815515655446004
        last_price = float(cs.iloc[-1])
        p50 = []
        p10 = []
        p90 = []
        cum_mean = 0.0
        for i in range(horizon_days):
            mu = float(mean[i])
            s = float(se[i])
            cum_mean += mu
            p50.append(last_price * float(np.exp(cum_mean)))
            p10.append(last_price * float(np.exp(cum_mean + z10 * s)))
            p90.append(last_price * float(np.exp(cum_mean + z90 * s)))
        return p50, {"p10": p10, "p50": p50, "p90": p90}
    except Exception:
        return baseline_forecast(cs, horizon_days)


def direct_forecast_from_raw(
    raw: pd.DataFrame,
    schema_map: Dict[str, str],
    extras: List[str],
    lookback_days: int,
    horizon_days: int,
    feature_mode: str,
    seed: int = 42,
) -> Tuple[List[float], Dict[str, List[float]], Dict[str, Any]]:
    np.random.seed(seed)
    date_col = schema_map.get("date")
    close_col = schema_map.get("close")
    df = raw.sort_values(by=[date_col]).reset_index(drop=True)
    feat_df, feature_cols = build_features(df, schema_map, extras, feature_mode)

    close = df[close_col].astype(float).values
    last_idx = len(df) - 1
    t_end = last_idx - horizon_days
    t_start = max(0, last_idx - lookback_days - horizon_days + 1)

    rows_X = []
    rows_y = []
    for t in range(t_start, t_end + 1):
        xi = feat_df.iloc[t][feature_cols].astype(float).fillna(0.0)
        yi = float(np.log(close[t + horizon_days]) - np.log(close[t]))
        if not np.isfinite(yi):
            continue
        rows_X.append(xi.values.astype(float))
        rows_y.append(yi)

    if len(rows_X) < 5:
        raise ValueError("Insufficient training samples for direct method")

    X = np.vstack(rows_X)
    y = np.array(rows_y)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    # ElasticNetCV with time-series split
    n_splits = 3 if len(y) >= 30 else 2
    tscv = TimeSeriesSplit(n_splits=n_splits)
    model = ElasticNetCV(
        l1_ratio=[0.2, 0.5, 0.8],
        alphas=np.logspace(-4, 0, 20),
        cv=tscv,
        max_iter=20000,
        n_jobs=None,
    )
    model.fit(Xs, y)
    y_hat = model.predict(Xs)
    resid = y_hat - y
    mae = float(np.mean(np.abs(resid)))
    rmse = float(np.sqrt(np.mean(resid**2)))
    sigma_y = float(np.std(resid, ddof=1)) if len(resid) > 1 else 0.0
    # Empirical residual quantiles
    q10_res = float(np.percentile(resid, 10)) if len(resid) > 5 else -1.2815515655446004 * sigma_y
    q90_res = float(np.percentile(resid, 90)) if len(resid) > 5 else 1.2815515655446004 * sigma_y

    # Quantile Regression for 0.1 and 0.9 (帯強化)
    y_q10_pred = None
    y_q90_pred = None
    try:
        q10 = QuantileRegressor(quantile=0.1, alpha=1e-4, solver='highs').fit(Xs, y)
        q90 = QuantileRegressor(quantile=0.9, alpha=1e-4, solver='highs').fit(Xs, y)
        # 予測は最新特徴量で実施
        # 注意: QuantileRegressorはスケール済み入力を想定
        # x_last_s は後段で計算
        use_qreg = True
    except Exception:
        use_qreg = False

    # Inference at last observed day
    x_last = feat_df.iloc[last_idx][feature_cols].astype(float).fillna(0.0).values.reshape(1, -1)
    x_last_s = scaler.transform(x_last)
    y_pred = float(model.predict(x_last_s)[0])
    daily_mu = y_pred / float(horizon_days)

    last_price = float(close[-1])
    p50 = []
    p10 = []
    p90 = []
    if use_qreg:
        try:
            y_q10_pred = float(q10.predict(x_last_s)[0])
            y_q90_pred = float(q90.predict(x_last_s)[0])
        except Exception:
            use_qreg = False

    for t in range(1, horizon_days + 1):
        mean_cum = daily_mu * t
        p50.append(last_price * float(np.exp(mean_cum)))
        if use_qreg and y_q10_pred is not None and y_q90_pred is not None:
            scale = t / float(horizon_days)
            p10.append(last_price * float(np.exp(scale * y_q10_pred)))
            p90.append(last_price * float(np.exp(scale * y_q90_pred)))
        else:
            scale = np.sqrt(t / float(horizon_days))
            p10.append(last_price * float(np.exp(mean_cum + q10_res * scale)))
            p90.append(last_price * float(np.exp(mean_cum + q90_res * scale)))

    diag = {
        "model": "elasticnetcv+quantile",
        "n_train": int(len(y)),
        "train": {"mae": mae, "rmse": rmse, "sigma_y": sigma_y},
        "coef_nonzero": int(np.sum(np.abs(model.coef_) > 1e-8)),
        "alpha_": float(model.alpha_),
        "l1_ratio_": float(model.l1_ratio_) if hasattr(model, 'l1_ratio_') else None,
        "quantile_used": bool(use_qreg),
    }

    return p50, {"p10": p10, "p50": p50, "p90": p90}, diag


def multistep_forecast_from_raw(
    raw: pd.DataFrame,
    schema_map: Dict[str, str],
    extras: List[str],
    lookback_days: int,
    horizon_days: int,
    feature_mode: str,
    seed: int = 42,
) -> Tuple[List[float], Dict[str, List[float]], Dict[str, Any]]:
    np.random.seed(seed)
    date_col = schema_map.get("date")
    close_col = schema_map.get("close")
    df = raw.sort_values(by=[date_col]).reset_index(drop=True)
    feat_df, feature_cols = build_features(df, schema_map, extras, feature_mode)

    close = df[close_col].astype(float).values
    last_idx = len(df) - 1
    t_end = last_idx - 1
    t_start = max(0, last_idx - lookback_days)

    rows_X = []
    rows_y = []
    for t in range(t_start, t_end + 1):
        xi = feat_df.iloc[t][feature_cols].astype(float).fillna(0.0)
        yi = float(np.log(close[t + 1]) - np.log(close[t]))
        if not np.isfinite(yi):
            continue
        rows_X.append(xi.values.astype(float))
        rows_y.append(yi)

    if len(rows_X) < 5:
        raise ValueError("Insufficient training samples for multistep method")

    X = np.vstack(rows_X)
    y = np.array(rows_y)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    gbr = GradientBoostingRegressor(
        n_estimators=400, learning_rate=0.03, max_depth=3, subsample=0.9, random_state=seed
    )
    gbr.fit(Xs, y)
    y_hat = gbr.predict(Xs)
    resid = y_hat - y
    mae = float(np.mean(np.abs(resid)))
    rmse = float(np.sqrt(np.mean(resid**2)))
    sigma_r = float(np.std(resid, ddof=1)) if len(resid) > 1 else 0.0

    # Predict next-day return iteratively with residual bootstrap
    x_vec = feat_df.iloc[last_idx][feature_cols].astype(float).fillna(0.0).values
    last_price = float(close[-1])

    # Helper: update features by shifting in predicted r
    lag_idx = {}
    for i, name in enumerate(feature_cols):
        if name.startswith('lag_ret_'):
            try:
                l = int(name.split('_')[-1])
                lag_idx[l] = i
            except Exception:
                pass
    roll_mean_idx = {}
    roll_std_idx = {}
    for i, name in enumerate(feature_cols):
        if name.startswith('roll_mean_'):
            try:
                w = int(name.split('_')[-1])
                roll_mean_idx[w] = i
            except Exception:
                pass
        if name.startswith('roll_std_'):
            try:
                w = int(name.split('_')[-1])
                roll_std_idx[w] = i
            except Exception:
                pass

    def update_lags_and_roll(x: np.ndarray, r: float) -> np.ndarray:
        if not lag_idx:
            pass
        # shift from high lag downwards
        for l in sorted(lag_idx.keys(), reverse=True):
            if l == 1:
                x[lag_idx[l]] = r
            else:
                prev = lag_idx.get(l - 1)
                if prev is not None:
                    x[lag_idx[l]] = x[prev]
        # approximate rolling mean/std updates (for logret) using simple rolling formula
        for w, idx in roll_mean_idx.items():
            old = float(x[idx]) if np.isfinite(x[idx]) else 0.0
            x[idx] = ((w - 1) / w) * old + (1.0 / w) * r
        for w, idx in roll_std_idx.items():
            old_std = float(x[idx]) if np.isfinite(x[idx]) else 0.0
            # rough update assuming rolling variance mix
            # new_var ≈ ((w-1)/w)*old_var + (1/w)*(r - roll_mean_w)^2
            mean_idx = roll_mean_idx.get(w)
            m = float(x[mean_idx]) if (mean_idx is not None and np.isfinite(x[mean_idx])) else 0.0
            new_var = ((w - 1) / w) * (old_std ** 2) + (1.0 / w) * ((r - m) ** 2)
            x[idx] = float(np.sqrt(max(new_var, 0.0)))
        return x

    # Deterministic p50 using iterative predictions (features partially updated)
    prices_path = []
    p = last_price
    x_curr = x_vec.copy()
    for t in range(horizon_days):
        r_hat = float(gbr.predict(scaler.transform(x_curr.reshape(1, -1)))[0])
        p = p * float(np.exp(r_hat))
        prices_path.append(p)
        x_curr = update_lags_and_roll(x_curr, r_hat)

    # Residual bootstrap for bands
    B = min(300, max(50, 10 * int(np.sqrt(horizon_days))))
    resid_choices = resid if len(resid) > 0 else np.array([0.0])
    paths = np.zeros((B, horizon_days), dtype=float)
    for b in range(B):
        p = last_price
        x_bt = x_vec.copy()
        for t in range(horizon_days):
            r_hat = float(gbr.predict(scaler.transform(x_bt.reshape(1, -1)))[0])
            eps = float(np.random.choice(resid_choices))
            r_sim = r_hat + eps
            p = p * float(np.exp(r_sim))
            paths[b, t] = p
            x_bt = update_lags_and_roll(x_bt, r_sim)

    p50 = paths.median(axis=0).tolist()
    p10 = np.percentile(paths, 10, axis=0).tolist()
    p90 = np.percentile(paths, 90, axis=0).tolist()

    diag = {
        "model": "gbr_1step_iter_bootstrap",
        "n_train": int(len(y)),
        "train": {"mae": mae, "rmse": rmse, "sigma_r": sigma_r},
    }

    return p50, {"p10": p10, "p50": p50, "p90": p90}, diag


def forecast(
    stock: str,
    lookback_days: int = 126,
    horizon_days: int = 63,
    method: str = "ensemble",
    feature_mode: str = "auto",
    seed: int = 42,
) -> ForecastResult:
    np.random.seed(seed)
    raw, schema_map, extras = _get_raw_df(stock)
    date_col = schema_map.get("date")
    close_col = schema_map.get("close")
    if raw.empty or close_col not in raw.columns:
        raise ValueError("No data for the requested stock or missing close column")

    # Use last lookback_days window for training
    if len(raw) < lookback_days + 5:
        raise ValueError(f"Insufficient data: need at least {lookback_days + 5} rows, got {len(raw)}")

    train = raw.iloc[-lookback_days:].copy()

    # Build features (even if baseline uses only close, we prepare for diagnostics/other methods)
    feat_df, feature_cols = build_features(raw, schema_map, extras, feature_mode)

    # Baseline prediction (default)
    p50, q = baseline_forecast(train[close_col], horizon_days)

    # Horizon dates: generate business days from the last observed date
    all_dates = raw[date_col].astype(str).tolist()
    last_obs_date = pd.to_datetime(all_dates[-1])
    horizon_dates = _business_day_range(last_obs_date, horizon_days)

    observed = train[close_col].astype(float).tolist()

    diagnostics = {
        "method": method,
        "components": {"baseline": {"mu": float(np.log(train[close_col]).diff().dropna().mean()), "sigma": float(np.log(train[close_col]).diff().dropna().std(ddof=1) if len(train) > 1 else 0.0)}},
        "cv": {"mae": None, "rmse": None},
        "feature_mode": feature_mode,
        "schema_map": {
            "code": schema_map.get("code"),
            "date": schema_map.get("date"),
            "close": schema_map.get("close"),
            "open": schema_map.get("open"),
            "high": schema_map.get("high"),
            "low": schema_map.get("low"),
            "volume": schema_map.get("volume"),
        },
    }

    # Weekly WF-CV (MAE) for ensemble weighting
    cv_detail: Dict[str, Any] = {}
    try:
        cv_mae = _weekly_wf_cv_mae(raw, schema_map, extras, lookback_days, min(5, horizon_days), feature_mode, seed)
        if cv_mae:
            cv_detail = cv_mae
            diagnostics["cv"] = cv_detail
    except Exception:
        diagnostics["cv"] = {"baseline": {"mae": None}, "direct": {"mae": None}, "multistep": {"mae": None}}

    # Compute method outputs
    outputs: Dict[str, Any] = {}
    outputs["baseline"] = (p50, q, {"status": "ok"})
    direct_diag = None
    multistep_diag = None
    try:
        # Prefer quantile-GBR direct; fallback to ElasticNetCV
        try:
            p50_d, q_d, direct_diag = quantile_direct_gbr_forecast_from_raw(
                raw, schema_map, extras, lookback_days, horizon_days, feature_mode, seed
            )
        except Exception:
            p50_d, q_d, direct_diag = direct_forecast_from_raw(
                raw, schema_map, extras, lookback_days, horizon_days, feature_mode, seed
            )
        outputs["direct"] = (p50_d, q_d, direct_diag)
    except Exception as e:
        # ARIMAバックアップに切替
        try:
            p50_a, q_a = arima_forecast(train[close_col], horizon_days)
            outputs["direct"] = (p50_a, q_a, {"model": "arima_backup", "error": str(e)})
        except Exception as e2:
            diagnostics.setdefault("components", {}).setdefault("direct", {})
            diagnostics["components"]["direct"].update({"status": "error", "error": f"{e}; backup:{e2}"})
    try:
        p50_m, q_m, multistep_diag = multistep_forecast_from_raw(
            raw, schema_map, extras, lookback_days, horizon_days, feature_mode, seed
        )
        outputs["multistep"] = (p50_m, q_m, multistep_diag)
    except Exception as e:
        diagnostics.setdefault("components", {}).setdefault("multistep", {})
        diagnostics["components"]["multistep"].update({"status": "error", "error": str(e)})

    # Choose output based on method
    if method == "baseline":
        chosen = "baseline"
    elif method == "direct" and "direct" in outputs:
        chosen = "direct"
    elif method == "multistep" and "multistep" in outputs:
        chosen = "multistep"
    else:
        chosen = "ensemble"

    if chosen == "ensemble":
        # compute stabilized softmax weights from CV MAE
        maes = {}
        for k in ("baseline", "direct", "multistep"):
            v = cv_detail.get(k) if isinstance(diagnostics.get("cv"), dict) else None
            mae = v.get("mae") if isinstance(v, dict) else None
            if mae is not None and k in outputs:
                maes[k] = float(mae)
        if not maes:
            keys = [k for k in ("baseline", "direct", "multistep") if k in outputs]
            w = {k: 1.0 / len(keys) for k in keys}
        else:
            arr = np.array([maes[k] for k in maes.keys()], dtype=float)
            tau = float(np.median(arr)) if np.median(arr) > 0 else 1.0
            logits = -arr / tau
            exps = np.exp(logits - np.max(logits))
            exps = exps / (np.sum(exps) + 1e-12)
            w = {k: float(exps[i]) for i, k in enumerate(maes.keys())}
            # clip weights lightly for stability
            if len(w) > 1:
                eps = 0.05
                w = {k: max(eps, min(1.0 - eps, v)) for k, v in w.items()}
                s = sum(w.values())
                w = {k: v / s for k, v in w.items()}

        # element-wise weighted sum
        horizon = horizon_days
        p50_comb = [0.0] * horizon
        p10_comb = [0.0] * horizon
        p90_comb = [0.0] * horizon
        for k, wk in w.items():
            p50_k, q_k, _ = outputs[k]
            p10_k = q_k["p10"]
            p90_k = q_k["p90"]
            for i in range(horizon):
                p50_comb[i] += wk * p50_k[i]
                p10_comb[i] += wk * p10_k[i]
                p90_comb[i] += wk * p90_k[i]
        p50, q = p50_comb, {"p10": p10_comb, "p50": p50_comb, "p90": p90_comb}
        diagnostics.setdefault("components", {})
        if direct_diag:
            diagnostics["components"]["direct"] = {"status": "ok", **direct_diag}
        if multistep_diag:
            diagnostics["components"]["multistep"] = {"status": "ok", **multistep_diag}
        diagnostics["components"]["weights"] = w
    else:
        p50, q, diag_chosen = outputs.get(chosen, outputs["baseline"])
        diagnostics.setdefault("components", {})
        name = chosen
        if chosen in ("direct", "multistep") and diag_chosen:
            diagnostics["components"][name] = {"status": "ok", **diag_chosen}
        diagnostics["components"].setdefault("baseline", {"status": "ok"})

    # Conformal calibration on chosen output (or ensemble) using last few days
    try:
        cv_len = min(15, max(5, horizon_days))
        if len(raw) >= lookback_days + cv_len + 5:
            raw_cv = raw.iloc[: len(raw) - cv_len].copy()
            actual_cv = raw.iloc[-cv_len:][close_col].astype(float).tolist()
            # Get method-specific predictions for cv_len
            if chosen == "baseline":
                p50_cv, q_cv = baseline_forecast(raw_cv.iloc[-lookback_days:][close_col], cv_len)
            elif chosen == "direct" and "direct" in outputs:
                try:
                    p50_cv, q_cv, _ = quantile_direct_gbr_forecast_from_raw(
                        raw_cv, schema_map, extras, lookback_days, cv_len, feature_mode, seed
                    )
                except Exception:
                    p50_cv, q_cv, _ = direct_forecast_from_raw(
                        raw_cv, schema_map, extras, lookback_days, cv_len, feature_mode, seed
                    )
            elif chosen == "multistep" and "multistep" in outputs:
                p50_cv, q_cv, _ = multistep_forecast_from_raw(
                    raw_cv, schema_map, extras, lookback_days, cv_len, feature_mode, seed
                )
            else:
                # ensemble calibration: combine cv predictions if available, else skip
                # fallback to baseline
                p50_cv, q_cv = baseline_forecast(raw_cv.iloc[-lookback_days:][close_col], cv_len)

            p10_adj, p50_adj, p90_adj = _conformal_adjust_prices(
                q.get("p10", []), p50, q.get("p90", []),
                q_cv.get("p10", []), p50_cv, q_cv.get("p90", []),
                actual_cv,
                train[close_col],
            )
            p50 = p50_adj
            q = {"p10": p10_adj, "p50": p50_adj, "p90": p90_adj}
            diagnostics.setdefault("calibration", {})
            diagnostics["calibration"]["method"] = "split_conformal+vol_scale"
            diagnostics["calibration"]["cv_len"] = cv_len
    except Exception:
        pass

    return ForecastResult(
        dates=horizon_dates,
        observed=observed,
        predicted_path=p50,
        quantiles=q,
        diagnostics=diagnostics,
    )
