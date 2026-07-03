"""P1 — Securities-Lending Fee & Rebate-Rate Forecasting.

Institutional-grade forecasting stack for hard-to-borrow (HTB) specialness
fees, combining:

1. An auditable Elastic Net linear tier (model-risk baseline).
2. A LightGBM quantile-regression tier (captures the convex utilization
   threshold effect above ~90% utilization) trained at the 0.1/0.5/0.9
   quantiles directly, rather than a single point-estimate model wrapped
   in a symmetric interval.
3. A split-conformal calibration layer (Lei et al., 2018; Romano et al.,
   2019 "Conformalized Quantile Regression") that guarantees *finite
   sample, distribution-free* marginal coverage of the prediction
   interval regardless of the base learner's miscalibration — this is
   the modern standard used by quantitative trading desks in place of
   naive Gaussian intervals.
4. A non-negative-least-squares blend of the Elastic Net and the GBM
   median forecast, weighted on a held-out walk-forward fold.

All validation is performed on expanding-window, walk-forward folds; a
random K-fold split is never used because fee observations are serially
autocorrelated and event-clustered (dividend / index-rebalance spikes),
which would otherwise leak information across the train/validation
boundary.
"""

from __future__ import annotations

import dataclasses

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.linear_model import ElasticNetCV

_QUANTILES = (0.1, 0.5, 0.9)


@dataclasses.dataclass(frozen=True)
class FeeForecastResult:
    """Container for a single walk-forward fold's forecast output.

    Attributes:
        dates: Forecast target dates.
        name_id: Security identifier per row.
        y_true: Realized fee (bps) at the forecast horizon.
        y_pred_blend: Blended point forecast (bps).
        y_pred_p10: Conformalized 10th-percentile forecast (bps).
        y_pred_p90: Conformalized 90th-percentile forecast (bps).
        weighted_mape: Notional-weighted MAPE for this fold (percent).
        directional_hit_rate: Hit-rate on |Δfee| > 25bps direction calls.
        pinball_loss_p10: Pinball loss at the 10th percentile.
        pinball_loss_p90: Pinball loss at the 90th percentile.
        coverage_80: Empirical coverage of the [P10, P90] interval.
    """

    dates: pd.Series
    name_id: pd.Series
    y_true: np.ndarray
    y_pred_blend: np.ndarray
    y_pred_p10: np.ndarray
    y_pred_p90: np.ndarray
    weighted_mape: float
    directional_hit_rate: float
    pinball_loss_p10: float
    pinball_loss_p90: float
    coverage_80: float


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """Computes the mean pinball (quantile) loss.

    The pinball loss is the proper scoring rule for quantile forecasts:
    it is asymmetric, penalizing under-prediction and over-prediction
    differently according to the target quantile ``q``. For ``q=0.5`` it
    reduces (up to a factor of two) to mean absolute error.

    Args:
        y_true: Realized values, shape (n,).
        y_pred: Forecast values for quantile ``quantile``, shape (n,).
        quantile: Target quantile in (0, 1).

    Returns:
        The mean pinball loss (lower is better).
    """
    diff = y_true - y_pred
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1) * diff)))


def weighted_mape(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray, min_fee_bps: float = 5.0) -> float:
    """Computes notional-weighted mean absolute percentage error.

    Per-name percentage error is weighted by loan notional so the metric
    reflects realized desk P&L impact rather than treating a $5,000 loan
    and a $50mm loan with equal importance. Names trading at near-zero
    fee (general collateral, not specialness) are excluded from the
    percentage-error calculation, consistent with desk practice: MAPE on
    a fee of 0.1bps is not economically meaningful and would otherwise
    dominate the metric with noise.

    Args:
        y_true: Realized fee, shape (n,).
        y_pred: Forecast fee, shape (n,).
        weights: Non-negative notional weights, shape (n,).
        min_fee_bps: Minimum realized fee (bps) for a row to count toward
            the metric — filters out uninformative GC-rate noise.

    Returns:
        Weighted MAPE over economically material (specialness) rows, as a percentage.
    """
    mask = np.abs(y_true) >= min_fee_bps
    if mask.sum() == 0:
        return float("nan")
    pct_err = np.abs((y_true[mask] - y_pred[mask]) / np.abs(y_true[mask]))
    return float(100.0 * np.average(pct_err, weights=weights[mask]))


def make_features(panel: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, list[str]]:
    """Builds the supervised learning frame with lagged and event features.

    Args:
        panel: Long-format panel with columns
            ``[date, name_id, utilization, days_to_cover, dividend_flag,
            gc_spread, fee_bps]`` sorted by ``(name_id, date)``.
        horizon: Forecast horizon in business days.

    Returns:
        A tuple of ``(frame, feature_columns)`` where ``frame`` carries a
        ``target`` column (fee at t+horizon) and rows with insufficient
        history dropped.
    """
    df = panel.sort_values(["name_id", "date"]).copy()
    grp = df.groupby("name_id")["fee_bps"]
    df["fee_lag1"] = grp.shift(1)
    df["fee_lag5"] = grp.shift(5)
    df["fee_ewm10"] = grp.transform(lambda s: s.shift(1).ewm(span=10, min_periods=1).mean())
    df["fee_mom_5d"] = df["fee_lag1"] - df["fee_lag5"]
    df["util_x_dtc"] = df["utilization"] * (1.0 / (df["days_to_cover"] + 1.0))
    df["util_squared_over_90"] = np.clip(df["utilization"] - 0.9, 0.0, None) ** 2
    df["target"] = df.groupby("name_id")["fee_bps"].shift(-horizon)
    df = df.dropna(subset=["target", "fee_lag5"]).reset_index(drop=True)
    feature_cols = [
        "utilization", "days_to_cover", "dividend_flag", "gc_spread",
        "fee_lag1", "fee_lag5", "fee_ewm10", "fee_mom_5d",
        "util_x_dtc", "util_squared_over_90",
    ]
    return df, feature_cols


def conformalize(
    residuals_calib: np.ndarray, y_pred_lo: np.ndarray, y_pred_hi: np.ndarray, alpha: float = 0.2
) -> tuple[np.ndarray, np.ndarray]:
    """Applies split-conformal calibration to LightGBM quantile forecasts.

    Implements Conformalized Quantile Regression (CQR, Romano, Patterson
    & Candès, 2019): the raw quantile-regression band is widened by the
    ``(1-alpha)``-quantile of the calibration-fold conformity scores,
    which guarantees marginal coverage of ``1-alpha`` regardless of how
    well the underlying quantile model is calibrated.

    Args:
        residuals_calib: Conformity scores on a held-out calibration
            fold, i.e. ``max(y_lo - y_true, y_true - y_hi)`` per row.
        y_pred_lo: Raw lower-quantile forecast on the test fold.
        y_pred_hi: Raw upper-quantile forecast on the test fold.
        alpha: Miscoverage rate (0.2 -> 80% interval).

    Returns:
        A tuple ``(y_lo_calibrated, y_hi_calibrated)``.
    """
    n = len(residuals_calib)
    q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    correction = float(np.quantile(residuals_calib, q_level))
    return y_pred_lo - correction, y_pred_hi + correction


def fit_and_evaluate_fold(
    train: pd.DataFrame, calib: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str]
) -> FeeForecastResult:
    """Fits the three-tier ensemble on one walk-forward fold and scores it.

    Args:
        train: Training rows (earliest in time).
        calib: Calibration rows (post-train, pre-test) used only for
            conformal interval calibration — never for point-forecast fitting.
        test: Held-out test rows (latest in time).
        feature_cols: Names of the feature columns shared across frames.

    Returns:
        A populated :class:`FeeForecastResult`.
    """
    x_train, y_train = train[feature_cols], train["target"].to_numpy()
    x_calib, y_calib = calib[feature_cols], calib["target"].to_numpy()
    x_test, y_test = test[feature_cols], test["target"].to_numpy()

    elastic_net = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0], cv=5, max_iter=10_000)
    elastic_net.fit(x_train, y_train)

    gbm_models: dict[float, lgb.LGBMRegressor] = {}
    for q in _QUANTILES:
        model = lgb.LGBMRegressor(
            objective="quantile", alpha=q, n_estimators=500, num_leaves=31,
            learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, verbosity=-1,
        )
        model.fit(x_train, y_train)
        gbm_models[q] = model

    en_pred_calib = elastic_net.predict(x_calib)
    gbm_median_calib = gbm_models[0.5].predict(x_calib)
    blend_pred_calib = np.column_stack([en_pred_calib, gbm_median_calib])
    weights, _ = nnls(blend_pred_calib, y_calib)
    weights = weights / weights.sum() if weights.sum() > 0 else np.array([0.5, 0.5])

    en_pred_test = elastic_net.predict(x_test)
    gbm_median_test = gbm_models[0.5].predict(x_test)
    y_pred_blend = np.column_stack([en_pred_test, gbm_median_test]) @ weights

    lo_calib_raw = gbm_models[0.1].predict(x_calib)
    hi_calib_raw = gbm_models[0.9].predict(x_calib)
    conformity_scores = np.maximum(lo_calib_raw - y_calib, y_calib - hi_calib_raw)

    lo_test_raw = gbm_models[0.1].predict(x_test)
    hi_test_raw = gbm_models[0.9].predict(x_test)
    y_pred_p10, y_pred_p90 = conformalize(conformity_scores, lo_test_raw, hi_test_raw, alpha=0.2)

    notional_weights = np.clip(test["utilization"].to_numpy(), 0.05, None)
    w_mape = weighted_mape(y_test, y_pred_blend, notional_weights)

    fee_lag1 = test["fee_lag1"].to_numpy()
    realized_move = y_test - fee_lag1
    predicted_move = y_pred_blend - fee_lag1
    material = np.abs(realized_move) > 25.0
    if material.sum() > 0:
        hit_rate = float(np.mean(np.sign(realized_move[material]) == np.sign(predicted_move[material])))
    else:
        hit_rate = float("nan")

    coverage = float(np.mean((y_test >= y_pred_p10) & (y_test <= y_pred_p90)))

    return FeeForecastResult(
        dates=test["date"].reset_index(drop=True),
        name_id=test["name_id"].reset_index(drop=True),
        y_true=y_test,
        y_pred_blend=y_pred_blend,
        y_pred_p10=y_pred_p10,
        y_pred_p90=y_pred_p90,
        weighted_mape=w_mape,
        directional_hit_rate=hit_rate,
        pinball_loss_p10=pinball_loss(y_test, y_pred_p10, 0.1),
        pinball_loss_p90=pinball_loss(y_test, y_pred_p90, 0.9),
        coverage_80=coverage,
    )


def walk_forward_backtest(
    panel: pd.DataFrame, horizon: int = 1, n_folds: int = 4
) -> list[FeeForecastResult]:
    """Runs an expanding-window walk-forward backtest across ``n_folds``.

    Each fold's test window is a disjoint, chronologically later slice of
    dates; the training window always begins at the start of history and
    expands, mirroring how the model would actually be retrained in
    production on a rolling basis.

    Args:
        panel: Raw long-format fee panel (see :func:`make_features`).
        horizon: Forecast horizon in business days.
        n_folds: Number of walk-forward folds to evaluate.

    Returns:
        A list of one :class:`FeeForecastResult` per fold, in chronological order.
    """
    frame, feature_cols = make_features(panel, horizon)
    dates_sorted = np.sort(frame["date"].unique())
    fold_edges = np.linspace(len(dates_sorted) * 0.5, len(dates_sorted), n_folds + 1).astype(int)

    results: list[FeeForecastResult] = []
    for i in range(n_folds):
        train_end = dates_sorted[fold_edges[i] - 1]
        calib_end_idx = min(fold_edges[i] + max(1, (fold_edges[i + 1] - fold_edges[i]) // 3), len(dates_sorted) - 1)
        calib_end = dates_sorted[calib_end_idx]
        test_end = dates_sorted[fold_edges[i + 1] - 1]

        train = frame[frame["date"] <= train_end]
        calib = frame[(frame["date"] > train_end) & (frame["date"] <= calib_end)]
        test = frame[(frame["date"] > calib_end) & (frame["date"] <= test_end)]
        if len(train) < 200 or len(calib) < 50 or len(test) < 50:
            continue
        results.append(fit_and_evaluate_fold(train, calib, test, feature_cols))
    return results


def _main() -> None:
    """Command-line entry point: runs a demo backtest on the shipped panel."""
    panel = pd.read_csv("data/sec_lending_panel.csv", parse_dates=["date"])
    panel = panel.rename(columns={"ticker": "name_id", "gc_rate": "gc_spread"})
    panel["gc_spread"] = panel["gc_spread"] - 5.0
    results = walk_forward_backtest(panel, horizon=1, n_folds=4)
    for i, r in enumerate(results):
        print(
            f"fold {i}: weighted_MAPE={r.weighted_mape:.2f}% "
            f"hit_rate={r.directional_hit_rate:.2%} "
            f"coverage_80={r.coverage_80:.2%}"
        )


if __name__ == "__main__":
    _main()
