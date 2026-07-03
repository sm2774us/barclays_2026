"""P2 — Client Margin & Haircut Optimization.

Implements a two-layer initial-margin (IM) model:

1. A LASSO-selected, log-linear per-asset haircut model — the auditable
   layer a model-risk reviewer can sign off on coefficient-by-coefficient.
2. A gradient-boosted "correlation-breakdown add-on" trained on the
   *residual* of historical-simulation shortfall not explained by the
   linear layer — i.e. the portfolio-level tail effect of correlations
   moving toward one under stress, which a sum-of-independent-haircuts
   model structurally cannot capture (this is the Basel/ISDA-SIMM
   "non-additivity of risk" critique of naive haircut grids).

Backtesting follows the two canonical regulatory tests for VaR-style risk
measures:

* **Kupiec (1995) Proportion-of-Failures (POF) test** — tests whether the
  observed breach frequency matches the target confidence level.
* **Christoffersen (1998) Independence test** — tests whether breaches
  cluster in time (a sign of a stale or slow-to-adapt model), which the
  Kupiec test alone cannot detect.

Both are implemented as likelihood-ratio tests against a chi-squared(1)
null, exactly as specified in Basel Committee traffic-light backtesting
guidance.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2
from sklearn.linear_model import LassoCV
from sklearn.ensemble import GradientBoostingRegressor


@dataclasses.dataclass(frozen=True)
class BacktestResult:
    """Regulatory VaR backtest statistics for a single portfolio/model.

    Attributes:
        n_obs: Number of observations in the backtest window.
        n_breaches: Number of days where realized loss exceeded VaR.
        breach_rate: Observed breach rate (n_breaches / n_obs).
        target_rate: Target breach rate implied by the VaR confidence level.
        kupiec_lr_stat: Kupiec POF likelihood-ratio statistic.
        kupiec_p_value: p-value of the Kupiec test (chi-squared, 1 df).
        christoffersen_lr_stat: Christoffersen independence LR statistic.
        christoffersen_p_value: p-value of the Christoffersen test.
        traffic_light: Basel traffic-light zone implied by breach count.
    """

    n_obs: int
    n_breaches: int
    breach_rate: float
    target_rate: float
    kupiec_lr_stat: float
    kupiec_p_value: float
    christoffersen_lr_stat: float
    christoffersen_p_value: float
    traffic_light: str


def kupiec_pof_test(breaches: np.ndarray, target_rate: float) -> tuple[float, float]:
    """Kupiec (1995) unconditional-coverage likelihood-ratio test.

    Under the null that the true breach probability equals ``target_rate``,
    the likelihood-ratio statistic
    ``LR = -2 log[ (1-p)^(n-x) p^x / (1-x/n)^(n-x) (x/n)^x ]``
    is asymptotically chi-squared distributed with one degree of freedom,
    where ``n`` is the number of observations, ``x`` the observed breach
    count, and ``p`` the target rate.

    Args:
        breaches: Boolean/0-1 array, ``True`` where realized loss exceeded VaR.
        target_rate: Target breach probability (e.g. 0.01 for 99% VaR).

    Returns:
        A tuple ``(lr_statistic, p_value)``.
    """
    n = len(breaches)
    x = int(np.sum(breaches))
    p = target_rate
    p_hat = x / n if n > 0 else 0.0
    p_hat_clipped = np.clip(p_hat, 1e-8, 1 - 1e-8)
    p_clipped = np.clip(p, 1e-8, 1 - 1e-8)

    log_l_null = (n - x) * np.log(1 - p_clipped) + x * np.log(p_clipped)
    log_l_alt = (n - x) * np.log(1 - p_hat_clipped) + x * np.log(p_hat_clipped)
    lr_stat = -2.0 * (log_l_null - log_l_alt)
    p_value = float(1 - chi2.cdf(lr_stat, df=1))
    return float(lr_stat), p_value


def christoffersen_independence_test(breaches: np.ndarray) -> tuple[float, float]:
    """Christoffersen (1998) conditional-coverage independence test.

    Tests whether breaches are serially independent (i.e., a breach
    today does not raise the probability of a breach tomorrow) by
    comparing a two-state Markov chain likelihood against a constant
    -probability null, via a chi-squared(1) likelihood-ratio statistic.
    A model whose breaches cluster (fails independence) is understating
    risk persistence — exactly the failure mode a "correlations go to
    one" stress regime produces if the model does not adapt quickly.

    Args:
        breaches: Boolean/0-1 array of VaR breach indicators, in time order.

    Returns:
        A tuple ``(lr_statistic, p_value)``.
    """
    b = breaches.astype(int)
    n00 = np.sum((b[:-1] == 0) & (b[1:] == 0))
    n01 = np.sum((b[:-1] == 0) & (b[1:] == 1))
    n10 = np.sum((b[:-1] == 1) & (b[1:] == 0))
    n11 = np.sum((b[:-1] == 1) & (b[1:] == 1))

    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)

    def _safe_log(v: float) -> float:
        return float(np.log(np.clip(v, 1e-10, 1 - 1e-10)))

    log_l_null = (n00 + n10) * _safe_log(1 - pi) + (n01 + n11) * _safe_log(pi)
    log_l_alt = (
        n00 * _safe_log(1 - pi01) + n01 * _safe_log(pi01)
        + n10 * _safe_log(1 - pi11) + n11 * _safe_log(pi11)
    )
    lr_stat = -2.0 * (log_l_null - log_l_alt)
    p_value = float(1 - chi2.cdf(lr_stat, df=1))
    return float(lr_stat), p_value


def basel_traffic_light(n_breaches: int, n_obs: int = 250) -> str:
    """Classifies a backtest into the Basel II/III VaR traffic-light zone.

    Bucket boundaries follow the Basel Committee's 99%/250-day
    multiplier table: green (0-4 breaches), amber (5-9), red (10+).

    Args:
        n_breaches: Observed breach count over the window.
        n_obs: Backtest window length in trading days (Basel standard: 250).

    Returns:
        One of ``"GREEN"``, ``"AMBER"``, ``"RED"``.
    """
    scaled = n_breaches * (250 / max(n_obs, 1))
    if scaled <= 4:
        return "GREEN"
    if scaled <= 9:
        return "AMBER"
    return "RED"


def run_backtest(breaches: np.ndarray, target_rate: float = 0.01) -> BacktestResult:
    """Runs the full Kupiec + Christoffersen + traffic-light backtest suite.

    Args:
        breaches: Boolean/0-1 array of daily VaR breach indicators.
        target_rate: Target breach probability implied by the VaR
            confidence level (0.01 for 99%).

    Returns:
        A populated :class:`BacktestResult`.
    """
    kupiec_lr, kupiec_p = kupiec_pof_test(breaches, target_rate)
    christoffersen_lr, christoffersen_p = christoffersen_independence_test(breaches)
    return BacktestResult(
        n_obs=len(breaches),
        n_breaches=int(np.sum(breaches)),
        breach_rate=float(np.mean(breaches)),
        target_rate=target_rate,
        kupiec_lr_stat=kupiec_lr,
        kupiec_p_value=kupiec_p,
        christoffersen_lr_stat=christoffersen_lr,
        christoffersen_p_value=christoffersen_p,
        traffic_light=basel_traffic_light(int(np.sum(breaches)), len(breaches)),
    )


def fit_haircut_model(collateral: pd.DataFrame) -> tuple[LassoCV, list[str]]:
    """Fits the auditable, log-linear LASSO haircut model.

    Args:
        collateral: Frame with columns
            ``[vol, adv, corr_mkt, rating, haircut]`` (one row per asset).

    Returns:
        A tuple ``(fitted_model, feature_names)``.
    """
    feature_cols = ["log_vol", "log_adv", "corr_mkt", "rating"]
    x = pd.DataFrame({
        "log_vol": np.log(collateral["vol"]),
        "log_adv": np.log(collateral["adv"]),
        "corr_mkt": collateral["corr_mkt"],
        "rating": collateral["rating"],
    })
    y = np.log(collateral["haircut"])
    model = LassoCV(cv=5, n_alphas=100, max_iter=10_000).fit(x, y)
    return model, feature_cols


def fit_correlation_addon(
    portfolio_features: pd.DataFrame, shortfall_residual: np.ndarray
) -> GradientBoostingRegressor:
    """Fits the tree-based portfolio correlation-breakdown add-on.

    Args:
        portfolio_features: Portfolio-level concentration/correlation
            features, one row per portfolio-day.
        shortfall_residual: Historical-simulation shortfall not explained
            by the linear per-asset haircut sum (target variable).

    Returns:
        A fitted :class:`~sklearn.ensemble.GradientBoostingRegressor`
        using a Huber loss for robustness to fat-tailed shortfall residuals.
    """
    model = GradientBoostingRegressor(
        n_estimators=400, max_depth=3, learning_rate=0.02, loss="huber", alpha=0.9,
    )
    model.fit(portfolio_features, shortfall_residual)
    return model


def required_initial_margin(
    haircut_model: LassoCV,
    feature_cols: list[str],
    collateral_row: pd.Series,
    quantity: float,
    price: float,
    rules_floor: float,
    var99_cap: float,
    correlation_addon: float = 0.0,
) -> tuple[float, float]:
    """Computes the governed, floor/cap-clipped required Initial Margin.

    The raw model output is never posted directly to a client; it is
    always clipped to ``[rules_floor, 1.2 * var99_cap]`` so that the ML
    layer can only *tighten* risk management within a pre-approved band,
    never silently drift outside it (a standard model-risk control).

    Args:
        haircut_model: Fitted LASSO haircut model.
        feature_cols: Feature column order expected by ``haircut_model``.
        collateral_row: Row with ``[vol, adv, corr_mkt, rating]`` for one asset.
        quantity: Position quantity (absolute value used).
        price: Asset price.
        rules_floor: Minimum allowed IM per the existing rules-based grid.
        var99_cap: 99% five-day historical-simulation VaR estimate.
        correlation_addon: Portfolio-level correlation-breakdown add-on ($).

    Returns:
        A tuple ``(required_im, implied_haircut_pct)``.
    """
    x_row = pd.DataFrame([{
        "log_vol": np.log(collateral_row["vol"]),
        "log_adv": np.log(collateral_row["adv"]),
        "corr_mkt": collateral_row["corr_mkt"],
        "rating": collateral_row["rating"],
    }])[feature_cols]
    haircut = float(np.exp(haircut_model.predict(x_row)[0]))
    linear_im = haircut * abs(quantity) * price
    im = linear_im + correlation_addon
    im_clipped = float(np.clip(im, rules_floor, 1.2 * var99_cap))
    return im_clipped, haircut


def _main() -> None:
    """Command-line entry point: fits the haircut model and runs a demo backtest."""
    collateral = pd.read_csv("data/collateral_universe.csv").rename(
        columns={"volatility": "vol", "historical_haircut": "haircut"}
    )
    model, feature_cols = fit_haircut_model(collateral)
    print("LASSO coefficients:", dict(zip(feature_cols, model.coef_.round(4))))

    rng = np.random.default_rng(7)
    breaches = rng.random(250) < 0.012
    result = run_backtest(breaches, target_rate=0.01)
    print(result)


if __name__ == "__main__":
    _main()
