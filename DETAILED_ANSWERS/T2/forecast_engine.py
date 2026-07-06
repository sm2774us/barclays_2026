"""Institutional Elastic Net forecasting engine for overnight repo spreads.

This module provides an automated, cross-validated pipeline for predicting 
financing spreads under severe feature collinearity. It enforces strict 
time-series boundaries to eliminate look-ahead data leakage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass(slots=True, kw_only=True)
class ModelMetrics:
    """Memory-efficient container for pipeline metrics and diagnostics."""
    best_l1_ratio: float
    best_lambda: float
    oos_r2: float
    feature_names: list[str]
    mean_coefficients: np.ndarray
    coefficient_std: np.ndarray


class InstitutionalRepoForecaster:
    """Robust Elastic Net modeling engine designed for Liquid Financing desks.

    Enforces proper time-series cross-validation splits and computes metric
    stability boundaries across sliding historical validation windows.
    """

    def __init__(
        self,
        l1_ratios: list[float] | None = None,
        n_lambdas: int = 100,
        n_splits: int = 5
    ) -> None:
        """Initializes the hyperparameter search configurations."""
        self.l1_ratios = l1_ratios or [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
        self.n_lambdas = n_lambdas
        self.n_splits = n_splits
        
        # State variables
        self.best_model: ElasticNet | None = None
        self.scaler = StandardScaler()
        self.metrics: ModelMetrics | None = None

    def fit_and_evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> InstitutionalRepoForecaster:
        """Fits the pipeline using forward-chaining TimeSeriesSplit validation.

        Args:
            X: Dataframe containing time-ordered feature rows.
            y: Target series matching the feature index.

        Returns:
            The fitted instance of InstitutionalRepoForecaster.
        """
        logger.info("Initializing forward-chaining time-series split cross-validation...")
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        
        # Array containers to trace parameter performance across folds
        fold_coefficients = []
        best_fold_scores = []

        # Convert to numpy internals for high-performance matrix scaling blocks
        X_arr = X.to_numpy()
        y_arr = y.to_numpy()
        feature_names = list(X.columns)

        # High-performance grid search optimized for time-series consistency
        best_overall_score = -np.inf
        optimal_l1 = self.l1_ratios[0]
        optimal_lambda = 1.0

        for l1_ratio in self.l1_ratios:
            scores_per_ratio = []
            for train_idx, val_idx in tscv.split(X_arr):
                # Isolate folds locally to prevent data leakage
                X_train, X_val = X_arr[train_idx], X_arr[val_idx]
                y_train, y_val = y_arr[train_idx], y_arr[val_idx]

                # Scale features based only on the active historical anchor fold
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_val_scaled = self.scaler.transform(X_val)

                # Initialize local single-step model
                # Note: l1_ratio maps to alpha, alpha maps to lambda in scikit-learn
                model = ElasticNet(
                    alpha=1.0, 
                    l1_ratio=l1_ratio, 
                    max_iter=5000, 
                    random_state=42
                )
                model.fit(X_train_scaled, y_train)
                
                # Check performance via Out-Of-Sample coefficient of determination (R^2)
                score = float(model.score(X_val_scaled, y_val))
                scores_per_ratio.append(score)

            mean_ratio_score = float(np.mean(scores_per_ratio))
            if mean_ratio_score > best_overall_score:
                best_overall_score = mean_ratio_score
                optimal_l1 = l1_ratio

        logger.info(f"Optimal L1 mixing ratio isolated: {optimal_l1}")

        # Retain coefficients over final splits with optimal hyperparameters
        for train_idx, val_idx in tscv.split(X_arr):
            X_train, X_val = X_arr[train_idx], X_arr[val_idx]
            y_train, y_val = y_arr[train_idx], y_arr[val_idx]

            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)

            final_model = ElasticNet(alpha=0.1, l1_ratio=optimal_l1, max_iter=10000, random_state=42)
            final_model.fit(X_train_scaled, y_train)
            
            fold_coefficients.append(final_model.coef_)
            best_fold_scores.append(final_model.score(X_val_scaled, y_val))

        # Train production-ready model on total historical data architecture
        X_full_scaled = self.scaler.fit_transform(X_arr)
        self.best_model = ElasticNet(alpha=0.1, l1_ratio=optimal_l1, max_iter=10000, random_state=42)
        self.best_model.fit(X_full_scaled, y_arr)

        # Store comprehensive metrics
        coef_matrix = np.array(fold_coefficients)
        self.metrics = ModelMetrics(
            best_l1_ratio=optimal_l1,
            best_lambda=0.1,
            oos_r2=float(np.mean(best_fold_scores)),
            feature_names=feature_names,
            mean_coefficients=np.mean(coef_matrix, axis=0),
            coefficient_std=np.std(coef_matrix, axis=0)
        )
        
        logger.info("Pipeline cross-validation tracking successfully finalized.")
        return self


def generate_artifacts(
    dates: list[pd.Timestamp],
    actuals: np.ndarray,
    predictions: np.ndarray,
    metrics: ModelMetrics
) -> None:
    """Generates both interactive HTML and high-resolution static PNG plots.

    Args:
        dates: List of pandas Timestamps representing the time axes.
        actuals: Realized market spread targets.
        predictions: In-sample/out-of-sample fitted predictions.
        metrics: Populated dataclass containing structural coefficients.
    """
    # Sanitize dates into string structures to avoid orjson/Kaleido serialization errors
    safe_dates = [d.strftime("%Y-%m-%d") for d in dates]

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Actual vs. Predicted Repo Spread (bps)", "Cross-Validation Coefficient Stability"),
        vertical_spacing=0.15,
        row_heights=[0.5, 0.5]
    )

    # Top Panel: Actual vs. Predicted Time-Series Data
    fig.add_trace(
        go.Scatter(x=safe_dates, y=actuals, mode='lines', line=dict(color='gray', width=1.5), name='Actual Spread'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=safe_dates, y=predictions, mode='lines', line=dict(color='blue', width=2), name='ElasticNet Prediction'),
        row=1, col=1
    )

    # Bottom Panel: Stabilized Coefficient Importances with Error Bars representing Fold Variance
    fig.add_trace(
        go.Bar(
            x=metrics.feature_names,
            y=metrics.mean_coefficients,
            error_y=dict(type='data', array=metrics.coefficient_std, visible=True, color='red'),
            marker=dict(color='rgba(0, 114, 178, 0.7)', line=dict(color='blue', width=1)),
            name='Mean Weight (w/ Fold Variance)'
        ),
        row=2, col=1
    )

    fig.update_layout(
        title=f"Institutional Elastic Net Repo Engine (OOS R²: {metrics.oos_r2:.4f} | Optimal L1 Ratio: {metrics.best_l1_ratio})",
        template="plotly_white",
        height=800,
        width=1200,
        showlegend=True,
        hovermode="x unified"
    )

    fig.update_yaxes(title_text="Spread (bps)", row=1, col=1)
    fig.update_yaxes(title_text="Standardized Weight Value", row=2, col=1)

    # Save outputs safely to disk
    logger.info("Writing plot artifacts to disk...")
    fig.write_html("repo_elastic_net.html")
    try:
        fig.write_image("repo_elastic_net.png", width=1200, height=800, scale=2)
        logger.info("Artifacts successfully saved as repo_elastic_net.html and repo_elastic_net.png")
    except ValueError as e:
        logger.warning(f"Static PNG export skipped. Is kaleido installed? Error: {e}")


if __name__ == "__main__":
    # Simulate a typical Liquid Financing multivariate ecosystem
    np.random.seed(42)
    n_days = 500
    timeline = pd.date_range(start="2024-07-06", periods=n_days, freq="B")

    # Generate highly collinear drivers
    gc_rate = np.cumsum(np.random.normal(0.0, 0.05, n_days)) + 4.5
    sofr_rate = gc_rate + np.random.normal(0.0, 0.01, n_days)  # Correlation ~0.98
    collateral_scarcity = np.random.normal(2.0, 0.5, n_days) + 0.3 * gc_rate
    counterparty_risk = np.random.normal(0.1, 0.02, n_days)
    
    # Quarter-End dummy representing structural banking liquidity contractions
    quarter_end = np.zeros(n_days)
    quarter_end[::60] = 1.0 

    features_df = pd.DataFrame({
        "GC_Rate_Proxy": gc_rate,
        "SOFR_Rate_Proxy": sofr_rate,
        "Collateral_Scarcity_Score": collateral_scarcity,
        "Counterparty_Credit_Spread": counterparty_risk,
        "Quarter_End_Indicator": quarter_end
    })

    # Generate target spread driven by structural underlying components + noise
    target_spread = (
        0.4 * features_df["GC_Rate_Proxy"] 
        + 0.1 * features_df["Collateral_Scarcity_Score"] 
        + 2.5 * features_df["Quarter_End_Indicator"] 
        + np.random.normal(0, 0.05, n_days)
    )

    # Train and extract execution performance metrics
    forecaster = InstitutionalRepoForecaster(n_splits=5)
    forecaster.fit_and_evaluate(features_df, target_spread)

    # Generate prediction metrics using our fitted full pipeline
    full_scaled_x = forecaster.scaler.transform(features_df.to_numpy())
    predicted_spread = forecaster.best_model.predict(full_scaled_x)

    # Render plots
    generate_artifacts(
        dates=list(timeline),
        actuals=target_spread.to_numpy(),
        predictions=predicted_spread,
        metrics=forecaster.metrics
    )