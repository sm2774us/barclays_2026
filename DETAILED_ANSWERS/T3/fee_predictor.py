"""Institutional financing fee prediction and explainability engine.

This module implements a production-grade LightGBM regressor with complete
TreeSHAP local and global explainability, outputting fully audited results
and custom Plotly visualizations to disk.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap

# Configure logger to adhere to standard governance tracking
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass(slots=True, kw_only=True)
class AuditMetrics:
    """Memory-efficient container for global model attributions."""
    mean_abs_shap: np.ndarray
    feature_names: list[str]
    sample_prediction: float
    sample_base_value: float
    sample_local_shap: np.ndarray


class AuditedFeePredictor:
    """LightGBM predictor integrated with exact TreeSHAP auditing capabilities."""

    def __init__(self, max_depth: int = 5, n_estimators: int = 100) -> None:
        """Initializes the underlying model hyper-parameters."""
        self.model_params = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.05,
            "max_depth": max_depth,
            "num_leaves": 2 ** max_depth - 1,
            "n_estimators": n_estimators,
            "random_state": 42,
            "verbose": -1,
            "n_jobs": -1  # Maximize core utilization
        }
        self.model: lgb.LGBMRegressor | None = None
        self.explainer: shap.TreeExplainer | None = None

    def train_and_audit(self, X: pd.DataFrame, y: pd.Series) -> AuditMetrics:
        """Trains the boosting tree ensemble and extracts SHAP attributions.

        Args:
            X: Input DataFrame with numeric or category-typed columns.
            y: Target Series (e.g., borrow or financing fee in bps).

        Returns:
            An AuditMetrics dataclass mapping global and sample local weights.
        """
        logger.info("Fitting function-space gradient boosted ensemble...")
        self.model = lgb.LGBMRegressor(**self.model_params)
        self.model.fit(X, y)

        logger.info("Initializing exact polynomial TreeSHAP explainer...")
        # TreeSHAP calculates exact values in polynomial time O(TLD^2)
        self.explainer = shap.TreeExplainer(self.model)
        
        logger.info("Calculating SHAP values across the asset universe...")
        shap_values_obj = self.explainer(X)
        
        # Extract matrices depending on return format variations
        shap_matrix = shap_values_obj.values
        base_value = float(shap_values_obj.base_values[0])

        mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)
        feature_names = list(X.columns)

        # Extract local metrics for a sample profile (e.g., the final row index)
        sample_local_shap = shap_matrix[-1, :]
        sample_prediction = float(self.model.predict(X.iloc[[-1]])[0])

        logger.info("Model risk audit generation complete.")
        return AuditMetrics(
            mean_abs_shap=mean_abs_shap,
            feature_names=feature_names,
            sample_prediction=sample_prediction,
            sample_base_value=base_value,
            sample_local_shap=sample_local_shap
        )


def generate_plots(metrics: AuditMetrics, sample_row: pd.Series) -> None:
    """Generates dual-panel Plotly visualizations for Model Validation sign-off.

    Args:
        metrics: Populated AuditMetrics configuration dataclass.
        sample_row: The series representing feature values of the audited row.
    """
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Global Feature Importance (Mean Absolute SHAP)", "Local Prediction Waterfall Decomposition"),
        vertical_spacing=0.18,
        row_heights=[0.4, 0.6]
    )

    # Panel 1: Global Importance
    sorted_idx = np.argsort(metrics.mean_abs_shap)
    sorted_features = [metrics.feature_names[i] for i in sorted_idx]
    sorted_global_values = metrics.mean_abs_shap[sorted_idx]

    fig.add_trace(
        go.Bar(
            x=sorted_global_values,
            y=sorted_features,
            orientation='h',
            marker=dict(color='rgba(44, 160, 44, 0.75)', line=dict(color='green', width=1)),
            name='Global Mean Attribution'
        ),
        row=1, col=1
    )

    # Panel 2: Local Waterfall Plot for Model Audit
    # Sort local attributions by absolute impact size
    local_idx = np.argsort(np.abs(metrics.sample_local_shap))[::-1]
    
    cumulative_val = metrics.sample_base_value
    y_labels = ["Base Expected Value"]
    x_steps = [metrics.sample_base_value]
    measure_types = ["absolute"]

    for idx in local_idx:
        val = metrics.sample_local_shap[idx]
        feat_name = metrics.feature_names[idx]
        actual_val = sample_row.iloc[idx]
        
        y_labels.append(f"{feat_name} ({actual_val:.2f})")
        x_steps.append(val)
        measure_types.append("relative")
        cumulative_val += val

    y_labels.append("Final Model Output")
    x_steps.append(metrics.sample_prediction)
    measure_types.append("total")

    fig.add_trace(
        go.Waterfall(
            orientation="h",
            measure=measure_types,
            y=y_labels,
            x=x_steps,
            connector=dict(line=dict(color="rgb(63, 63, 63)", width=1, dash="dot")),
            decreasing=dict(marker=dict(color="rgba(219, 68, 85, 0.8)")),
            increasing=dict(marker=dict(color="rgba(44, 160, 44, 0.8)")),
            totals=dict(marker=dict(color="rgba(31, 119, 180, 0.9)")),
            name="Local Attribution Trace"
        ),
        row=2, col=1
    )

    fig.update_layout(
        title="Institutional Financing Fee Explainability Dashboard (SHAP Audit Core)",
        template="plotly_white",
        height=900,
        width=1200,
        showlegend=False
    )
    
    fig.update_xaxes(title_text="Mean |SHAP Value| (bps)", row=1, col=1)
    fig.update_xaxes(title_text="Financing Fee Cumulative Value (bps)", row=2, col=1)

    logger.info("Writing explainability artifacts to disk...")
    fig.write_html("fee_explainability.html")
    try:
        fig.write_image("fee_explainability.png", width=1200, height=900, scale=2)
        logger.info("Artifacts successfully saved to disk.")
    except ValueError as e:
        logger.warning(f"Static image generation failed. Is kaleido installed? Error: {e}")


if __name__ == "__main__":
    # Simulate an institutional tabular dataset for cross-asset borrow fees
    np.random.seed(42)
    n_records = 5000

    # Continuous and indicator feature definitions
    collateral_scarcity = np.random.exponential(scale=1.5, size=n_records)
    counterparty_tier = np.random.choice([1, 2, 3], size=n_records, p=[0.5, 0.3, 0.2])
    notional_size_m = np.random.uniform(1.0, 150.0, size=n_records)
    market_volatility = np.random.normal(loc=18.0, scale=4.0, size=n_records)

    features_df = pd.DataFrame({
        "Collateral_Scarcity_Score": collateral_scarcity,
        "Counterparty_Tier_Code": counterparty_tier.astype(float),
        "Transaction_Notional_M": notional_size_m,
        "Market_Volatility_Index": market_volatility
    })

    # Generate non-linear target fee function (trees excel at identifying these step-boundaries)
    base_fee = 10.0 + 8.5 * (features_df["Collateral_Scarcity_Score"] ** 1.5)
    base_fee += 15.0 * (features_df["Counterparty_Tier_Code"] == 3)
    base_fee -= 2.0 * (features_df["Transaction_Notional_M"] > 100.0)
    target_fees = base_fee + np.random.normal(0, 1.5, size=n_records)

    # Initialize and execute pipeline
    auditor = AuditedFeePredictor()
    audit_metrics = auditor.train_and_audit(features_df, target_fees)

    # Extract the individual row sample targeted for local decomposition
    audited_row_data = features_df.iloc[-1]

    # Generate interactive and static diagnostics
    generate_plots(audit_metrics, audited_row_data)