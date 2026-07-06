"""Regime-aware financing spread forecaster for Liquid Financing.

This module implements a hybrid classical-deep learning forecasting engine.
It uses an ARIMA-GARCH baseline to capture linear auto-regression and
volatility clustering, combined with a LayerNorm-stabilized GRU network
to correct non-linear residuals. The neural network is gated by a robust
Mahalanobis distance check (Minimum Covariance Determinant) to prevent
model hallucination during unprecedented market regimes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import scipy.linalg as la
import torch
from arch import arch_model
from sklearn.covariance import MinCovDet
from statsmodels.tsa.arima.model import ARIMA
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure module-level logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass(slots=True, kw_only=True)
class ForecastResult:
    """Memory-efficient container for forecast outputs.

    Attributes:
        point_forecast_bps: The absolute forecasted spread in basis points.
        lower_bps: Lower bound of the confidence interval.
        upper_bps: Upper bound of the confidence interval.
        regime_flag: Indicator of whether the input is within the training distribution.
        ood_distance: The computed Mahalanobis distance.
    """
    point_forecast_bps: float
    lower_bps: float
    upper_bps: float
    regime_flag: Literal["IN_DISTRIBUTION", "OUT_OF_DISTRIBUTION"]
    ood_distance: float


class ResidualCorrector(nn.Module):
    """Robust GRU-based residual predictor for financial time series.

    Attributes:
        rnn: The Gated Recurrent Unit layer.
        ln: Layer normalization to stabilize gradients against noisy inputs.
        dropout: Dropout layer for regularization.
        regressor: Final linear projection to the residual prediction.
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 32,
        dropout: float = 0.2
    ) -> None:
        """Initializes the deep learning corrector."""
        super().__init__()
        self.rnn = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.ln = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Executes the forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_features).

        Returns:
            A 1D tensor of shape (batch_size,) containing residual predictions.
        """
        rnn_out, _ = self.rnn(x)
        last_out = rnn_out[:, -1, :]
        norm_out = self.ln(last_out)
        drop_out = self.dropout(norm_out)
        return self.regressor(drop_out).squeeze(-1)


class RegimeGatedForecaster:
    """Production-grade hybrid classical/DL forecaster.

    Attributes:
        arima_order: The (p, d, q) parameters for the ARIMA model.
        ood_threshold: Threshold for the Mahalanobis distance OOD gating.
        seq_len: Number of historical steps passed to the GRU per prediction.
        device: The compute device for PyTorch execution ('cpu' or 'cuda').
    """

    def __init__(
        self,
        arima_order: tuple[int, int, int] = (2, 1, 2),
        ood_threshold: float = 3.5,
        seq_len: int = 5,
        device: str = "cpu",
    ) -> None:
        """Initializes the forecaster with structural hyperparameters."""
        self.arima_order = arima_order
        self.ood_threshold = ood_threshold
        self.seq_len = seq_len
        self.device = torch.device(device)

        self._arima_res = None
        self._garch_res = None
        self._garch_scale: float = 1.0
        self._dl_corrector: nn.Module | None = None

        self._train_mean: np.ndarray | None = None
        self._cholesky_l: np.ndarray | None = None

    def fit(
        self,
        series: np.ndarray,
        features: np.ndarray,
        epochs: int = 50,
        batch_size: int = 64
    ) -> None:
        """Fits classical models, robust OOD parameters, and the DL corrector.

        Args:
            series: 1D array of the target financing spread, in bps.
            features: 2D array (n_obs, n_features) of exogenous drivers.
            epochs: Number of training epochs for the neural network.
            batch_size: Batch size for neural network training.

        Raises:
            ValueError: If input arrays have incorrect dimensions.
            RuntimeError: If covariance matrix computation fails.
        """
        if series.ndim != 1 or features.ndim != 2:
            raise ValueError("Invalid shapes: series must be 1D, features 2D.")
        if len(series) != len(features):
            raise ValueError("Series and features must have the same length.")

        logger.info("Fitting classical ARIMA baseline...")
        # Suppress convergence warnings common in automated ARIMA fitting
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._arima_res = ARIMA(series, order=self.arima_order).fit()

        logger.info("Fitting GARCH volatility model...")
        resid = self._arima_res.resid
        self._garch_scale = float(np.std(resid)) if np.std(resid) > 0 else 1.0
        scaled_resid = resid / self._garch_scale

        garch = arch_model(scaled_resid, vol="GARCH", p=1, q=1)
        self._garch_res = garch.fit(disp="off")

        logger.info("Computing robust covariance (MCD) for OOD detection...")
        mcd = MinCovDet(random_state=42).fit(features)
        self._train_mean = mcd.location_

        cov = mcd.covariance_ + 1e-6 * np.eye(features.shape[1])
        try:
            self._cholesky_l = la.cholesky(cov, lower=True)
        except la.LinAlgError as e:
            logger.error("Covariance matrix is not positive definite.")
            raise RuntimeError("Cholesky decomposition failed.") from e

        logger.info("Initializing and training DL corrector...")
        model = ResidualCorrector(n_features=features.shape[1]).to(self.device)
        self._train_dl(model, features, resid, epochs, batch_size)
        self._dl_corrector = model.eval()

    def _create_sequences(
        self, features: np.ndarray, residuals: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generates overlapping sliding windows for RNN training.

        Args:
            features: 2D array of features.
            residuals: 1D array of ARIMA residuals.

        Returns:
            A tuple of (X_tensor, y_tensor) for PyTorch ingestion.
        """
        x_seq, y_seq = [], []
        for i in range(len(features) - self.seq_len):
            x_seq.append(features[i : i + self.seq_len])
            y_seq.append(residuals[i + self.seq_len])
        return (
            torch.tensor(np.array(x_seq), dtype=torch.float32),
            torch.tensor(np.array(y_seq), dtype=torch.float32)
        )

    def _train_dl(
        self,
        model: nn.Module,
        features: np.ndarray,
        residuals: np.ndarray,
        epochs: int,
        batch_size: int
    ) -> None:
        """Executes the PyTorch training loop with AdamW and learning rate scheduling."""
        x_data, y_data = self._create_sequences(features, residuals)
        dataset = TensorDataset(x_data, y_data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                predictions = model(batch_x)
                loss = criterion(predictions, batch_y)
                loss.backward()
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()

            scheduler.step(epoch_loss)

        logger.info(f"DL Corrector training complete. Final loss: {epoch_loss:.4f}")

    def predict(
        self, latest_features: np.ndarray, horizon: int = 1
    ) -> ForecastResult:
        """Produces a point forecast and dynamic confidence band.

        Args:
            latest_features: 2D array of shape (seq_len, n_features) representing
                the most recent feature window.
            horizon: Forecast horizon in periods.

        Returns:
            A ForecastResult dataclass instance.

        Raises:
            RuntimeError: If called before the model is fitted.
            ValueError: If the feature window does not match seq_len.
        """
        if self._arima_res is None or self._cholesky_l is None:
            raise RuntimeError("Model must be fitted before calling predict.")

        if latest_features.ndim != 2 or latest_features.shape[0] != self.seq_len:
            raise ValueError(
                f"latest_features must be 2D with shape ({self.seq_len}, n_features)."
            )

        # 1. Classical Forecast
        # FIX: statsmodels returns a raw numpy array when trained on numpy arrays.
        # We use standard standard index [-1] instead of Pandas .iloc[-1]
        base_forecast = self._arima_res.forecast(steps=horizon)[-1]

        scaled_var = self._garch_res.forecast(horizon=horizon).variance.values[-1, -1]
        vol_forecast = np.sqrt(scaled_var) * self._garch_scale

        # 2. Robust OOD Detection (using only the most recent feature vector t=0)
        current_step_features = latest_features[-1]
        diff = current_step_features - self._train_mean
        y = la.solve_triangular(self._cholesky_l, diff, lower=True)
        ood_distance = float(np.sqrt(np.dot(y, y)))

        is_ood = ood_distance > self.ood_threshold
        regime_flag = "OUT_OF_DISTRIBUTION" if is_ood else "IN_DISTRIBUTION"

        # 3. Regime-Gated DL Correction
        correction = 0.0
        if not is_ood and self._dl_corrector is not None:
            with torch.no_grad():
                x_tensor = torch.tensor(
                    latest_features, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                correction = self._dl_corrector(x_tensor).item()

        # 4. Final Ensemble & Dynamic Banding
        point_forecast = base_forecast + correction
        z_score = 2.58 if is_ood else 1.96
        band_width = vol_forecast * z_score

        return ForecastResult(
            point_forecast_bps=float(point_forecast),
            lower_bps=float(point_forecast - band_width),
            upper_bps=float(point_forecast + band_width),
            regime_flag=regime_flag,
            ood_distance=ood_distance
        )


def generate_plot(
    dates: list[pd.Timestamp],
    actuals: list[float],
    predictions: list[float],
    lowers: list[float],
    uppers: list[float],
    flags: list[str]
) -> None:
    """Generates and saves interactive and static Plotly visualizations.

    Args:
        dates: Timestamps for the x-axis.
        actuals: Actual financing spread values.
        predictions: Model point forecasts.
        lowers: Confidence interval lower bounds.
        uppers: Confidence interval upper bounds.
        flags: Regime flags ('IN_DISTRIBUTION' or 'OUT_OF_DISTRIBUTION').
    """
    # FIX: Sanitize pandas Timestamps to ISO strings for orjson/Kaleido compatibility
    safe_dates = [d.strftime("%Y-%m-%d") for d in dates]
    
    fig = make_subplots(rows=1, cols=1, subplot_titles=["Regime-Aware Spread Forecast"])

    # Confidence Interval Band
    fig.add_trace(go.Scatter(
        x=safe_dates + safe_dates[::-1],
        y=uppers + lowers[::-1],
        fill='toself',
        fillcolor='rgba(0,176,246,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=True,
        name='Dynamic Confidence Interval'
    ))

    # Actuals
    fig.add_trace(go.Scatter(
        x=safe_dates, y=actuals,
        mode='lines', line=dict(color='gray', width=2),
        name='Actual Spread (bps)'
    ))

    # Predictions
    fig.add_trace(go.Scatter(
        x=safe_dates, y=predictions,
        mode='lines', line=dict(color='blue', width=2),
        name='Hybrid Forecast (bps)'
    ))

    # Highlight OOD Regimes
    ood_indices = [i for i, flag in enumerate(flags) if flag == "OUT_OF_DISTRIBUTION"]
    for idx in ood_indices:
        fig.add_vrect(
            x0=safe_dates[idx], 
            x1=safe_dates[min(idx + 1, len(safe_dates) - 1)],
            fillcolor="red", opacity=0.2, layer="below", line_width=0,
        )

    # Dummy trace for OOD legend entry
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(size=10, color='rgba(255,0,0,0.2)', symbol='square'),
        name='OOD Regime (DL Gated)'
    ))

    fig.update_layout(
        title="Institutional Cross-Asset Spread Forecasting Engine",
        xaxis_title="Date",
        yaxis_title="Spread (bps)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    logger.info("Saving plots to disk...")
    try:
        fig.write_image("forecast_regimes.png", width=1200, height=600, scale=2)
    except ValueError as e:
        logger.warning(f"Failed to save PNG (is kaleido installed?). Error: {e}")
        
    fig.write_html("forecast_regimes.html")
    logger.info("Plots saved successfully as forecast_regimes.png and forecast_regimes.html")


if __name__ == "__main__":
    # 1. Generate Synthetic Multi-Asset Data Matrix
    np.random.seed(42)
    n_obs = 1000
    n_features = 3
    seq_len = 5

    # Base features: Random walk proxies for GC rate, VIX, Collateral Scarcity
    features_data = np.cumsum(np.random.normal(0, 0.1, size=(n_obs, n_features)), axis=0)

    # Inject a massive out-of-distribution regime shock in the last 50 days
    features_data[-50:, :] += np.random.normal(5.0, 2.0, size=(50, n_features))

    # Target series: Auto-regressive baseline + feature exposure + GARCH noise
    noise = np.random.normal(0, 0.5, size=n_obs)
    noise[-50:] *= 3.0  # Volatility expansion during regime shock
    
    series_data = np.zeros(n_obs)
    for t in range(1, n_obs):
        series_data[t] = 0.8 * series_data[t-1] + np.dot(features_data[t], [0.5, -0.3, 0.2]) + noise[t]

    # Split into train/test
    train_split = 900
    train_series = series_data[:train_split]
    train_features = features_data[:train_split]

    # 2. Initialize and Train Forecaster
    forecaster = RegimeGatedForecaster(
        arima_order=(2, 0, 1),
        ood_threshold=4.0, # Strict tuning
        seq_len=seq_len,
        device="cpu"
    )

    forecaster.fit(train_series, train_features, epochs=30, batch_size=32)

    # 3. Out-of-Sample Walk-Forward Prediction
    test_dates = pd.date_range(start="2026-06-01", periods=n_obs - train_split, freq="B")
    
    actuals, predictions, lowers, uppers, flags = [], [], [], [], []

    logger.info("Starting out-of-sample walk-forward predictions...")
    for i in range(train_split, n_obs):
        # Slice the sequence window required by the GRU
        window_start = i - seq_len
        latest_window = features_data[window_start:i]

        res = forecaster.predict(latest_features=latest_window, horizon=1)
        
        actuals.append(series_data[i])
        predictions.append(res.point_forecast_bps)
        lowers.append(res.lower_bps)
        uppers.append(res.upper_bps)
        flags.append(res.regime_flag)

    # 4. Render and Export Artifacts
    generate_plot(
        dates=list(test_dates),
        actuals=actuals,
        predictions=predictions,
        lowers=lowers,
        uppers=uppers,
        flags=flags
    )