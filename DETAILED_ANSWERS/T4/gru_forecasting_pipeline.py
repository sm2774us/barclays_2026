"""Production-grade Gated Recurrent Unit (GRU) time-series forecasting engine.

Implements sequential lookback tensor generation, gradient-clipped BPTT execution, 
and automated interactive diagnostics for liquid financing spreads.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Configure logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass(slots=True, kw_only=True)
class TrainingMetrics:
    """Container for tracking loss dynamics across optimization steps."""

    epoch_losses: list[float]
    final_oos_rmse: float
    actual_series: np.ndarray
    predicted_series: np.ndarray
    timestamps: list[str]


class TimeSeriesSequenceDataset(Dataset):
    """Generates sequential historical feature matrices with zero look-ahead bias."""

    def __init__(self, X: np.ndarray, y: np.ndarray, lookback: int) -> None:
        """Initializes stride structures over multi-dimensional arrays.

        Args:
            X: Standardized feature matrix of shape [TotalSamples, Features].
            y: Target array of shape [TotalSamples].
            lookback: Number of historical lags per sequence step.
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.lookback = lookback

    def __len__(self) -> int:
        return len(self.X) - self.lookback

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns localized temporal matrix pairs."""
        X_seq = self.X[idx : idx + self.lookback]
        y_target = self.y[idx + self.lookback]
        return X_seq, y_target


class InstitutionalGRUNet(nn.Module):
    """Regularized GRU network for processing financial time series."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2) -> nn.Module:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Recurrent layer allocation
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.15 if num_layers > 1 else 0.0,
        )

        # LayerNorm stabilizes sequence activation variance
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(p=0.2)
        self.projection = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Processes tensor shapes of size [Batch, SeqLen, Features]."""
        # Initialize hidden state containers dynamically on active device
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device)

        # Forward pass through recurrent units
        gru_out, _ = self.gru(x, h_0)

        # Isolate the final hidden state from the sequence dimension
        out_terminal = gru_out[:, -1, :]

        # Apply regularization layers
        out_norm = self.dropout(self.layer_norm(out_terminal))
        return self.projection(out_norm).squeeze(-1)


class GRUForecastingPipeline:
    """Manages the network lifecycle, including optimization loops and forecasting."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        num_layers: int = 2,
        lr: float = 0.002,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = InstitutionalGRUNet(input_dim, hidden_dim, num_layers).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.criterion = nn.MSELoss()

    def execute_lifecycle(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        test_dates: list[pd.Timestamp],
        lookback: int = 30,
        epochs: int = 20,
        batch_size: int = 64,
    ) -> TrainingMetrics:
        """Trains the network and generates out-of-sample predictions."""
        train_dataset = TimeSeriesSequenceDataset(X_train, y_train, lookback)
        test_dataset = TimeSeriesSequenceDataset(X_test, y_test, lookback)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        epoch_losses = []
        logger.info("Initiating gradient-clipped recurrent training cycle...")

        for epoch in range(epochs):
            self.model.train()
            batch_losses = []

            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                self.optimizer.zero_grad()
                predictions = self.model(X_batch)
                loss = self.criterion(predictions, y_batch)
                loss.backward()

                # Clip gradients to prevent exploding gradients
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                batch_losses.append(loss.item())

            mean_epoch_loss = float(np.mean(batch_losses))
            epoch_losses.append(mean_epoch_loss)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(f"Epoch {epoch+1:02d}/{epochs:02d} | Train MSE Loss: {mean_epoch_loss:.6f}")

        # Out-of-sample evaluation pass
        self.model.eval()
        oos_predictions = []
        oos_actuals = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                preds = self.model(X_batch)
                oos_predictions.extend(preds.cpu().numpy())
                oos_actuals.extend(y_batch.numpy())

        oos_predictions_arr = np.array(oos_predictions)
        oos_actuals_arr = np.array(oos_actuals)
        final_rmse = float(np.sqrt(np.mean((oos_predictions_arr - oos_actuals_arr) ** 2)))

        # Align timestamps to match length of sequence outputs
        safe_dates = [d.strftime("%Y-%m-%d") for d in test_dates[lookback:]]

        return TrainingMetrics(
            epoch_losses=epoch_losses,
            final_oos_rmse=final_rmse,
            actual_series=oos_actuals_arr,
            predicted_series=oos_predictions_arr,
            timestamps=safe_dates,
        )


def generate_visual_artifacts(metrics: TrainingMetrics) -> None:
    """Generates HTML and high-resolution PNG diagnostic charts."""
    fig = go.Figure()

    # Out-of-Sample Actual Path
    fig.add_trace(
        go.Scatter(
            x=metrics.timestamps,
            y=metrics.actual_series,
            mode="lines",
            line=dict(color="rgba(100, 100, 100, 0.7)", width=1.5),
            name="Realized Repo Spread",
        )
    )

    # GRU Forecast Path
    fig.add_trace(
        go.Scatter(
            x=metrics.timestamps,
            y=metrics.predicted_series,
            mode="lines",
            line=dict(color="rgba(219, 68, 85, 0.9)", width=2),
            name="GRU Recurrent Prediction",
        )
    )

    fig.update_layout(
        title=f"Institutional Recurrent GRU Performance (OOS Forecast RMSE: {metrics.final_oos_rmse:.5f} bps)",
        xaxis_title="Forward Test Horizon Timeline",
        yaxis_title="Financing Spread (bps)",
        template="plotly_white",
        height=550,
        width=1100,
        hovermode="x unified",
    )

    logger.info("Saving visualization artifacts to disk...")
    fig.write_html("recurrent_performance.html")
    try:
        fig.write_image("recurrent_performance.png", width=1100, height=550, scale=2)
        logger.info("Artifacts saved as recurrent_performance.html and recurrent_performance.png")
    except ValueError as e:
        logger.warning(f"Static image export bypassed. Verify kaleido dependency. Error: {e}")


if __name__ == "__main__":
    # Generate mock cross-asset financing dataset
    np.random.seed(42)
    total_days = 800
    date_axis = pd.date_range(start="2023-01-01", periods=total_days, freq="B")

    # Features: Underlying interest rate indexes and volatility metrics
    base_rate = np.cumsum(np.random.normal(0, 0.04, total_days)) + 4.0
    volatility = np.abs(np.random.normal(15.0, 3.0, total_days))
    utilization_rate = np.random.uniform(0.6, 0.95, total_days)

    features = np.stack([base_rate, volatility, utilization_rate], axis=1)

    # Target spread with a non-linear combination and periodic quarter-end spikes
    target = 0.25 * base_rate + 0.05 * volatility + np.sin(np.arange(total_days) * (2 * np.pi / 63)) * 0.5
    target += np.random.normal(0, 0.08, total_days)

    # Train/Test Split (80/20)
    split_idx = int(total_days * 0.8)
    X_train, X_test = features[:split_idx], features[split_idx:]
    y_train, y_test = target[:split_idx], target[split_idx:]
    test_timestamps = list(date_axis[split_idx:])

    # Run the pipeline
    pipeline = GRUForecastingPipeline(input_dim=features.shape[1], hidden_dim=24, num_layers=2)
    execution_metrics = pipeline.execute_lifecycle(
        X_train,
        y_train,
        X_test,
        y_test,
        test_dates=test_timestamps,
        lookback=20,
        epochs=25,
        batch_size=32,
    )

    # Render diagnostics
    generate_visual_artifacts(execution_metrics)