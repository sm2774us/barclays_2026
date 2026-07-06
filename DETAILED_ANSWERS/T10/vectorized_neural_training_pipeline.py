"""Vectorized neural training pipeline with manual Adam optimization.

Implements deep learning backpropagation, inverted dropout, weight decay, 
and an early stopping validation architecture.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go

# Configure infrastructure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s] %(message)s"
)
logger = logging.getLogger("FrontOffice-DeepLearning")


@dataclass(slots=True)
class AdamParameterState:
    """State containers tracking first and second momentum matrices for Adam."""
    m_w: np.ndarray
    v_w: np.ndarray
    m_b: np.ndarray
    v_b: np.ndarray


class RegularizedFinancialMLP:
    """Multi-layer perceptron featuring vectorized backpropagation and dropout."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, l2_lambda: float = 0.01) -> None:
        self.l2_lambda = l2_lambda
        
        # Xavier/Glorot Initialization for weight tensors
        self.w1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / (input_dim + hidden_dim))
        self.b1 = np.zeros((1, hidden_dim))
        self.w2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
        self.b2 = np.zeros((1, output_dim))

        # Internal operational activation caches
        self.cache_x: np.ndarray | None = None
        self.cache_z1: np.ndarray | None = None
        self.cache_a1: np.ndarray | None = None
        self.cache_mask1: np.ndarray | None = None

    def forward(self, x: np.ndarray, dropout_prob: float = 0.0, is_training: bool = True) -> np.ndarray:
        """Executes forward path computations, caching layer activations."""
        self.cache_x = x
        self.cache_z1 = np.dot(x, self.w1) + self.b1
        self.cache_a1 = np.maximum(0, self.cache_z1)  # ReLU Activation Function

        if is_training and dropout_prob > 0.0:
            # Construct and apply inverted dropout mask
            mask = (np.random.rand(*self.cache_a1.shape) >= dropout_prob) / (1.0 - dropout_prob)
            self.cache_a1 = self.cache_a1 * mask
            self.cache_mask1 = mask
        else:
            self.cache_mask1 = None

        out = np.dot(self.cache_a1, self.w2) + self.b2
        return out

    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Calculates the mean squared error combined with an L2 weight decay penalty."""
        n = y_true.shape[0]
        data_loss = (1.0 / (2.0 * n)) * np.sum((y_pred - y_true) ** 2)
        
        # Compute Frobenius norm regularization penalty
        l2_penalty = (self.l2_lambda / 2.0) * (np.sum(self.w1 ** 2) + np.sum(self.w2 ** 2))
        return float(data_loss + l2_penalty)

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> dict[str, np.ndarray]:
        """Calculates exact analytical gradients across layers using the chain rule."""
        n = y_true.shape[0]
        
        # Output layer gradient calculation
        dy_pred = (1.0 / n) * (y_pred - y_true)
        dw2 = np.dot(self.cache_a1.T, dy_pred) + self.l2_lambda * self.w2
        db2 = np.sum(dy_pred, axis=0, keepdims=True)

        # Propagate error back to hidden layer
        da1 = np.dot(dy_pred, self.w2.T)
        if self.cache_mask1 is not None:
            da1 = da1 * self.cache_mask1  # Propagate gradients through dropout mask

        # ReLU backpropagation step
        dz1 = da1.copy()
        dz1[self.cache_z1 <= 0] = 0.0
        
        dw1 = np.dot(self.cache_x.T, dz1) + self.l2_lambda * self.w1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        return {"w1": dw1, "b1": db1, "w2": dw2, "b2": db2}


class VectorizedAdamOptimizer:
    """Stateful Adam optimizer that applies momentum and adaptive step sizes."""

    def __init__(self, model: RegularizedFinancialMLP, lr: float = 0.005, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
        self.model = model
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        
        # Initialize tracking moment structures for each parameter array
        self.states = {
            "layer1": AdamParameterState(np.zeros_like(model.w1), np.zeros_like(model.w1), np.zeros_like(model.b1), np.zeros_like(model.b1)),
            "layer2": AdamParameterState(np.zeros_like(model.w2), np.zeros_like(model.w2), np.zeros_like(model.b2), np.zeros_like(model.b2))
        }

    def step(self, grads: dict[str, np.ndarray]) -> None:
        """Updates model weights using bias-corrected moment velocity vectors."""
        self.t += 1
        
        # Update Layer 1 parameters
        s1 = self.states["layer1"]
        s1.m_w = self.beta1 * s1.m_w + (1.0 - self.beta1) * grads["w1"]
        s1.v_w = self.beta2 * s1.v_w + (1.0 - self.beta2) * (grads["w1"] ** 2)
        m_w_hat = s1.m_w / (1.0 - self.beta1 ** self.t)
        v_w_hat = s1.v_w / (1.0 - self.beta2 ** self.t)
        self.model.w1 -= self.lr * m_w_hat / (np.sqrt(v_w_hat) + self.eps)

        s1.m_b = self.beta1 * s1.m_b + (1.0 - self.beta1) * grads["b1"]
        s1.v_b = self.beta2 * s1.v_b + (1.0 - self.beta2) * (grads["b1"] ** 2)
        m_b_hat = s1.m_b / (1.0 - self.beta1 ** self.t)
        v_b_hat = s1.v_b / (1.0 - self.beta2 ** self.t)
        self.model.b1 -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.eps)

        # Update Layer 2 parameters
        s2 = self.states["layer2"]
        s2.m_w = self.beta1 * s2.m_w + (1.0 - self.beta1) * grads["w2"]
        s2.v_w = self.beta2 * s2.v_w + (1.0 - self.beta2) * (grads["w2"] ** 2)
        m_w_hat2 = s2.m_w / (1.0 - self.beta1 ** self.t)
        v_w_hat2 = s2.v_w / (1.0 - self.beta2 ** self.t)
        self.model.w2 -= self.lr * m_w_hat2 / (np.sqrt(v_w_hat2) + self.eps)

        s2.m_b = self.beta1 * s2.m_b + (1.0 - self.beta1) * grads["b2"]
        s2.v_b = self.beta2 * s2.v_b + (1.0 - self.beta2) * (grads["b2"] ** 2)
        m_b_hat2 = s2.m_b / (1.0 - self.beta1 ** self.t)
        v_b_hat2 = s2.v_b / (1.0 - self.beta2 ** self.t)
        self.model.b2 -= self.lr * m_b_hat2 / (np.sqrt(v_b_hat2) + self.eps)


# --- Infrastructure Simulation & Diagnostic Visualization Drivers ---

def generate_synthetic_desk_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generates synthetic multi-asset feature arrays containing structured noise parameters."""
    np.random.seed(42)
    # Generate 2500 samples resembling fee quote patterns
    total_samples = 2500
    features_count = 10
    
    x_space = np.random.randn(total_samples, features_count)
    # Construct a nonlinear hidden target sequence
    true_beta = np.array([[1.5], [-2.0], [0.5], [0.0], [0.0], [1.2], [-0.8], [0.0], [0.3], [0.1]])
    y_space = np.dot(x_space, true_beta) + np.sin(x_space[:, :1]) * 1.5 + np.random.randn(total_samples, 1) * 0.7

    # Split into training and validation subsets (80/20)
    split_index = 2000
    return x_space[:split_index], y_space[:split_index], x_space[split_index:], y_space[split_index:]


def execute_pipeline_training() -> tuple[list[float], list[float], int]:
    """Runs the training loop and implements early stopping to monitor model convergence."""
    x_train, y_train, x_val, y_val = generate_synthetic_desk_data()
    
    # Initialize networks and the Adam optimizer
    network = RegularizedFinancialMLP(input_dim=10, hidden_dim=32, output_dim=1, l2_lambda=0.01)
    optimizer = VectorizedAdamOptimizer(model=network, lr=0.01)

    train_loss_history: list[float] = []
    val_loss_history: list[float] = []
    
    # Early stopping limits configuration
    patience = 15
    best_val_loss = float("inf")
    patience_counter = 0
    optimal_epoch = 0

    max_epochs = 200
    for epoch in range(max_epochs):
        # Forward pass with dropout enabled
        y_train_pred = network.forward(x_train, dropout_prob=0.15, is_training=True)
        train_loss = network.compute_loss(y_train_pred, y_train)
        
        # Backward error propagation and optimizer step
        gradients = network.backward(y_train_pred, y_train)
        optimizer.step(gradients)

        # Evaluate performance on validation data
        y_val_pred = network.forward(x_val, dropout_prob=0.0, is_training=False)
        val_loss = network.compute_loss(y_val_pred, y_val)

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            optimal_epoch = epoch
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.warning(f"Early stopping triggered at epoch {epoch+1}. Validation divergence caught.")
            break

    return train_loss_history, val_loss_history, optimal_epoch


def plot_loss_divergence(train_loss: list[float], val_loss: list[float], stop_point: int) -> None:
    """Generates chart reports analyzing optimization steps and early stopping checkpoints."""
    epochs = list(range(1, len(train_loss) + 1))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=train_loss, name="Training Set Loss Profile", mode="lines", line=dict(color="#2980b9", width=2)))
    fig.add_trace(go.Scatter(x=epochs, y=val_loss, name="Validation Set Loss Profile", mode="lines", line=dict(color="#e67e22", width=2)))
    
    # Highlight the early stopping checkpoint
    fig.add_vline(x=stop_point + 1, line_width=2, line_dash="dash", line_color="crimson")
    fig.add_annotation(x=stop_point + 1, y=max(train_loss)*0.7, text="Early Stopping Freeze Point", showarrow=True, arrowhead=1, ax=50, ay=-30)

    fig.update_layout(
        title="Optimization Dynamics: Vectorized Adam Gradient Descent with Capacity Control",
        xaxis_title="Operational Learning Iterations (Epochs)",
        yaxis_title="Calculated Objective Loss",
        template="plotly_white",
        width=950,
        height=500
    )
    fig.write_html("dl_training_profile.html")
    logger.info("Saved structural convergence dashboard visualization to 'dl_training_profile.html'.")


if __name__ == "__main__":
    t_start = time.perf_counter()
    history_train, history_val, best_epoch = execute_pipeline_training()
    compute_duration = (time.perf_counter() - t_start) * 1000.0
    
    print("\n" + "="*100)
    print("                     institutional neural network optimization log                     ")
    print("="*100)
    print(f" ├── Total Optimization Steps   : {len(history_train)} Active Learning Epochs")
    print(f" ├── Execution Compute Latency  : {compute_duration:.3f} ms (Total Training Iteration)")
    print(f" ├── Initial Validation Loss    : {history_val[0]:.6f}")
    print(f" ├── Optimal Model Checkpoint   : Epoch {best_epoch + 1}")
    print(f" └── Factual Validation Floor   : {history_val[best_epoch]:.6f} (Minimum Realized Error)")
    print("="*100 + "\n")

    plot_loss_divergence(history_train, history_val, best_epoch)