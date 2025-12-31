import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
import numpy as np
import torch

from multi_time_gnn.dataset import Normalizer
from multi_time_gnn.test import predict_multi_step, prediction_step
from multi_time_gnn.utils import get_logger

log = get_logger()


def plot_batch(x, y, show=True):
    """
    Plots the input sequence and the target sequence for a batch.

    Parameters:
    - x: torch.Tensor of shape (batch_size, 1, N, timepoints_input)
    - y: torch.Tensor of shape (batch_size, 1, N, 1)
    """
    batch_size, _, N, timepoints_input = x.shape
    colors_name = list(TABLEAU_COLORS.keys())
    fig = plt.figure(figsize=(15, 5 * N))
    for i in range(N):
        plt.subplot(N, 1, i + 1)
        for b in range(batch_size):
            plt.plot(
                np.arange(timepoints_input),
                x[b, 0, i, :].cpu(),
                "-x",
                label=f"Input Seq Batch {b}",
                color=TABLEAU_COLORS[colors_name[b]],
            )
            plt.scatter(
                timepoints_input,
                y[b, 0, i, 0].cpu(),
                label=f"Target Batch {b}",
                color=TABLEAU_COLORS[colors_name[b]],
            )
        plt.title(f"Node {i}")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.grid()
    plt.tight_layout()
    if show:
        plt.show()
    return fig

def plot_signal_prediction(time, signal, label=None, color=None):
    """
    Plots a signal over time.

    Parameters:
    - time: array-like, time points
    - signal: array-like, signal values
    - label: str, label for the plot
    - color: str, color for the plot
    """
    N,T = signal.shape
    for i in range(N):
        plt.subplot(N, 1, i + 1)
        plt.plot(time, signal[i, :], label=label, color=color)
        plt.title(f"Node {i}")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.grid()

def plot_prediction(true, predicted_one_step, full_prediction, show=True):
    """
    Plots the true values, one-step predictions, and full multi-step predictions.

    Parameters:
    - true: np.ndarray of shape (N, T), the ground truth time series data.
    - predicted_one_step: np.ndarray of shape (N, T - timepoints_input),
      the one-step ahead predictions.
    - full_prediction: np.ndarray of shape (N, T - timepoints_input),
      the multi-step ahead predictions.
    """
    N, T = true.shape
    timepoints_input = T - predicted_one_step.shape[1]

    time_true = np.arange(T)
    time_pred = np.arange(timepoints_input, T)

    fig = plt.figure(figsize=(15 * T / 1000, 5 * N))
    plot_signal_prediction(time_true, true, "True", "tab:blue")
    plot_signal_prediction(time_pred, predicted_one_step, "One-step Prediction", "tab:orange")
    plot_signal_prediction(time_pred, full_prediction, "Multi-step Prediction", "tab:green")

    plt.tight_layout()
    if show:
        plt.show()
    return fig

def plot_prediction_horizons(true, signal_horizons, selected_horizons, show=True):
    """
    Plots the true values, one-step predictions, and full multi-step predictions.

    Parameters:
    - true: np.ndarray of shape (N, T), the ground truth time series data.
    - signal_horizons: np.ndarray of shape (n_horizons, N, T - timepoints_input),
    - selected_horizons : list of int, indices of horizons to plot
    """
    N, T = true.shape
    timepoints_input = T - signal_horizons.shape[2]

    time_true = np.arange(T)
    time_pred = np.arange(timepoints_input, T)

    fig = plt.figure(figsize=(15 * T / 1000, 5 * N))
    plot_signal_prediction(time_true, true, "True")
    for h in selected_horizons:
        signal = signal_horizons[h - 1]  # h starts at 1, shape (N, T - timepoints_input)
        plot_signal_prediction(time_pred + (h - 1), signal, f"Horizon {h}")

    plt.tight_layout()
    if show:
        plt.show()
    return fig


def plot_graph(adj_matrix, show=True):
    """
    Plots the adjacency matrix as a heatmap.

    Parameters:
    - adj_matrix: np.ndarray of shape (N, N), the adjacency matrix to plot.
    """
    fig = plt.figure(figsize=(8, 6))
    plt.imshow(adj_matrix, cmap="viridis", interpolation="nearest")
    plt.colorbar(label="Edge Weight")
    plt.title("Learned Adjacency Matrix")
    plt.xlabel("Node")
    plt.ylabel("Node")
    if show:
        plt.show()
    return fig


def pipeline_plotting(model, test_data, normalizer:Normalizer, config, clip_N: int | None = 10):
    """
    Generates predictions using the model and plots the results.

    Parameters:
    - model: trained model
    - test_data: np.ndarray of shape (N, T), the test time series data.
    - mean : Normalizer object used for denormalization
    - config : configuration
    - clip_N: int, number of nodes to visualize (default is 10)
    """
    N, T = test_data.shape
    timepoints_input = config.timepoints_input
    n_steps = T - timepoints_input - 1
    log.info(f" data shape N,T: {N,T} - timepoints_input: {timepoints_input} - n_steps: {n_steps}")

    # One-step ahead predictions
    predicted_one_step_points = prediction_step(
        model, config, test_data.copy(), 
    )  # Shape (N, n_steps)
    predicted_one_step_points = predicted_one_step_points.cpu().detach().numpy()
    log.info(f"predicted_one_step_points shape : {predicted_one_step_points.shape}")

    # Multi-step ahead predictions
    input_sequence = test_data[:,:config.timepoints_input]
    input_sequence = torch.from_numpy(input_sequence[None,None,:,:]).float().to(config.device)
    full_prediction = predict_multi_step(
        model, input_sequence, n_steps
    )  # Shape (N, n_steps)
    log.info(f"full_prediction shape : {full_prediction.shape}")

    predicted_one_step_points = normalizer.denormalize(predicted_one_step_points)
    full_prediction = normalizer.denormalize(full_prediction)
    test_data = normalizer.denormalize(test_data)
    if clip_N is not None:
        predicted_one_step_points = predicted_one_step_points[:clip_N,:]
        full_prediction = full_prediction[:clip_N,:] 
        test_data = test_data[:clip_N, :] # NxT

    log.info(" * Plotting Predictions")
    fig_predict = plot_prediction(
        true=test_data,
        predicted_one_step=predicted_one_step_points,
        full_prediction=full_prediction,
        show=False,
    )
    fig_predict.savefig(config.output_dir + "/predictions.png", dpi=300)

    log.info(" * Plotting Graph Learned")
    fig_graph = plot_graph(
        adj_matrix=model.graph_learn().cpu().detach().numpy(), show=False
    )
    fig_graph.savefig(config.output_dir + "/learned_graph.png")
