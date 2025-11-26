from types import NoneType
import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
import numpy as np
import torch
from multi_time_gnn.dataset import TimeSeriesDataset, get_batch
from multi_time_gnn.test import predict_multi_step, test_step
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


def plot_prediction(true, predicted_one_step, full_prediction, show=True):
    """
    Plots the true values, one-step predictions, and full multi-step predictions.

    Parameters:
    - true: np.ndarray of shape (N, T), the ground truth time series data.
    - predicted_one_step: np.ndarray of shape (N, T - timepoints_input - 1),
      the one-step ahead predictions.
    - full_prediction: np.ndarray of shape (N, T - timepoints_input - 1),
      the multi-step ahead predictions.
    """
    N, T = true.shape
    timepoints_input = T - predicted_one_step.shape[0] - 1

    time_true = np.arange(T)
    time_pred = np.arange(timepoints_input + 1, T)

    fig = plt.figure(figsize=(15, 5 * N))
    for i in range(N):
        plt.subplot(N, 1, i + 1)
        plt.plot(time_true, true[i, :], label="True", color="blue")
        plt.plot(
            time_pred,
            predicted_one_step[:, i],
            label="One-step Prediction",
            color="orange",
        )
        plt.plot(
            time_pred,
            full_prediction[:, i],
            label="Multi-step Prediction",
            color="green",
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


def pipeline_plotting(model, test_data, config, clip_N: int | NoneType = 10):
    """
    Generates predictions using the model and plots the results.

    Parameters:
    - model: trained model
    - test_data: np.ndarray of shape (N, T), the test time series data.
    - config : configuration
    - clip_N: int, number of nodes to visualize (default is 10)
    """
    N, T = test_data.shape
    timepoints_input = config.timepoints_input
    n_steps = T - timepoints_input - 1

    # One-step ahead predictions
    predicted_one_step_points = test_step(
        model, test_data, config=config
    )  # Shape (n_steps, N)

    # Multi-step ahead predictions)
    input_sequence, _ = get_batch(
        1, test_data, timepoints_input, 1, index=[0], device=config.device
    )  # Shape (1, 1, N, timepoints_input + 1)
    full_prediction = predict_multi_step(
        model, input_sequence, n_steps
    )  # Shape (N, n_steps)

    if clip_N is not None and N > clip_N:
        predicted_one_step_points = predicted_one_step_points[:, :clip_N].T
        full_prediction = full_prediction[:, :clip_N].T # NxT
        test_data = test_data[:clip_N, :] # NxT

    log.info(" * Plotting Predictions")
    fig_predict = plot_prediction(
        true=test_data,
        predicted_one_step=predicted_one_step_points,
        full_prediction=full_prediction,
        show=False,
    )
    fig_predict.savefig(config.output_dir + "/predictions.png")

    log.info(" * Plotting Graph Learned")
    fig_graph = plot_graph(
        adj_matrix=model.graph_learn().cpu().detach().numpy(), show=False
    )
    fig_graph.savefig(config.output_dir + "/learned_graph.png")
