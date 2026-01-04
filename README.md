# Multivariate Time Series Forecasting with Graph Neural Networks
## Application to EEG Data

**Authors:** [Tom Mariani](https://github.com/t-mariani), [Hugo Pavy](https://github.com/hpavy)  
**Date:** January 2026

This repository contains a PyTorch implementation of the **MTGNN (Multivariate Time Series Graph Neural Network)** architecture, originally proposed by Wu et al. (2020). While the original framework was evaluated on traffic and solar data, this project extends the application to **Electroencephalography (EEG)** data, leveraging learned graph structures to capture inter-channel dependencies in brain activity.

```bibtex
@misc{wu2020connectingdotsmultivariatetime,
      title={Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks}, 
      author={Zonghan Wu and Shirui Pan and Guodong Long and Jing Jiang and Xiaojun Chang and Chengqi Zhang},
      year={2020},
      eprint={2005.11650},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2005.11650}, 
}
```

### 🔍 Overview

The goal of this work is to study the use of Graph Neural Networks (GNNs) for multivariate time series forecasting. We implement the model from scratch in a modular, object-oriented Python architecture.

**Key Contributions:**
* **From-scratch implementation** of the MTGNN architecture using PyTorch.
* **Adaptation for EEG**: Dedicated preprocessing pipeline (Bandpass filtering, Resampling, Normalization).
* **Comparison Benchmarks**: Evaluation against Autoregressive (AR) baselines.
* **Graph Learning**: Visualization of learned adjacency matrices to interpret spatial dependencies between EEG sensors.

You can find the report of our work in the file [report.pdf](report.pdf).

---

### 🧠 Methodology

The model operates on multivariate time series $X \in \mathbb{R}^{N \times T}$ to predict values at a specific horizon $h$. It simultaneously learns an underlying graph structure and performs forecasting.

#### Architecture Components
1.  **Graph Learning Layer**: Learns an adaptive adjacency matrix $A$ end-to-end, discovering hidden dependencies between sensors without needing a predefined physical topology.
2.  **Time Convolution Module**: Uses dilated inception layers to extract temporal features (short-term trends and long-term periodicities).
3.  **Graph Convolution Module**: Fuses information spatially using Mix-Hop propagation based on the learned graph.
4.  **Output Module**: Aggregates latent representations to generate final predictions.

#### EEG Specifics
Due to the non-stationary nature of EEG signals, we implement a strict preprocessing pipeline:
* **Filtering**: 0.5 Hz - 50 Hz Bandpass filter to remove drifts and power-line noise.
* **Resampling**: Downsampling to 100 Hz to reduce computational cost.
* **Normalization**: Channel-wise scaling (Z-score or Max).

---

### 💻 Installation

This project manages dependencies using **[Poetry](https://python-poetry.org)**.

**Install dependencies and environment:**

```
pip install poetry
poetry install
```

---

### 🚀 Minimal run snippet
Below is a minimal run sketch to train and evaluate on one dataset. Adjust paths and configs as needed.

1) Configure `config.yaml` with dataset, training rates, and model hyperparameters.
2) Run training (example):

```bash
poetry run python main.py
```

Outputs include TensorBoard logs (under `saved_models/.../runs/`), and best model weights.

- If you want to use the AR model, change config.yaml: 

```
model_kind: AR_global
```

- If you want to use the MTGNN, change config.yaml:

```
model_kind: MTGNN
```
---

### The dataset

- If you want to change the dataset, change config.yaml:
```
dataset_name : solar # or electricity / traffic / exchange
``` 

In order to download the dataset used in the paper, you can clone this repo: [Repo data](https://github.com/laiguokun/multivariate-time-series-data) in the parent folder of multi-time-gnn. 

#### Use of EEG

You can find open EEG data at this link: [open eeg data](https://openneuro.org).

If you want to use it, change the config.yaml:

```
dataset_name: eeg,
path_eeg: {the path to your .bdf eeg}

```


