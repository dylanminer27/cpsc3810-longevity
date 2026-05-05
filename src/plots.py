# Plot helpers

import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as skmetrics


COLORS = [
    'tab:blue',
    'tab:green',
    'tab:red',
    'tab:orange',
    'tab:purple',
    'tab:brown',
    'tab:pink',
    'tab:gray',
    'tab:olive',
]


def loss_curves(losses_by_optimizer: Dict[str, List[float]],
                title: str,
                out_path: str) -> None:

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    for (label, losses), c in zip(losses_by_optimizer.items(), COLORS):
        ax.plot(np.arange(len(losses)), losses, label=label, color=c)
    ax.set_title(title)
    ax.set_xlabel('Logged step')
    ax.set_yscale('log') 
    ax.set_ylabel('Loss')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def confusion_matrix(y_true: np.ndarray,
                     y_pred: np.ndarray,
                     labels: list,
                     display_labels: list,
                     title: str,
                     out_path: str) -> None:

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cm = skmetrics.confusion_matrix(y_true, y_pred, labels=labels)
    disp = skmetrics.ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=display_labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def kmeans_elbow(ks: List[int],
                 inertias: List[float],
                 out_path: str) -> None:

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ks, inertias, marker='o', color='tab:blue')
    ax.set_title('K-Means inertia vs k')
    ax.set_xlabel('k')
    ax.set_ylabel('Inertia (within-cluster sum of squares)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

# Bar chart of mean lifestyle features per cluster
def cluster_lifestyle_profile(profile_means: np.ndarray,
                              feature_names: List[str],
                              out_path: str) -> None:
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    k, n = profile_means.shape
    width = 0.8 / k
    x = np.arange(n)
    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(k):
        ax.bar(x + i * width, profile_means[i], width, label='Cluster {}'.format(i), color=COLORS[i % len(COLORS)])
    ax.set_xticks(x + width * (k - 1) / 2)
    ax.set_xticklabels(feature_names, rotation=20, ha='right')
    ax.set_ylabel('Standardized mean')
    ax.set_title('Lifestyle profile by cluster')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

# Bar chart of mean age at death per cluster
def cluster_mean_age(cluster_ids: List[int],
                     mean_ages: List[float],
                     sizes: List[int],
                     out_path: str) -> None:

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        [str(c) for c in cluster_ids],
        mean_ages,
        color=[COLORS[i % len(COLORS)] for i in range(len(cluster_ids))])
    for bar, size, age in zip(bars, sizes, mean_ages):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                'n={}\n{:.1f}'.format(size, age),
                ha='center', va='bottom', fontsize=9)
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Mean age at death')
    ax.set_title('Mean age at death by cluster')
    ax.set_ylim(0, max(mean_ages) * 1.15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def correlation_heatmap(corr_matrix: np.ndarray,
                        feature_names: List[str],
                        out_path: str) -> None:
  
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(feature_names)))
    ax.set_yticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    ax.set_yticklabels(feature_names)
    fig.colorbar(im, ax=ax)
    ax.set_title('Feature correlation')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
