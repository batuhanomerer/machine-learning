from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image, display
from sklearn.metrics import (
    accuracy_score, adjusted_rand_score, precision_recall_fscore_support,
    roc_auc_score, silhouette_score,
)

SEEDS = [42, 52, 62]
FINAL_SEED = 42
REPRESENTATIONS = ["unscaled", "scaled"]
FIGURES = Path("figures")


def show_figure(fig, dataset, name):
    FIGURES.mkdir(exist_ok=True)
    path = FIGURES / f"{dataset}_{name}.png"
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    display(Image(filename=str(path), width=850))


def classifier_metrics(y, pred, probabilities, classes, dataset):
    # Check denominators before asking sklearn to replace undefined scores.
    undefined_precision = [int(label) for label in classes if not (pred == label).any()]
    undefined_recall = [int(label) for label in classes if not (y == label).any()]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, pred, labels=classes, average="macro", zero_division=0
    )
    if dataset == "breast_cancer":
        malignant = list(classes).index(0)
        auc = roc_auc_score((y == 0).astype(int), probabilities[:, malignant])
    else:
        auc = roc_auc_score(y, probabilities, labels=classes, multi_class="ovr", average="macro")
    scores = dict(accuracy=accuracy_score(y, pred), macro_precision=precision,
                  macro_recall=recall, macro_f1=f1, roc_auc=auc)
    return scores, undefined_precision, undefined_recall


def cluster_metrics(X, y, clusters):
    occupied = len(np.unique(clusters))
    silhouette = np.nan
    if 2 <= occupied < len(X):
        silhouette = silhouette_score(X, clusters)
    return dict(ari=adjusted_rand_score(y, clusters), silhouette=silhouette)


def summarize(runs, keys, metrics):
    # Pandas groupby std uses sample standard deviation (ddof=1).
    summary = runs.groupby(keys, sort=False, dropna=False)[metrics].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def parameter_figure(summary, algorithm, settings, title):
    fig, ax = plt.subplots(figsize=(8.4, 3.4), layout="constrained")
    for index, (rep, color) in enumerate(zip(REPRESENTATIONS, ["#285783", "#b34d19"])):
        rows = summary.query("algorithm == @algorithm and representation == @rep")
        rows = rows.set_index("parameter_value").loc[settings]
        for j, (split, line) in enumerate([("train", "--"), ("val", "-")]):
            key = f"{split}_macro_f1"
            x = np.arange(len(settings)) + (index * 2 + j - 1.5) * 0.055
            split_name = {"train": "Training", "val": "Validation"}[split]
            ax.errorbar(x, rows[f"{key}_mean"], yerr=rows[f"{key}_std"],
                        fmt=f"o{line}", capsize=3, color=color, label=f"{rep.title()} · {split_name}")
    ax.set(xticks=range(len(settings)), xticklabels=settings,
           xlabel="Maximum depth" if algorithm == "Decision Tree" else "Number of neighbours",
           ylabel="Macro-F1", title=title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower left", ncol=2, fontsize=8)
    return fig
