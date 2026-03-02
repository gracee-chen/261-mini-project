"""
Evaluation metrics: accuracy, per-class accuracy, confusion matrix,
precision/recall/F1 (macro & weighted), top-k accuracy.
"""

from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    class_names: list[str] | None = None,
    top_k: int = 5,
    num_classes: int | None = None,
) -> dict:
    """
    Compute all evaluation metrics.
    y_true, y_pred: (N,) integer labels
    y_prob: (N, num_classes) probabilities for top-k (optional)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if num_classes is None:
        num_classes = int(max(y_true.max(), y_pred.max()) + 1)

    metrics = {}

    # Overall accuracy
    metrics["accuracy"] = accuracy_score(y_true, y_pred)

    # Per-class accuracy (recall per class)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    per_class_correct = np.diag(cm)
    per_class_total = cm.sum(axis=1)
    per_class_acc = np.where(per_class_total > 0, per_class_correct / per_class_total, 0.0)
    metrics["per_class_accuracy"] = per_class_acc
    metrics["per_class_total"] = per_class_total

    # Confusion matrix
    metrics["confusion_matrix"] = cm

    # Precision, Recall, F1 (macro & weighted)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    metrics["precision_per_class"] = precision
    metrics["recall_per_class"] = recall
    metrics["f1_per_class"] = f1

    for avg in ["macro", "weighted"]:
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, average=avg, zero_division=0
        )
        metrics[f"precision_{avg}"] = p
        metrics[f"recall_{avg}"] = r
        metrics[f"f1_{avg}"] = f

    # Top-k accuracy
    if y_prob is not None and y_prob.shape[1] >= top_k:
        top_k_preds = np.argsort(y_prob, axis=1)[:, -top_k:]
        metrics[f"top_{top_k}_accuracy"] = np.mean(
            [y_true[i] in top_k_preds[i] for i in range(len(y_true))]
        )
    else:
        metrics[f"top_{top_k}_accuracy"] = None

    metrics["class_names"] = class_names
    metrics["num_classes"] = num_classes

    return metrics


def print_metrics(metrics: dict, top_k: int = 5) -> None:
    """Print metrics to console."""
    print("\n" + "=" * 60)
    print("EVALUATION METRICS")
    print("=" * 60)

    print(f"\nOverall Accuracy: {metrics['accuracy']:.4f}")

    if metrics.get(f"top_{top_k}_accuracy") is not None:
        print(f"Top-{top_k} Accuracy: {metrics[f'top_{top_k}_accuracy']:.4f}")

    print("\n--- Macro Averages ---")
    print(f"Precision: {metrics['precision_macro']:.4f}")
    print(f"Recall:    {metrics['recall_macro']:.4f}")
    print(f"F1-Score:  {metrics['f1_macro']:.4f}")

    print("\n--- Weighted Averages ---")
    print(f"Precision: {metrics['precision_weighted']:.4f}")
    print(f"Recall:    {metrics['recall_weighted']:.4f}")
    print(f"F1-Score:  {metrics['f1_weighted']:.4f}")

    class_names = metrics.get("class_names")
    if class_names and len(class_names) <= 50:
        print("\n--- Per-Class Accuracy (sample, first 10) ---")
        for i in range(min(10, len(class_names))):
            name = class_names[i] if i < len(class_names) else f"Class_{i}"
            acc = metrics["per_class_accuracy"][i]
            total = int(metrics["per_class_total"][i])
            print(f"  {name}: {acc:.4f} ({total} samples)")
        if len(class_names) > 10:
            print(f"  ... ({len(class_names)} classes total)")
    elif class_names:
        print(f"\nPer-class accuracy: {len(class_names)} classes (see saved report)")

    print("=" * 60)


def save_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    save_path: Path,
    figsize: tuple[int, int] | None = None,
) -> None:
    """Save confusion matrix as image."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    n = len(class_names)
    if figsize is None:
        figsize = (max(12, n * 0.3), max(10, n * 0.25)) if n <= 50 else (16, 14)

    plt.figure(figsize=figsize)
    show_labels = n <= 50
    sns.heatmap(
        cm,
        xticklabels=class_names if show_labels else False,
        yticklabels=class_names if show_labels else False,
        annot=n <= 30,
        fmt="d",
        cmap="Blues",
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_metrics_report(metrics: dict, save_path: Path, top_k: int = 5) -> None:
    """Save full metrics to text file."""
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("EVALUATION METRICS REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Overall Accuracy: {metrics['accuracy']:.4f}\n")
        top_k_key = f"top_{top_k}_accuracy"
        if metrics.get(top_k_key) is not None:
            f.write(f"Top-{top_k} Accuracy: {metrics[top_k_key]:.4f}\n")
        f.write("\nMacro - Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}\n".format(
            metrics["precision_macro"], metrics["recall_macro"], metrics["f1_macro"]
        ))
        f.write("Weighted - Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}\n".format(
            metrics["precision_weighted"], metrics["recall_weighted"], metrics["f1_weighted"]
        ))
        f.write("\n--- Per-Class Accuracy ---\n")
        class_names = metrics.get("class_names") or [f"Class_{i}" for i in range(metrics["num_classes"])]
        for i, name in enumerate(class_names):
            acc = metrics["per_class_accuracy"][i]
            total = int(metrics["per_class_total"][i])
            f.write(f"  {name}: {acc:.4f} ({total} samples)\n")
