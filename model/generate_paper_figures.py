"""
Generate publication-quality evaluation figures for the paper.
Run after training all models:
    cd model/
    python generate_paper_figures.py

Outputs saved to ../figures/
"""

import sys
import json
from pathlib import Path

import numpy as np
import torch
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

# ---------- paths ----------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_SCRIPT_DIR))

from dataset import get_dataloaders, get_data_root
from models import build_model
from models.svm_classifier import extract_features, FeatureExtractor

FIGURE_DIR = _PROJECT_ROOT / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# ---------- matplotlib global style (publication) ----------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.4,
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# model display config
MODEL_NAMES = ["resnet50", "efficientnet_b2", "vit_b_16", "convnext_tiny", "svm_resnet_features"]
DISPLAY_NAMES = ["ResNet-50", "EfficientNet-B2", "ViT-B/16", "ConvNeXt-Tiny", "SVM+ResNet-18"]
# colorblind-friendly palette (Tol bright)
COLORS = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#AA3377"]
HATCHES = ["//", "\\\\", "||", "--", "xx"]


# ================================================================
# 1. Evaluate all models → collect raw predictions
# ================================================================
def evaluate_all():
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    _, _, test_loader = get_dataloaders(batch_size=32, num_workers=4)
    class_names = test_loader.dataset.classes
    num_classes = len(class_names)

    results = {}
    for model_name in MODEL_NAMES:
        print(f"\nEvaluating {model_name}...")

        if model_name == "svm_resnet_features":
            ckpt_path = _PROJECT_ROOT / "checkpoints" / "svm_resnet_features" / "svm.joblib"
            data = joblib.load(ckpt_path)
            svm, scaler = data["svm"], data["scaler"]
            X_test, y_true = extract_features(test_loader, device)
            X_scaled = scaler.transform(X_test)
            y_pred = svm.predict(X_scaled)
            y_prob = svm.predict_proba(X_scaled) if hasattr(svm, "predict_proba") else None
            y_true, y_pred = np.array(y_true), np.array(y_pred)
        else:
            ckpt_path = _PROJECT_ROOT / "checkpoints" / model_name / "best.pt"
            model = build_model(model_name, num_classes)
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            model.load_state_dict(ckpt["model"])
            model = model.to(device).eval()
            all_preds, all_labels, all_probs = [], [], []
            with torch.no_grad():
                for imgs, labels in test_loader:
                    imgs = imgs.to(device)
                    logits = model(imgs)
                    probs = torch.softmax(logits, dim=1)
                    all_preds.extend(logits.argmax(1).cpu().numpy())
                    all_labels.extend(labels.numpy())
                    all_probs.append(probs.cpu().numpy())
            y_true = np.array(all_labels)
            y_pred = np.array(all_preds)
            y_prob = np.vstack(all_probs)

        # compute metrics
        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
        per_class_acc = np.where(cm.sum(1) > 0, np.diag(cm) / cm.sum(1), 0.0)
        prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
        prec_pc, rec_pc, f1_pc, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)

        top5 = None
        if y_prob is not None and y_prob.shape[1] >= 5:
            top5_preds = np.argsort(y_prob, axis=1)[:, -5:]
            top5 = np.mean([y_true[i] in top5_preds[i] for i in range(len(y_true))])

        results[model_name] = {
            "accuracy": acc, "top5": top5,
            "precision_macro": prec_m, "recall_macro": rec_m, "f1_macro": f1_m,
            "precision_weighted": prec_w, "recall_weighted": rec_w, "f1_weighted": f1_w,
            "per_class_accuracy": per_class_acc,
            "precision_per_class": prec_pc, "recall_per_class": rec_pc, "f1_per_class": f1_pc,
            "confusion_matrix": cm,
            "y_true": y_true, "y_pred": y_pred,
        }
        print(f"  Accuracy: {acc:.4f}  |  Top-5: {top5:.4f}" if top5 else f"  Accuracy: {acc:.4f}")

    return results, class_names, num_classes


# ================================================================
# Figure 1: Overall comparison bar chart (Acc, P, R, F1)
# ================================================================
def fig1_overall_comparison(results):
    metrics_keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
    n_models = len(MODEL_NAMES)
    n_metrics = len(metrics_keys)

    fig, ax = plt.subplots(figsize=(7, 3.2))
    x = np.arange(n_metrics)
    width = 0.15
    offsets = np.arange(n_models) - (n_models - 1) / 2

    for i, (mname, dname) in enumerate(zip(MODEL_NAMES, DISPLAY_NAMES)):
        vals = [results[mname][k] for k in metrics_keys]
        bars = ax.bar(x + offsets[i] * width, vals, width * 0.9,
                      label=dname, color=COLORS[i], edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f"{v:.1%}" if v < 1 else "100%",
                    ha="center", va="bottom", fontsize=5.5, rotation=90)

    ax.set_ylim(0.82, 1.02)
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(loc="lower left", ncol=3, frameon=True, framealpha=0.9,
              edgecolor="gray", fancybox=False, columnspacing=0.8)
    ax.set_title("(a) Test Set Performance Comparison", fontsize=11, pad=8)

    fig.savefig(FIGURE_DIR / "fig1_overall_comparison.pdf")
    fig.savefig(FIGURE_DIR / "fig1_overall_comparison.png")
    plt.close(fig)
    print("Saved fig1_overall_comparison")


# ================================================================
# Figure 2: Top-1 vs Top-5 accuracy grouped bar
# ================================================================
def fig2_top1_vs_top5(results):
    fig, ax = plt.subplots(figsize=(5.5, 3))
    x = np.arange(len(MODEL_NAMES))
    width = 0.32

    top1 = [results[m]["accuracy"] for m in MODEL_NAMES]
    top5 = [results[m]["top5"] if results[m]["top5"] is not None else 0 for m in MODEL_NAMES]

    bars1 = ax.bar(x - width / 2, top1, width, label="Top-1", color="#4477AA", edgecolor="white")
    bars2 = ax.bar(x + width / 2, top5, width, label="Top-5", color="#66CCEE", edgecolor="white")

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.003,
                        f"{h:.1%}", ha="center", va="bottom", fontsize=7)

    ax.set_ylim(0.85, 1.03)
    ax.set_ylabel("Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(DISPLAY_NAMES, rotation=15, ha="right")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(frameon=True, framealpha=0.9, edgecolor="gray", fancybox=False)
    ax.set_title("(b) Top-1 vs Top-5 Accuracy", fontsize=11, pad=8)

    fig.savefig(FIGURE_DIR / "fig2_top1_top5.pdf")
    fig.savefig(FIGURE_DIR / "fig2_top1_top5.png")
    plt.close(fig)
    print("Saved fig2_top1_top5")


# ================================================================
# Figure 3: Per-class accuracy distribution (box + strip)
# ================================================================
def fig3_perclass_distribution(results):
    fig, ax = plt.subplots(figsize=(5.5, 3.2))

    data = [results[m]["per_class_accuracy"] for m in MODEL_NAMES]
    bp = ax.boxplot(data, positions=range(len(MODEL_NAMES)), widths=0.5,
                    patch_artist=True, showfliers=True,
                    flierprops=dict(marker="o", markersize=2.5, alpha=0.4),
                    medianprops=dict(color="black", linewidth=1.2),
                    whiskerprops=dict(linewidth=0.8),
                    capprops=dict(linewidth=0.8))
    for patch, color in zip(bp["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        patch.set_edgecolor("gray")

    # scatter individual points
    for i, d in enumerate(data):
        jitter = np.random.default_rng(42).normal(0, 0.06, size=len(d))
        ax.scatter(np.full_like(d, i) + jitter, d, s=4, alpha=0.3, color=COLORS[i], zorder=3)

    ax.set_xticks(range(len(MODEL_NAMES)))
    ax.set_xticklabels(DISPLAY_NAMES, rotation=15, ha="right")
    ax.set_ylabel("Per-Class Accuracy")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_title("(c) Per-Class Accuracy Distribution", fontsize=11, pad=8)

    fig.savefig(FIGURE_DIR / "fig3_perclass_box.pdf")
    fig.savefig(FIGURE_DIR / "fig3_perclass_box.png")
    plt.close(fig)
    print("Saved fig3_perclass_box")


# ================================================================
# Figure 4: Confusion matrices (best model + SVM side by side)
# ================================================================
def fig4_confusion_matrices(results, class_names):
    n = len(class_names)

    for mname, dname in [("convnext_tiny", "ConvNeXt-Tiny"), ("svm_resnet_features", "SVM+ResNet-18")]:
        cm = results[mname]["confusion_matrix"]
        # normalize by row
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Recall", fontsize=9)

        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Label", fontsize=10)
        ax.set_title(f"Normalized Confusion Matrix — {dname}", fontsize=11, pad=10)

        # show ticks only every 10 classes
        tick_positions = list(range(0, n, 10))
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_positions, fontsize=7)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_positions, fontsize=7)

        safe_name = mname.replace("/", "_")
        fig.savefig(FIGURE_DIR / f"fig4_cm_{safe_name}.pdf")
        fig.savefig(FIGURE_DIR / f"fig4_cm_{safe_name}.png")
        plt.close(fig)
    print("Saved fig4 confusion matrices")


# ================================================================
# Figure 5: Precision / Recall / F1 per model (radar or grouped)
# ================================================================
def fig5_precision_recall_f1(results):
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8), sharey=True)
    metric_pairs = [
        ("precision_macro", "precision_weighted", "Precision"),
        ("recall_macro", "recall_weighted", "Recall"),
        ("f1_macro", "f1_weighted", "F1-Score"),
    ]

    x = np.arange(len(MODEL_NAMES))
    width = 0.32

    for ax, (macro_key, weighted_key, title) in zip(axes, metric_pairs):
        macro_vals = [results[m][macro_key] for m in MODEL_NAMES]
        weighted_vals = [results[m][weighted_key] for m in MODEL_NAMES]

        ax.bar(x - width / 2, macro_vals, width, label="Macro", color="#4477AA", edgecolor="white")
        ax.bar(x + width / 2, weighted_vals, width, label="Weighted", color="#EE6677", edgecolor="white")

        ax.set_xticks(x)
        ax.set_xticklabels([n.split("-")[0].split("+")[0][:8] for n in DISPLAY_NAMES],
                           rotation=30, ha="right", fontsize=7)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0.82, 1.02)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

    axes[0].set_ylabel("Score")
    axes[0].legend(frameon=True, framealpha=0.9, edgecolor="gray",
                   fancybox=False, fontsize=7, loc="lower left")

    fig.suptitle("(d) Macro vs Weighted Metrics", fontsize=11, y=1.02)
    fig.savefig(FIGURE_DIR / "fig5_macro_weighted.pdf")
    fig.savefig(FIGURE_DIR / "fig5_macro_weighted.png")
    plt.close(fig)
    print("Saved fig5_macro_weighted")


# ================================================================
# Figure 6: Bottom-10 hardest classes (for best model)
# ================================================================
def fig6_hardest_classes(results, class_names):
    best_model = "convnext_tiny"
    per_class = results[best_model]["per_class_accuracy"]
    indices = np.argsort(per_class)[:10]

    fig, ax = plt.subplots(figsize=(5.5, 3))
    names = [class_names[i] for i in indices]
    accs = [per_class[i] for i in indices]

    bars = ax.barh(range(len(names)), accs, color="#EE6677", edgecolor="white", height=0.6)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Accuracy")
    ax.set_xlim(0, 1.05)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.invert_yaxis()

    for bar, v in zip(bars, accs):
        ax.text(bar.get_width() + 0.015, bar.get_y() + bar.get_height() / 2,
                f"{v:.1%}", va="center", fontsize=7.5)

    ax.set_title(f"(e) Hardest Classes — ConvNeXt-Tiny", fontsize=11, pad=8)

    fig.savefig(FIGURE_DIR / "fig6_hardest_classes.pdf")
    fig.savefig(FIGURE_DIR / "fig6_hardest_classes.png")
    plt.close(fig)
    print("Saved fig6_hardest_classes")


# ================================================================
# Figure 7: Per-class accuracy heatmap across all models
# ================================================================
def fig7_perclass_heatmap(results, class_names):
    matrix = np.array([results[m]["per_class_accuracy"] for m in MODEL_NAMES])  # (5, 102)

    fig, ax = plt.subplots(figsize=(10, 2.8))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0.4, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("Accuracy", fontsize=9)

    ax.set_yticks(range(len(MODEL_NAMES)))
    ax.set_yticklabels(DISPLAY_NAMES, fontsize=8)
    ax.set_xlabel("Class Index", fontsize=10)
    ax.set_title("(f) Per-Class Accuracy Across Models", fontsize=11, pad=8)

    fig.savefig(FIGURE_DIR / "fig7_perclass_heatmap.pdf")
    fig.savefig(FIGURE_DIR / "fig7_perclass_heatmap.png")
    plt.close(fig)
    print("Saved fig7_perclass_heatmap")


# ================================================================
# Save LaTeX table
# ================================================================
def save_latex_table(results):
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Test set performance of all models on Caltech-101.}")
    lines.append(r"\label{tab:main_results}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lccccc}")
    lines.append(r"\toprule")
    lines.append(r"Model & Acc. (\%) & Top-5 (\%) & Prec. & Rec. & F1 \\")
    lines.append(r"\midrule")

    best_acc = max(results[m]["accuracy"] for m in MODEL_NAMES)
    for mname, dname in zip(MODEL_NAMES, DISPLAY_NAMES):
        r = results[mname]
        acc_str = f"{r['accuracy']*100:.2f}"
        if r["accuracy"] == best_acc:
            acc_str = r"\textbf{" + acc_str + "}"
        top5_str = f"{r['top5']*100:.2f}" if r["top5"] is not None else "—"
        lines.append(
            f"  {dname} & {acc_str} & {top5_str} & "
            f"{r['precision_macro']:.4f} & {r['recall_macro']:.4f} & {r['f1_macro']:.4f} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    table_path = FIGURE_DIR / "table_main_results.tex"
    with open(table_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved {table_path}")


# ================================================================
# main
# ================================================================
if __name__ == "__main__":
    results, class_names, num_classes = evaluate_all()

    fig1_overall_comparison(results)
    fig2_top1_vs_top5(results)
    fig3_perclass_distribution(results)
    fig4_confusion_matrices(results, class_names)
    fig5_precision_recall_f1(results)
    fig6_hardest_classes(results, class_names)
    fig7_perclass_heatmap(results, class_names)
    save_latex_table(results)

    print(f"\nAll figures saved to {FIGURE_DIR}/")
