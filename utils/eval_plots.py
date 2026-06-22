import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _ensure_dir(save_dir):
    os.makedirs(save_dir, exist_ok=True)


def _as_numpy(values):
    return np.asarray(values)


def _majority_vote_by_group(values, group_ids, num_classes):
    values = _as_numpy(values).astype(int)
    group_ids = _as_numpy(group_ids)

    voted = []
    for group_id in np.unique(group_ids):
        group_values = values[group_ids == group_id]
        counts = np.bincount(group_values, minlength=num_classes)
        voted.append(counts.argmax())

    return np.asarray(voted)


def _mean_score_by_group(scores, group_ids):
    scores = _as_numpy(scores)
    group_ids = _as_numpy(group_ids)

    return np.asarray([
        scores[group_ids == group_id].mean()
        for group_id in np.unique(group_ids)
    ])


def _try_import_sklearn():
    try:
        from sklearn.metrics import (
            classification_report,
            confusion_matrix,
            roc_auc_score,
            roc_curve,
        )
    except ImportError:
        return None

    return {
        "classification_report": classification_report,
        "confusion_matrix": confusion_matrix,
        "roc_auc_score": roc_auc_score,
        "roc_curve": roc_curve,
    }


def plot_confusion_matrix(y_true, y_pred, class_names, save_path, normalize=False):
    metrics = _try_import_sklearn()
    if metrics is None:
        print("sklearn not installed, skipped confusion matrix plot.")
        return

    y_true = _as_numpy(y_true)
    y_pred = _as_numpy(y_pred)
    labels = np.arange(len(class_names))
    cm = metrics["confusion_matrix"](y_true, y_pred, labels=labels)

    if normalize:
        row_sum = cm.sum(axis=1, keepdims=True)
        cm_to_plot = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum != 0)
        fmt = ".2f"
        title = "Normalized Confusion Matrix"
        colorbar_label = "Recall"
    else:
        cm_to_plot = cm
        fmt = "d"
        title = "Confusion Matrix"
        colorbar_label = "Count"

    fig, ax = plt.subplots(figsize=(max(7.0, len(class_names)), max(6.0, 0.85 * len(class_names))))
    im = ax.imshow(cm_to_plot, interpolation="nearest", cmap="Blues")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)

    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(labels)
    ax.set_yticks(labels)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    threshold = cm_to_plot.max() / 2.0 if cm_to_plot.size else 0.0
    for i in range(cm_to_plot.shape[0]):
        for j in range(cm_to_plot.shape[1]):
            value = cm_to_plot[i, j]
            ax.text(
                j,
                i,
                format(value, fmt),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=9,
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_accuracy(y_true, y_pred, class_names, save_path):
    y_true = _as_numpy(y_true)
    y_pred = _as_numpy(y_pred)

    accuracies = []
    for class_idx in range(len(class_names)):
        mask = y_true == class_idx
        accuracies.append(float((y_pred[mask] == y_true[mask]).mean()) if mask.sum() else 0.0)

    x = np.arange(len(class_names))
    fig, ax = plt.subplots(figsize=(max(8.0, 1.1 * len(class_names)), 5.0))
    bars = ax.bar(x, accuracies, color="#4C78A8")
    ax.set_title("Per-class Accuracy")
    ax.set_xlabel("Class")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2.0, acc + 0.02, f"{acc:.2f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_classification_report(y_true, y_pred, class_names, save_path):
    metrics = _try_import_sklearn()
    # for test
    if metrics is None:
        print("sklearn not installed, skipped classification report plot.")
        return

    report = metrics["classification_report"](
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    precision = [report[name]["precision"] for name in class_names]
    recall = [report[name]["recall"] for name in class_names]
    f1 = [report[name]["f1-score"] for name in class_names]

    x = np.arange(len(class_names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(9.0, 1.25 * len(class_names)), 5.5))
    ax.bar(x - width, precision, width, label="Precision", color="#4C78A8")
    ax.bar(x, recall, width, label="Recall", color="#F58518")
    ax.bar(x + width, f1, width, label="F1-score", color="#54A24B")
    ax.set_title("Classification Report")
    ax.set_xlabel("Class")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curve(id_scores, ood_scores, save_path, title):
    metrics = _try_import_sklearn()
    if metrics is None:
        print(f"sklearn not installed, skipped {title} plot.")
        return

    id_scores = _as_numpy(id_scores)
    ood_scores = _as_numpy(ood_scores)
    y_true = np.concatenate([
        np.zeros_like(id_scores, dtype=int),
        np.ones_like(ood_scores, dtype=int),
    ])
    y_score = np.concatenate([id_scores, ood_scores])

    fpr, tpr, _ = metrics["roc_curve"](y_true, y_score)
    auroc = metrics["roc_auc_score"](y_true, y_score)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot(fpr, tpr, color="#4C78A8", linewidth=2.0, label=f"AUROC = {auroc:.4f}")
    ax.plot([0, 1], [0, 1], color="#888888", linestyle="--", linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_score_distribution(id_scores, ood_scores, save_path, title, xlabel):
    id_scores = _as_numpy(id_scores)
    ood_scores = _as_numpy(ood_scores)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.hist(id_scores, bins=40, density=True, alpha=0.65, label="ID", color="#4C78A8")
    ax.hist(ood_scores, bins=40, density=True, alpha=0.65, label="OOD", color="#F58518")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_all_evaluation_plots(
    y_true,
    y_pred,
    id_energy,
    ood_energy,
    id_dyn,
    ood_dyn,
    id_fusion,
    ood_fusion,
    save_dir,
    class_names,
    id_group_ids=None,
    ood_group_ids=None,
):
    _ensure_dir(save_dir)

    plot_confusion_matrix(y_true, y_pred, class_names, os.path.join(save_dir, "confusion_matrix.png"), normalize=False)
    plot_confusion_matrix(y_true, y_pred, class_names, os.path.join(save_dir, "normalized_confusion_matrix.png"), normalize=True)
    plot_per_class_accuracy(y_true, y_pred, class_names, os.path.join(save_dir, "per_class_accuracy.png"))
    plot_classification_report(y_true, y_pred, class_names, os.path.join(save_dir, "classification_report.png"))

    plot_roc_curve(id_energy, ood_energy, os.path.join(save_dir, "energy_score_roc.png"), "Energy Score ROC")
    plot_roc_curve(id_dyn, ood_dyn, os.path.join(save_dir, "dynamics_score_roc.png"), "Dynamics Score ROC")
    plot_roc_curve(id_fusion, ood_fusion, os.path.join(save_dir, "fusion_score_roc.png"), "Fusion Score ROC")

    plot_score_distribution(id_energy, ood_energy, os.path.join(save_dir, "energy_score_distribution.png"), "Energy Score Distribution", "Energy score")
    plot_score_distribution(id_dyn, ood_dyn, os.path.join(save_dir, "dynamics_error_distribution.png"), "Dynamics Error Distribution", "Dynamics error")
    plot_score_distribution(id_fusion, ood_fusion, os.path.join(save_dir, "fusion_score_distribution.png"), "Fusion Score Distribution", "Fusion score")

    if id_group_ids is not None and ood_group_ids is not None:
        y_true_sample = _majority_vote_by_group(y_true, id_group_ids, len(class_names))
        y_pred_sample = _majority_vote_by_group(y_pred, id_group_ids, len(class_names))

        plot_confusion_matrix(
            y_true_sample,
            y_pred_sample,
            class_names,
            os.path.join(save_dir, "sample_level_confusion_matrix.png"),
            normalize=False,
        )
        plot_confusion_matrix(
            y_true_sample,
            y_pred_sample,
            class_names,
            os.path.join(save_dir, "sample_level_normalized_confusion_matrix.png"),
            normalize=True,
        )
        plot_per_class_accuracy(
            y_true_sample,
            y_pred_sample,
            class_names,
            os.path.join(save_dir, "sample_level_per_class_accuracy.png"),
        )
        plot_classification_report(
            y_true_sample,
            y_pred_sample,
            class_names,
            os.path.join(save_dir, "sample_level_classification_report.png"),
        )

        id_energy_sample = _mean_score_by_group(id_energy, id_group_ids)
        ood_energy_sample = _mean_score_by_group(ood_energy, ood_group_ids)
        id_dyn_sample = _mean_score_by_group(id_dyn, id_group_ids)
        ood_dyn_sample = _mean_score_by_group(ood_dyn, ood_group_ids)
        id_fusion_sample = _mean_score_by_group(id_fusion, id_group_ids)
        ood_fusion_sample = _mean_score_by_group(ood_fusion, ood_group_ids)

        plot_roc_curve(
            id_energy_sample,
            ood_energy_sample,
            os.path.join(save_dir, "sample_level_energy_score_roc.png"),
            "Sample-level Energy Score ROC",
        )
        plot_roc_curve(
            id_dyn_sample,
            ood_dyn_sample,
            os.path.join(save_dir, "sample_level_dynamics_score_roc.png"),
            "Sample-level Dynamics Score ROC",
        )
        plot_roc_curve(
            id_fusion_sample,
            ood_fusion_sample,
            os.path.join(save_dir, "sample_level_fusion_score_roc.png"),
            "Sample-level Fusion Score ROC",
        )

        plot_score_distribution(
            id_energy_sample,
            ood_energy_sample,
            os.path.join(save_dir, "sample_level_energy_score_distribution.png"),
            "Sample-level Energy Score Distribution",
            "Energy score",
        )
        plot_score_distribution(
            id_dyn_sample,
            ood_dyn_sample,
            os.path.join(save_dir, "sample_level_dynamics_error_distribution.png"),
            "Sample-level Dynamics Error Distribution",
            "Dynamics error",
        )
        plot_score_distribution(
            id_fusion_sample,
            ood_fusion_sample,
            os.path.join(save_dir, "sample_level_fusion_score_distribution.png"),
            "Sample-level Fusion Score Distribution",
            "Fusion score",
        )

    print(f"Saved evaluation plots to {save_dir}")

