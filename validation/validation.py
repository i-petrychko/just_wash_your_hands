import os
import sys
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from models.base_model import ImagePrediction
from validation.constants import IDX_TO_CLASS, NO_PARASITE_CLASS
from validation.utils import match_predictions


def plot_precision_recall_curve(
    y_trues: List[int], y_preds: List[int], y_scores: List[float], output_dir: str
):
    """Plot precision-recall curve for each class in a grid and a general PR curve separately"""

    # Get all unique classes
    all_classes = sorted(set(y_trues))

    # Create a grid for per class PR curves
    num_classes = len(all_classes)
    cols = 3
    rows = (num_classes + cols - 1) // cols  # Calculate number of rows needed
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()

    # Plot PR curve for each class
    for idx, class_id in enumerate(all_classes):
        # Binarize the output
        y_true_bin = []
        y_scores_bin = []

        for y_true, y_pred, y_score in zip(y_trues, y_preds, y_scores):
            y_true_bin.append(1 if y_true == class_id else 0)
            y_scores_bin.append(y_score if y_pred == class_id else 0)

        precision, recall, _ = precision_recall_curve(y_true_bin, y_scores_bin)
        average_precision = average_precision_score(y_true_bin, y_scores_bin)

        ax = axes[idx]
        ax.plot(recall, precision, lw=2, label=f"AP={average_precision:.2f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"Class {IDX_TO_CLASS[class_id]}")
        ax.legend(loc="best")

    # Remove empty subplots
    for idx in range(len(all_classes), len(axes)):
        fig.delaxes(axes[idx])

    # Save the grid of per class PR curves
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pr_curve_per_class_grid.png"))
    plt.close(fig)

    # Plot a single general PR curve
    plt.figure(figsize=(10, 8))

    y_true_bin = []
    y_scores_bin = []
    # Binarize the output for all classes combined
    for y_true, y_pred, y_score in zip(y_trues, y_preds, y_scores):
        y_true_bin.append(1 if y_true != NO_PARASITE_CLASS else 0)
        y_scores_bin.append(y_score if y_pred == y_true else 0)

    precision, recall, _ = precision_recall_curve(y_true_bin, y_scores_bin)
    average_precision = average_precision_score(y_true_bin, y_scores_bin)

    plt.plot(recall, precision, lw=2, label=f"Overall (AP={average_precision:.2f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Overall Precision-Recall Curve")
    plt.legend(loc="best")
    plt.savefig(os.path.join(output_dir, "pr_curve_overall.png"))
    plt.close()


def plot_confusion_matrix(y_true: List[int], y_pred: List[int], output_dir: str):
    """Plot confusion matrix"""
    # Collect all unique class labels from ground truth and predictions
    all_classes = set(y_true)
    all_classes.update(y_pred)

    num_classes = len(all_classes)

    # Initialize confusion matrix
    conf_matrix = np.zeros((num_classes, num_classes))

    for y_true, y_pred in zip(y_true, y_pred):
        conf_matrix[y_true][y_pred] += 1

    # Use class names for the confusion matrix labels
    class_names = [IDX_TO_CLASS[class_id] for class_id in sorted(all_classes)]

    plt.figure(figsize=(18, 18))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="g",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()


def plot_precision_recall_f1_confidence_curve(
    y_trues: List[int], y_preds: List[int], y_scores: List[float], output_dir: str
):
    """Plot precision-recall-f1-confidence curve"""

    # Overall precision-recall-confidence curve
    plt.figure(figsize=(10, 8))
    y_true_bin = [
        1 if y_true == y_pred else 0
        for i, (y_true, y_pred) in enumerate(zip(y_trues, y_preds))
    ]
    y_scores_bin = y_scores
    precision, recall, thresholds = precision_recall_curve(y_true_bin, y_scores_bin)
    average_precision = average_precision_score(y_true_bin, y_scores_bin)

    plt.plot(thresholds, precision[:-1], lw=2, label="Precision")
    plt.plot(thresholds, recall[:-1], lw=2, label="Recall")
    f1_scores = (
        2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
    )
    plt.plot(thresholds, f1_scores, lw=2, label="F1 Score")

    plt.xlabel("Confidence Threshold")
    plt.ylabel("Score")
    plt.title(f"Overall Precision-Recall-F1 Curve (AP={average_precision:.2f})")
    plt.legend(loc="best")
    plt.savefig(os.path.join(output_dir, "prf_curve_overall.png"))
    plt.close()

    # Per class precision-recall-confidence curves
    unique_classes = sorted(set(y_trues))
    num_classes = len(unique_classes)
    fig, axes = plt.subplots(
        nrows=(num_classes + 1) // 2,
        ncols=2,
        figsize=(20, 5 * ((num_classes + 1) // 2)),
    )

    for class_id in unique_classes:
        y_true_class = []
        y_scores_class = []
        for y_true, y_pred, score in zip(y_trues, y_preds, y_scores):
            if y_true == class_id:
                y_true_class.append(1)
                y_scores_class.append(score)
            elif y_true == NO_PARASITE_CLASS:
                y_true_class.append(0)
                y_scores_class.append(score)

        precision, recall, thresholds = precision_recall_curve(
            y_true_class, y_scores_class
        )
        average_precision = average_precision_score(y_true_class, y_scores_class)

        ax = axes[class_id // 2, class_id % 2]
        ax.plot(thresholds, precision[:-1], lw=2, label="Precision")
        ax.plot(thresholds, recall[:-1], lw=2, label="Recall")
        f1_scores = (
            2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
        )
        ax.plot(thresholds, f1_scores, lw=2, label="F1 Score")

        ax.set_xlabel("Confidence Threshold")
        ax.set_ylabel("Score")
        ax.set_title(
            f"Precision-Recall-F1 Curve for {IDX_TO_CLASS[class_id]} (AP={average_precision:.2f})"
        )
        ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "prf_curve_per_class.png"))
    plt.close()


def validate_predictions(
    gt_labels: List[ImagePrediction],
    predictions: List[ImagePrediction],
    iou_threshold: float,
    min_confidence: float,
    output_dir: str,
):
    """Run validation and generate plots"""
    os.makedirs(output_dir, exist_ok=True)

    y_true, y_pred, y_scores = match_predictions(
        gt_labels, predictions, iou_threshold, min_confidence
    )

    plot_precision_recall_curve(y_true, y_pred, y_scores, output_dir)
    plot_confusion_matrix(y_true, y_pred, output_dir)
    plot_precision_recall_f1_confidence_curve(y_true, y_pred, y_scores, output_dir)
