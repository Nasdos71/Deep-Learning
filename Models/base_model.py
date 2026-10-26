"""
Base Model Definition & Evaluation Utilities
Provides a standardized interface for all models in the Models/ directory.
"""

import os
import shutil
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
import torch.nn as nn


class BaseModel:
    """Standard interface that all model plugins in Models/ implement."""
    MODEL_NAME = "Base Model"
    DESCRIPTION = "Abstract Model"

    def train(self, data: dict, output_dir: str, **kwargs) -> float:
        """
        Trains the model on `data` dictionary and saves artifacts into `output_dir`.
        Returns the final test accuracy (float between 0 and 1).
        """
        raise NotImplementedError("Each model must implement the train method.")


def save_evaluation_results(output_dir, dataset_name, model_name, y_true, y_pred, class_names, model_obj, save_filename):
    """
    Computes metrics, prints classification report, plots confusion matrix,
    and persists model checkpoint and report to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    acc = accuracy_score(y_true, y_pred)

    print("\n" + "=" * 60, flush=True)
    print(f"  [RESULT] {model_name.upper()} Accuracy on {dataset_name.upper()}: {acc * 100:.2f}%", flush=True)
    print("=" * 60, flush=True)

    report_str = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    print("\nClassification Report:\n", flush=True)
    print(report_str, flush=True)

    # 1. Save classification report text
    report_path = os.path.join(output_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Dataset: {dataset_name}\nModel: {model_name}\nAccuracy: {acc * 100:.2f}%\n\n")
        f.write(report_str)

    # 2. Plot & Save Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    figsize = (6, 5) if len(class_names) <= 10 else (8, 7)
    plt.figure(figsize=figsize)
    annotate = len(class_names) <= 10
    sns.heatmap(cm, annot=annotate, fmt='d' if annotate else '', cmap='Blues',
                xticklabels=class_names if annotate else False,
                yticklabels=class_names if annotate else False)
    plt.title(f"{model_name.upper()} on {dataset_name.capitalize()} (Acc: {acc * 100:.1f}%)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=180)

    # Copy to latest preview for GUI
    root_output = os.path.abspath(os.path.join(output_dir, "..", ".."))
    latest_cm = os.path.join(root_output, "last_run_cm.png")
    try:
        shutil.copyfile(cm_path, latest_cm)
    except Exception:
        pass
    plt.close()

    # 3. Save Model Weights
    model_save_path = os.path.join(output_dir, save_filename)
    if isinstance(model_obj, nn.Module):
        torch.save(model_obj.state_dict(), model_save_path)
    else:
        joblib.dump(model_obj, model_save_path)

    print(f"[*] Saved model to: {model_save_path}", flush=True)
    print(f"[*] Saved confusion matrix plot to: {cm_path}", flush=True)
    print(f"[*] Saved classification report to: {report_path}", flush=True)
    print("[DONE] Training finished successfully!", flush=True)

    return acc
