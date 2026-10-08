"""
evaluation.py - shared helpers to score models and draw report charts.

Metric cheat-sheet (attack = "positive"):
  Accuracy  - % of all rows labelled correctly
  Precision - when we shout "attack", how often it really is one
  Recall    - of all real attacks, how many we caught   (most important for an IDS)
  F1        - balance of precision and recall
  FPR       - % of NORMAL traffic wrongly flagged as attack (false alarms)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)

from config import CLASSES, RESULT_DIR


def binary_metrics(y_true_attack, y_pred_attack):
    """Attack-vs-normal metrics. Inputs are boolean arrays (True = attack)."""
    t, p = np.asarray(y_true_attack, bool), np.asarray(y_pred_attack, bool)
    fp = np.sum(p & ~t)
    tn = np.sum(~p & ~t)
    return {
        "accuracy": accuracy_score(t, p),
        "precision": precision_score(t, p, zero_division=0),
        "recall": recall_score(t, p, zero_division=0),
        "f1": f1_score(t, p, zero_division=0),
        "fpr": fp / max(fp + tn, 1),
    }


def multiclass_metrics(y_true, y_pred, name):
    """5-class metrics + saves the confusion matrix and a text report."""
    report = classification_report(y_true, y_pred, labels=range(len(CLASSES)),
                                   target_names=CLASSES, digits=4, zero_division=0)
    (RESULT_DIR / f"report_{name}.txt").write_text(report)
    plot_confusion(y_true, y_pred, f"Confusion matrix - {name}", RESULT_DIR / f"cm_{name}.png")
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    out.update({f"bin_{k}": v for k, v in binary_metrics(y_true != 0, y_pred != 0).items()})
    return out


def plot_confusion(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))
    row_pct = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(row_pct, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)), CLASSES)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{cm[i, j]:,}\n{row_pct[i, j]:.0%}", ha="center", va="center",
                    fontsize=8, color="white" if row_pct[i, j] > 0.5 else "#0b0b0b")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, label="Share of actual class")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_history(history, title, path, ylabel="Loss"):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(history["train_loss"]) + 1), history["train_loss"],
            color="#2a78d6", lw=2, label="Training")
    if history.get("val_loss"):
        ax.plot(range(1, len(history["val_loss"]) + 1), history["val_loss"],
                color="#eb6834", lw=2, label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_metric_bars(df, metrics, title, path):
    """Grouped bar chart: one group per metric, one bar per model (rows of df)."""
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    n = len(df)
    width = 0.8 / n
    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(metrics)), 4.5))
    for i, (name, row) in enumerate(df.iterrows()):
        bars = ax.bar(x + i * width - 0.4 + width / 2, [row[m] for m in metrics], width * 0.92,
                      label=name, color=palette[i % len(palette)])
        ax.bar_label(bars, labels=[f"{row[m]:.2f}" for m in metrics], fontsize=7, padding=1)
    ax.set_xticks(x, [m.replace("_", " ").upper() if len(m) <= 4 else m.replace("_", " ").title()
                      for m in metrics])
    ax.set_ylim(0, 1.08)
    ax.set_title(title)
    ax.legend(frameon=False, ncol=n, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
