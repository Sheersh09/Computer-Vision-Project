"""
Reproduces the exact metrics reported in Tables 2-8 of the paper.

The paper reports "Macro Accuracy Avg" and "Weighted Accuracy Avg" with
support=7023 (the full dataset), which aren't standard sklearn outputs.
Reverse-engineering Table 2 (Xception: train_acc=0.9952, test_acc=0.9527,
n_train=5712, n_test=1311) shows:

    macro_accuracy_avg    = (train_acc + test_acc) / 2
                           = (0.9952 + 0.9527) / 2 = 0.97395 -> rounds to 0.9740  (matches)

    weighted_accuracy_avg = (train_acc * n_train + test_acc * n_test) / (n_train + n_test)
                           = (0.9952*5712 + 0.9527*1311) / 7023 = 0.98724 -> rounds to 0.9873 (matches)

Both formulas were checked against Table 2 and reproduce the paper's
reported values, so they're implemented here as-is.
"""
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def compute_full_report(y_true, y_pred, class_names, train_acc, test_acc,
                         train_loss, test_loss, n_train, n_test):
    per_class = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    n_total = n_train + n_test
    summary = {
        "training_accuracy": float(train_acc),
        "testing_accuracy": float(test_acc),
        "macro_accuracy_avg": float((train_acc + test_acc) / 2),
        "weighted_accuracy_avg": float((train_acc * n_train + test_acc * n_test) / n_total),
        "training_loss": float(train_loss),
        "testing_loss": float(test_loss),
    }
    return per_class, summary


def save_confusion_matrix(y_true, y_pred, class_names, out_path, title=None):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return cm
