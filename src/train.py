"""
Step 4 of the build order: get ONE backbone (Xception, the paper's best
performer) fully working end-to-end before generalizing to the other five.

Usage:
    python src/train.py --backbone xception
"""
import os
import json
import argparse
import numpy as np
import tensorflow as tf

from config import (IMAGE_SIZE, BATCH_SIZE, EPOCHS, CLASS_NAMES,
                     RESULTS_TABLES_DIR, RESULTS_PLOTS_DIR, MODELS_DIR, SEED)
from dataset import load_processed_arrays, make_tf_dataset
from models import build_model
from metrics_utils import compute_full_report, save_confusion_matrix


def train_and_evaluate(backbone_name: str, epochs: int = EPOCHS, verbose: int = 1):
    tf.keras.utils.set_random_seed(SEED)  # python/numpy/TF seeds
    os.makedirs(RESULTS_TABLES_DIR, exist_ok=True)
    os.makedirs(RESULTS_PLOTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    (x_train, y_train), (x_test, y_test) = load_processed_arrays()

    train_ds = make_tf_dataset(x_train, y_train, training=True, batch_size=BATCH_SIZE)
    # Deterministic test set (no random augmentation) — see dataset.py docstring.
    test_ds = make_tf_dataset(x_test, y_test, training=False, batch_size=BATCH_SIZE, augment_test=False)

    model = build_model(backbone_name, image_size=IMAGE_SIZE, num_classes=len(CLASS_NAMES))
    model.summary()

    model.fit(train_ds, epochs=epochs, verbose=verbose)

    train_loss, train_acc = model.evaluate(train_ds, verbose=0)
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)

    y_pred = np.argmax(model.predict(test_ds, verbose=0), axis=1)

    per_class, summary = compute_full_report(
        y_test, y_pred, CLASS_NAMES, train_acc, test_acc, train_loss, test_loss,
        n_train=len(y_train), n_test=len(y_test),
    )

    save_confusion_matrix(
        y_test, y_pred, CLASS_NAMES,
        os.path.join(RESULTS_PLOTS_DIR, f"{backbone_name}_confusion_matrix.png"),
        title=f"{backbone_name} confusion matrix",
    )

    report_path = os.path.join(RESULTS_TABLES_DIR, f"{backbone_name}_report.json")
    with open(report_path, "w") as f:
        json.dump({"per_class": per_class, "summary": summary}, f, indent=2)

    model.save(os.path.join(MODELS_DIR, f"{backbone_name}.keras"))

    print(f"\n== {backbone_name} summary ==")
    print(json.dumps(summary, indent=2))
    print(f"Saved report to {report_path}")

    return summary, per_class


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="xception",
                         help="one of: " + ", ".join(
                             ["xception", "mobilenetv2", "inceptionv3", "resnet50", "vgg16", "densenet121"]))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()
    train_and_evaluate(args.backbone, epochs=args.epochs)
