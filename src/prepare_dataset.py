"""
Run once, before any training: preprocesses every raw image and caches the
result as .npy arrays under data/processed/. The contour-based crop is CPU
work (OpenCV), not something you want to redo every epoch, so we do it a
single time here.

Usage:
    python src/prepare_dataset.py
"""
import os
import numpy as np
from tqdm import tqdm

from config import TRAIN_DIR, TEST_DIR, CLASS_NAMES, IMAGE_SIZE, PROCESSED_DIR, \
    EXPECTED_TRAIN_COUNTS, EXPECTED_TEST_COUNTS
from data_preprocessing import preprocess_image


def build_split(split_dir: str):
    images, labels = [], []
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(
                f"Expected class folder not found: {class_dir}\n"
                f"Check that the Kaggle archive was extracted so that "
                f"{split_dir} directly contains {CLASS_NAMES}."
            )
        filenames = sorted(os.listdir(class_dir))
        for fname in tqdm(filenames, desc=f"{os.path.basename(split_dir)}/{class_name}"):
            fpath = os.path.join(class_dir, fname)
            try:
                img = preprocess_image(fpath, IMAGE_SIZE)
            except Exception as exc:  # corrupt/non-image file, skip and report
                print(f"  [skip] {fpath}: {exc}")
                continue
            images.append(img)
            labels.append(class_idx)
    return np.array(images, dtype=np.uint8), np.array(labels, dtype=np.int64)


def _report_counts(name, labels, expected):
    counts = {CLASS_NAMES[i]: int((labels == i).sum()) for i in range(len(CLASS_NAMES))}
    print(f"{name} counts: {counts}")
    mismatches = {k: (counts[k], expected[k]) for k in counts if counts[k] != expected[k]}
    if mismatches:
        print(f"  [warning] counts differ from the paper's Table stats: {mismatches}")
    else:
        print(f"  matches paper's reported {name} distribution.")


if __name__ == "__main__":
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("Processing training split...")
    x_train, y_train = build_split(TRAIN_DIR)
    print("Processing testing split...")
    x_test, y_test = build_split(TEST_DIR)

    np.save(os.path.join(PROCESSED_DIR, "x_train.npy"), x_train)
    np.save(os.path.join(PROCESSED_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(PROCESSED_DIR, "x_test.npy"), x_test)
    np.save(os.path.join(PROCESSED_DIR, "y_test.npy"), y_test)

    print(f"\nSaved to {PROCESSED_DIR}")
    print(f"Train shape: {x_train.shape}  Test shape: {x_test.shape}")
    _report_counts("Train", y_train, EXPECTED_TRAIN_COUNTS)
    _report_counts("Test", y_test, EXPECTED_TEST_COUNTS)
