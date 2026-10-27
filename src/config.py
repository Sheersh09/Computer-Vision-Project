"""
Central configuration for the brain tumor classification pipeline.
Values here are taken directly from the paper's Methods section (2.1-2.5)
unless a comment says otherwise.
"""
import os

# ---------------------------------------------------------------------------
# Paths (matches: data/raw/, data/processed/, src/, results/tables/, results/plots/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Which release of the Kaggle dataset to use:
#   "v1" = the release the paper used (7,023 images; 5,712 train / 1,311 test; unbalanced)
#   "v2" = current Kaggle default (7,200 images; 1,400 train / 400 test per class;
#          duplicates removed, train/test overlap removed)
# Override without editing: BRAIN_MRI_VERSION=v2 python train.py ...
DATASET_VERSION = os.environ.get("BRAIN_MRI_VERSION", "v1")
assert DATASET_VERSION in ("v1", "v2"), "DATASET_VERSION must be 'v1' or 'v2'"


def _versioned(*parts):
    """v1 uses the plain folders (data/raw, results/tables, ...); v2 adds a /v2 subfolder."""
    p = os.path.join(PROJECT_ROOT, *parts)
    return p if DATASET_VERSION == "v1" else os.path.join(p, DATASET_VERSION)


# Folder that directly contains Training/ and Testing/. Override with the
# BRAIN_MRI_RAW_DIR environment variable (e.g. D:\sem5\CV) to avoid copying data.
RAW_DIR = os.environ.get("BRAIN_MRI_RAW_DIR", _versioned("data", "raw"))

# The Kaggle download extracts to a folder containing Training/ and Testing/.
# If your extracted folder has a different top-level name, point these two
# at it directly (e.g. RAW_DIR + "/brain-tumor-mri-dataset").
TRAIN_DIR = os.path.join(RAW_DIR, "Training")
TEST_DIR = os.path.join(RAW_DIR, "Testing")

PROCESSED_DIR = _versioned("data", "processed")
RESULTS_TABLES_DIR = _versioned("results", "tables")
RESULTS_PLOTS_DIR = _versioned("results", "plots")
MODELS_DIR = _versioned("saved_models")

# ---------------------------------------------------------------------------
# Dataset (Section 2.1) — alphabetical order matches Keras' default class
# indexing (flow_from_directory / listdir sorted), and matches the class
# order used in the paper's confusion matrices.
# ---------------------------------------------------------------------------
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
_EXPECTED_COUNTS = {
    "v1": (
        {"glioma": 1321, "meningioma": 1339, "notumor": 1595, "pituitary": 1457},  # 5,712
        {"glioma": 300, "meningioma": 306, "notumor": 405, "pituitary": 300},      # 1,311
    ),
    "v2": (
        {name: 1400 for name in CLASS_NAMES},  # 5,600
        {name: 400 for name in CLASS_NAMES},   # 1,600
    ),
}
EXPECTED_TRAIN_COUNTS, EXPECTED_TEST_COUNTS = _EXPECTED_COUNTS[DATASET_VERSION]

# ---------------------------------------------------------------------------
# Preprocessing / augmentation (Sections 2.2-2.3)
# ---------------------------------------------------------------------------
IMAGE_SIZE = 128          # final resize target (128 x 128)
CHANNELS = 3              # model input is RGB-shaped, see data_preprocessing.py
BRIGHTNESS_RANGE = (0.8, 1.2)
CONTRAST_RANGE = (0.8, 1.2)

# ---------------------------------------------------------------------------
# Model / training (Section 2.5, Table 1)
# ---------------------------------------------------------------------------
BATCH_SIZE = 20
EPOCHS = 5
LEARNING_RATE = 1e-4
DROPOUT_1 = 0.3
DROPOUT_2 = 0.2
DENSE_UNITS = 128

BACKBONES = ["xception", "mobilenetv2", "inceptionv3", "resnet50", "vgg16", "densenet121"]

SEED = 42