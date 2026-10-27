# Brain Tumor MRI Classification — Reproduction

Reproduces Disci, Gurcan & Soylu, *"Advanced Brain Tumor Classification in MR Images Using Transfer Learning and Pre-Trained Deep CNN Models"* (Cancers,2025) on the Kaggle **Brain Tumor MRI Dataset**
(`masoudnickparvar/brain-tumor-mri-dataset`, 7,023 images, 4 classes:glioma, meningioma, notumor, pituitary; 5,712 train / 1,311 test).

## Ambiguities in the paper and the assumptions taken

The paper's Methods section doesn't fully pin these down. Each is also flagged as a comment at the exact line of code it affects.

1. **Does the model see the binary threshold mask, or the original grayscale intensities?** Section 2.2 describes binarizing the image, but also says the input is "RGB" and that intensity variation (not color) carries the discriminative signal — a pure black/white mask would erase exactly that signal. 
**Assumption:** the threshold/contour step is used only to *find* the brain's bounding box; the original grayscale pixels (not the mask) are cropped, resized, and replicated to 3 channels. `data_preprocessing.py` exposes `use_binary_mask=True` if you want the literal mask instead.


2. **Is the pretrained backbone frozen or fully fine-tuned?** Not stated.
   **Assumption:** fully fine-tuned (`trainable_base=True` in `models.py`) — a frozen feature extractor plus a small dense head is very unlikely to reach the ~97–99% *training* accuracy the paper reports after only 5
   epochs (Table 8).


3. **Is the test set really augmented?** Section 2.3 states brightness/contrast augmentation was applied "to both the training and testing datasets," which is non-standard (it makes test accuracy stochastic between runs). 
 **Assumption:** the authors most likely meant identical *preprocessing* (resize, normalize) on both splits, not random
brightness/contrast on test images. Default: augment train only, test setis deterministic. Pass `augment_test=True` to `dataset.make_tf_dataset` to follow the paper's wording literally.


4. **"Macro Accuracy Avg" / "Weighted Accuracy Avg"** in Table 8 aren't standard sklearn metrics. 
**Resolved, not assumed:** reverse-engineered from Table 2's numbers and confirmed to reproduce them exactly:
   `macro = (train_acc + test_acc) / 2`,
   `weighted = (train_acc·n_train + test_acc·n_test) / (n_train + n_test)`.
   See the docstring in `metrics_utils.py` for the worked check.


5. **Exact crop/threshold parameters** (Gaussian kernel size, threshold value, erosion/dilation iteration counts) aren't given at all — the paper only names the technique. 
**Assumption:** standard values from the well-known "crop brain contour" recipe this section is clearly based on (5×5 blur, threshold=45, 2 erode/dilate iterations). This is the parameter set most likely to need retuning if `check_preprocessing.py`'s output looks wrong on your machine.


6. **Which dataset release?** The paper used the 7,023-image release (5,712 train / 1,311 test; class counts are unbalanced despite the paper calling it balanced). Kaggle now serves a newer "Version 2" (7,200 images, 1,400/400 per class, duplicates and train/test overlap removed).
**Decision: this reproduction uses the paper's 7,023-image release only.** `config.py` defaults to `DATASET_VERSION = "v1"`, so nothing needs setting. (The v2 option is still in the code for later; you can ignore it.)

Given (1)-(3) are genuine judgment calls, don't expect to hit the paper's exact 98.73% figure for Xception — aim to reproduce the *pattern* instead (Xception/MobileNetV2/InceptionV3 near the top, ResNet50 showing a bigger train/test gap, Glioma/Meningioma recall being the weakest spots).


## 1. Environment setup

### Local machine (venv)

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## 2. Get the dataset (paper's 7,023-image release)

Kaggle's default download is now the newer 7,200-image release, so you need
the **older version** of `masoudnickparvar/brain-tumor-mri-dataset`. Whichever
route you use, extract so the folders sit at `data/raw/v1/Training/...` and
`data/raw/v1/Testing/...`, then verify the counts (below) before going on.

**Route A: website (simplest).** Open the dataset page on Kaggle, go to the
Data tab, use the version selector to pick the older release, and download.

**Route B: REST API.** The Kaggle download endpoint accepts a
`datasetVersionNumber` query parameter. Create a token first (kaggle.com ->
profile -> Account -> "Create New Token"; the downloaded `kaggle.json`
holds your username and key), then:

```bash
mkdir -p data/raw/v1
curl -L -u YOUR_KAGGLE_USERNAME:YOUR_KAGGLE_KEY \
  -o data/raw/v1/dataset.zip \
  "https://www.kaggle.com/api/v1/datasets/download/masoudnickparvar/brain-tumor-mri-dataset?datasetVersionNumber=1"
unzip data/raw/v1/dataset.zip -d data/raw/v1
```


Expected: Training glioma 1321, meningioma 1339, notumor 1595, pituitary 1457
(total 5,712); Testing glioma 300, meningioma 306, notumor 405, pituitary 300
(total 1,311). If you see 1400 per training class and 400 per test class, that
is the newer release, not the paper's. If the archive extracts with an extra
nesting level, move the folders up or edit `TRAIN_DIR`/`TEST_DIR` in
`src/config.py`.


```powershell
$env:BRAIN_MRI_RAW_DIR = "D:\sem5\CV"      # PowerShell (VS Code default on Windows)
set BRAIN_MRI_RAW_DIR=D:\sem5\CV           # cmd.exe
export BRAIN_MRI_RAW_DIR=/path/to/data      # macOS / Linux / WSL
```

It applies only to the current terminal, so set it again in each new one.
`run_all_models.py` skips backbones that already have a report; add `--force`
to retrain them.

Every output folder below (`data/processed/`, `results/tables/`,
`results/plots/`, `saved_models/`) is created automatically by the scripts
that write to it — nothing to set up manually beyond `data/raw/v1/` above.

## 3. Build order (run in order, from `src/`)

```bash
cd src

# Steps 2-3: preprocess a few sample images and LOOK at the result first.
python check_preprocessing.py
# -> ../results/v1/plots/preprocessing_sanity_check.png
# Every tile should show a brain filling most of the frame. If it doesn't,
# fix the crop parameters (see ambiguity #5 above) before continuing.

# Preprocess the FULL dataset once and cache it (a few minutes for 7023 images).
python prepare_dataset.py
# -> ../data/processed/v1/{x_train,y_train,x_test,y_test}.npy
# Check the printed class counts match Section 2.1 of the paper
# (train 1321/1339/1595/1457, test 300/306/405/300).

# Step 4: get ONE backbone working end-to-end (Xception, the paper's best performer).
python train.py --backbone xception
# -> ../saved_models/v1/xception.keras
# -> ../results/v1/tables/xception_report.json
# -> ../results/v1/plots/xception_confusion_matrix.png

# Step 5: generalize to the remaining five backbones + Table 8 + Figure 2.
python run_all_models.py
# -> ../results/v1/tables/model_comparison.csv        (Table 8 equivalent)
# -> ../results/v1/plots/model_comparison_bar.png     (Figure 2 equivalent)
# -> one *_report.json / *_confusion_matrix.png per backbone
```

## 4. File reference

| File | Purpose |
|---|---|
| `config.py` | All paths + hyperparameters (image size, batch size, epochs, LR, dropout, class names). |
| `data_preprocessing.py` | Per-image pipeline: grayscale → blur → threshold → contour crop → resize. |
| `check_preprocessing.py` | Visual sanity check — run before processing the full dataset. |
| `prepare_dataset.py` | Applies the pipeline to all 7,023 images once, caches to `.npy`. |
| `dataset.py` | `tf.data` pipeline: shuffling, brightness/contrast augmentation, normalization. |
| `models.py` | Builds the Table-1 architecture on top of any of the six backbones. |
| `metrics_utils.py` | Per-class precision/recall/F1, confusion matrices, macro/weighted accuracy-avg. |
| `train.py` | Train + evaluate a single backbone end-to-end. |
| `run_all_models.py` | Loops `train.py` over all six backbones, builds the comparison table + bar chart. |

No other files are needed to run the pipeline — everything else (data,
results, saved model files) is generated at runtime.
