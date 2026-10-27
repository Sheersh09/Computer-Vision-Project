"""
Step 3 of the build order: eyeball a handful of preprocessed images per
class BEFORE touching model code. A broken crop (e.g. cropping to a corner,
or an empty box) is invisible in metrics until much later — catch it here.

Usage:
    python src/check_preprocessing.py
"""
import os
import random
import matplotlib.pyplot as plt

from config import TRAIN_DIR, CLASS_NAMES, IMAGE_SIZE, RESULTS_PLOTS_DIR, SEED
from data_preprocessing import preprocess_image

random.seed(SEED)


def main(n_per_class: int = 4):
    os.makedirs(RESULTS_PLOTS_DIR, exist_ok=True)
    fig, axes = plt.subplots(
        len(CLASS_NAMES), n_per_class,
        figsize=(2.5 * n_per_class, 2.5 * len(CLASS_NAMES)),
    )
    for row, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(TRAIN_DIR, class_name)
        files = os.listdir(class_dir)
        sample = random.sample(files, min(n_per_class, len(files)))
        for col in range(n_per_class):
            ax = axes[row][col]
            if col < len(sample):
                img = preprocess_image(os.path.join(class_dir, sample[col]), IMAGE_SIZE)
                ax.imshow(img)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(class_name)
                ax.set_title(class_name, loc="left")

    out_path = os.path.join(RESULTS_PLOTS_DIR, "preprocessing_sanity_check.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    print("Open it and confirm: brain is centered/filling the frame (not a black "
          "corner or empty crop) for every class before running prepare_dataset.py "
          "on the full set.")


if __name__ == "__main__":
    main()
