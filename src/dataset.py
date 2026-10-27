"""
Builds tf.data.Dataset objects from the cached, preprocessed .npy arrays.

Implements Section 2.3 exactly:
  - random brightness factor in [0.8, 1.2]
  - random contrast factor in [0.8, 1.2]
  - normalize by dividing by 255 (done AFTER the brightness/contrast step)

The paper's text says augmentation/preprocessing was applied to both the
training and testing datasets (Section 2.3). Random brightness/contrast on
test images would make reported accuracy vary run to run, so the default
here is: augment TRAIN only; the test set gets only the deterministic
normalization (/255). Pass `augment_test=True` to make_tf_dataset to follow
the paper's wording literally. Flagged again in README.md.
"""
import numpy as np
import tensorflow as tf

from config import PROCESSED_DIR, BRIGHTNESS_RANGE, CONTRAST_RANGE
import os


def load_processed_arrays():
    x_train = np.load(os.path.join(PROCESSED_DIR, "x_train.npy"))
    y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
    x_test = np.load(os.path.join(PROCESSED_DIR, "x_test.npy"))
    y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))
    return (x_train, y_train), (x_test, y_test)


def _random_brightness_contrast(image: tf.Tensor) -> tf.Tensor:
    image = tf.cast(image, tf.float32)

    b_factor = tf.random.uniform([], BRIGHTNESS_RANGE[0], BRIGHTNESS_RANGE[1])
    image = image * b_factor

    c_factor = tf.random.uniform([], CONTRAST_RANGE[0], CONTRAST_RANGE[1])
    mean = tf.reduce_mean(image, axis=[0, 1], keepdims=True)
    image = (image - mean) * c_factor + mean

    return tf.clip_by_value(image, 0.0, 255.0)


def _augment_and_normalize(image, label):
    image = _random_brightness_contrast(image)
    image = image / 255.0
    return image, label


def _normalize_only(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def make_tf_dataset(images, labels, training: bool, batch_size: int, augment_test: bool = False):
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    if training:
        ds = ds.shuffle(buffer_size=len(images), seed=42, reshuffle_each_iteration=True)
        ds = ds.map(_augment_and_normalize, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        map_fn = _augment_and_normalize if augment_test else _normalize_only
        ds = ds.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
