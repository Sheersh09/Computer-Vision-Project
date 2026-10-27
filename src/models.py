"""Model architecture from Table 1 of the paper on top of any of the
six pretrained backbones:

    Input(128,128,3) -> BaseModel -> Flatten -> Dropout(0.3)
    -> Dense(128, relu) -> Dropout(0.2) -> Dense(4, softmax)

Design note / assumption: the paper doesn't explicitly say whether the base model's ImageNet weights are frozen or fine-tuned (unfrozen). Given the
5-epoch training run still reaches ~97-99% *training* accuracy across every backbone (Table 8), the base must be trainable — a frozen feature extractor 
feeding only a small dense head would not fit that fast on a 4-class medical task. 
Default here is `trainable_base=True` (full fine-tuning). Set it to False if you want the frozen-feature-extractor variant instead.
"""
import tensorflow as tf
from tensorflow.keras import layers, models, applications

from config import IMAGE_SIZE, DROPOUT_1, DROPOUT_2, DENSE_UNITS, LEARNING_RATE

BACKBONE_CONSTRUCTORS = {
    "xception": applications.Xception,
    "mobilenetv2": applications.MobileNetV2,
    "inceptionv3": applications.InceptionV3,
    "resnet50": applications.ResNet50,
    "vgg16": applications.VGG16,
    "densenet121": applications.DenseNet121,
}


def build_model(backbone_name: str, image_size: int = IMAGE_SIZE, num_classes: int = 4,
                 trainable_base: bool = True) -> tf.keras.Model:
    key = backbone_name.lower()
    if key not in BACKBONE_CONSTRUCTORS:
        raise ValueError(f"Unknown backbone '{backbone_name}'. Choose from {list(BACKBONE_CONSTRUCTORS)}")

    base = BACKBONE_CONSTRUCTORS[key](
        include_top=False, weights="imagenet", input_shape=(image_size, image_size, 3)
    )
    base.trainable = trainable_base

    inputs = layers.Input(shape=(image_size, image_size, 3))
    # No explicit `training=` here: Keras propagates the right mode (train vs
    # evaluate/predict) so BatchNorm uses moving averages at inference time.
    x = base(inputs)
    x = layers.Flatten()(x)
    x = layers.Dropout(DROPOUT_1)(x)
    x = layers.Dense(DENSE_UNITS, activation="relu")(x)
    x = layers.Dropout(DROPOUT_2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name=f"{key}_brain_tumor")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    return model
