from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers.experimental.preprocessing import Rescaling, RandomFlip, RandomRotation, RandomTranslation, \
    RandomContrast

def image_augmentation_model():
    img_augmentation = Sequential(
        [
            Rescaling(scale=1.0 / 255),
            RandomRotation(factor=0.15),
            RandomTranslation(height_factor=0.1, width_factor=0.1),
            RandomFlip(),
            RandomContrast(factor=0.1),
        ],
        name="img_augmentation",
    )

    return img_augmentation


def freeze_first_n_percent(model: Model, n: int):
    # Fine tune from this layer onwards
    fine_tune_at = len(model.layers) * n // 100
    print(fine_tune_at)
    # Freeze all the layers before the `fine_tune_at` layer
    for layer in model.layers[:fine_tune_at]:
        layer.trainable = False
    for layer in model.layers[fine_tune_at:]:
        layer.trainable = True

    return model
