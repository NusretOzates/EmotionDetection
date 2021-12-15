from tensorflow import data
import tensorflow as tf
from tensorflow.keras.models import Model
from sklearn.utils import class_weight

import numpy as np


class BaseModel:
    checkpoint_path = ""
    model: Model = None

    def train(self, epochs: int, train: data.Dataset, validation: data.Dataset, name: str):

        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
                                                         save_weights_only=True,
                                                         save_best_only=True,
                                                         verbose=0)

        log_dir = f"logs/fit/{name}"
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        history = self.model.fit(x=train,
                                 validation_data=validation,
                                 epochs=epochs,
                                 verbose=1,
                                 callbacks=[cp_callback, tensorboard_callback],
                                 )
        return history
