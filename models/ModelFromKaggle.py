import os

import tensorflow as tf
from tensorflow.keras.layers import *


class KaggleModel:

    def __init__(self):
        self.checkpoint_path = "kaggle_training_1/cp.ckpt"
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)
        self.model = tf.keras.Sequential([

            Conv2D(32, (3, 3), activation='relu', input_shape=(160, 160, 3)),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(64, (3, 3), activation='relu', padding='same'),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(96, (3, 3), activation='relu', padding='same', dilation_rate=(2, 2)),
            Conv2D(96, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(128, (3, 3), activation='relu', padding='same', dilation_rate=(2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),

            Flatten(),

            Dense(64, activation='relu'),
            Dropout(0.4),
            Dense(7, activation='softmax')

        ])

        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'],
                           )

    def train(self, epochs, train_dataset, validation_dataset):
        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
                                                         save_weights_only=True,
                                                         save_best_only=True,
                                                         verbose=0)
        history = self.model.fit(x=train_dataset,
                                 validation_data=validation_dataset,
                                 epochs=epochs,
                                 verbose=1,
                                 callbacks=[cp_callback]
                                 )
        return history
