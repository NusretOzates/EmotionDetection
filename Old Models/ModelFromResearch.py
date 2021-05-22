import os

import tensorflow as tf
from tensorflow.keras.layers import *


class ResearchModel:

    def __init__(self):
        self.checkpoint_path = "research_training_1/cp.ckpt"
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)
        self.model = tf.keras.Sequential([
            Conv2D(32, (5, 5), activation='relu', input_shape=(197, 197, 3)),
            BatchNormalization(),
            MaxPooling2D(pool_size=(3, 3), strides=2),

            Conv2D(32, (4, 4), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(3, 3), strides=2),

            Conv2D(64, (5, 5), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(3, 3), strides=2),

            Flatten(),
            Dense(1024, activation='relu'),
            BatchNormalization(),

            Dense(1024, activation='relu'),
            Dropout(0.3),
            BatchNormalization(),
            Dense(7, activation='softmax')
        ])

        #self.model.load_weights(self.checkpoint_path)

        self.model.compile(loss=tf.keras.losses.CategoricalCrossentropy.__name__,
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
