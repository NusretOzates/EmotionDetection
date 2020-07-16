import os

import tensorflow as tf
from tensorflow.keras.layers import *


class Resnet50:

    def __init__(self, target_size):
        DROPOUT_RATE = 0.5
        self.checkpoint_path = "model_weights/resnet_50__training_1/cp.ckpt"
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)

        self.net = tf.keras.applications.ResNet50V2(input_shape=(target_size, target_size, 3),
                                                    include_top=False,
                                                    weights='imagenet'
                                                    )
        self.net.trainable = True
        # Fine tune from this layer onwards
        fine_tune_at = 60

        # Freeze all the layers before the `fine_tune_at` layer
        for layer in self.net.layers[:fine_tune_at]:
            layer.trainable = False

        self.model = tf.keras.Sequential([
            self.net,
            Flatten(),
            tf.keras.layers.Dense(7, activation='softmax')
        ])

       # self.model.load_weights(self.checkpoint_path)

        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adamax',
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