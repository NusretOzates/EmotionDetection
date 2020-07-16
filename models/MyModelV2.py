import os

import tensorflow as tf
from tensorflow.keras.layers import *


class MyModelV2:

    def __init__(self):
        self.checkpoint_path = "mymodelV2_training_1/cp.ckpt"
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)
        self.model = tf.keras.Sequential([
            Conv2D(64, (3, 3), activation='elu', padding='same', input_shape=(160, 160, 3)),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='elu', padding='same'),
            BatchNormalization(),
            Conv2D(64, (5, 5), activation='elu', padding='same'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, (3, 3), activation='elu', padding='same'),
            Dropout(0.5),
            Conv2D(64, (3, 3), activation='elu', padding='same'),
            GlobalAveragePooling2D(),
            Dense(7, activation='softmax')

        ])

        #self.model.load_weights(self.checkpoint_path)

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
