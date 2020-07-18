import os

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import *


class Simple_ExpertNet:

    def exFeat(self, x, kernel_size):
        l1 = Conv2D(kernel_size, (1, 1), activation='relu', padding='same')(x)
        l2 = Conv2D(kernel_size, (3, 3), activation='relu', padding='same')(x)
        l3 = Conv2D(kernel_size, (5, 5), activation='relu', padding='same')(x)
        l4 = Conv2D(kernel_size, (7, 7), activation='relu', padding='same')(x)

        return Add()([l1, l2, l3, l4])

    def __init__(self, target_size):
        self.checkpoint_path = "model_weights/simple_expertnet/cp.ckpt"
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)
        input = Input(shape=(target_size, target_size, 3))
        x = Conv2D(32, (5, 5), activation='relu', padding='same')(input)
        x = Conv2D(32, (3, 3), activation='relu', padding='same', strides=(2, 2))(x)
        x = self.exFeat(x, 32)
        x = Conv2D(64, 3, strides=(2, 2), padding='same', activation='relu')(x)
        x = self.exFeat(x, 64)
        x = Conv2D(96, 3, strides=(2, 2), activation='relu', padding='same')(x)
        x = self.exFeat(x, 96)
        x = Conv2D(128, 3, strides=(2, 2), activation='relu', padding='same')(x)
        x = self.exFeat(x, 128)
        x = Conv2D(184, kernel_size=3, activation='relu', strides=(2, 2))(x)
        x = Conv2D(256, kernel_size=3, activation='relu', strides=(2, 2))(x)
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        x = Dense(1024, activation='relu')(x)
        output = Dense(7, activation='softmax')(x)

        self.model = Model(inputs=input, outputs=output)
        self.model.load_weights(self.checkpoint_path)

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=tf.keras.optimizers.Adamax(learning_rate=0.0005),
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
