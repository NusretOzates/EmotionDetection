import os

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import *
from tensorflow.python.keras.regularizers import l2


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
        c1 = Conv2D(32, (3, 3), activation='relu', padding='same', strides=(2, 2))(x)
        x = self.exFeat(c1, 32)
        x = Add()([c1, x])
        c2 = Conv2D(64, 3, strides=(2, 2), padding='same', activation='relu')(x)
        x = self.exFeat(c2, 64)
        x = Add()([c2, x])
        c3 = Conv2D(96, 3, strides=(2, 2), activation='relu', padding='same')(x)
        x = self.exFeat(c3, 96)
        x = Add()([c3, x])
        c4 = Conv2D(128, 3, strides=(2, 2), activation='relu', padding='same')(x)
        x = self.exFeat(c4, 128)
        x = Add()([c4, x])
        x = Conv2D(184, kernel_size=3, activation='relu', strides=(2, 2))(x)
        x = Conv2D(256, kernel_size=3, activation='relu', strides=(2, 2))(x)
        x = Flatten()(x)
        x = Dense(512, activation='relu', kernel_regularizer=l2(l=0.001))(x)
        x = Dense(1024, activation='relu', kernel_regularizer=l2(l=0.001), name='Last_Layer3')(x)
        output = Dense(7, activation='softmax')(x)

        self.model = Model(inputs=input, outputs=output)
        self.model.load_weights(self.checkpoint_path)

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=tf.keras.optimizers.Adamax(learning_rate=0.0001),
                           metrics=['accuracy'],
                           )

    def train(self, epochs, train_dataset, validation_dataset):
        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
                                                         save_weights_only=True,
                                                         save_best_only=True,
                                                         verbose=0)
        log_dir = "logs/fit/expertnet"
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        history = self.model.fit(x=train_dataset,
                                 validation_data=validation_dataset,
                                 epochs=epochs,
                                 verbose=1,
                                 callbacks=[cp_callback, tensorboard_callback]
                                 )
        return history
