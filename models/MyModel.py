import os

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.python.keras.regularizers import l2


class MyModel:

    def __init__(self, target_size):
        self.checkpoint_path = "mymodel_training_1/cp.ckpt"
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)
        num_classes = 7

        self.model = tf.keras.Sequential()

        # 1st convolution layer
        self.model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(target_size, target_size, 3)))
        self.model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))

        # 2nd convolution layer
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

        # 3rd convolution layer
        self.model.add(Conv2D(128, (3, 3), activation='relu'))
        self.model.add(Conv2D(128, (3, 3), activation='relu'))
        self.model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

        self.model.add(Flatten())

        # fully connected neural networks
        self.model.add(Dense(1024, activation='relu', kernel_regularizer=l2(l=0.0001)))
        self.model.add(Dropout(0.4))
        self.model.add(Dense(1024, activation='relu', name='Last_Layer2', kernel_regularizer=l2(l=0.0001)))
        self.model.add(Dropout(0.4))

        self.model.add(Dense(num_classes, activation='softmax'))

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
        log_dir = "logs/fit/mymodel"
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        history = self.model.fit(x=train_dataset,
                                 validation_data=validation_dataset,
                                 epochs=epochs,
                                 verbose=1,
                                 callbacks=[cp_callback, tensorboard_callback]
                                 )
        return history
