import os

import tensorflow as tf
from tensorflow.keras.layers import *


class MyModel:

    def __init__(self,target_size):
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
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dropout(0.2))

        self.model.add(Dense(num_classes, activation='softmax'))

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
