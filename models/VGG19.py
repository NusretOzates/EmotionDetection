import os

import tensorflow as tf
from tensorflow.keras.layers import *
from models.BaseModel import BaseModel
from utils import *


class VGG19(BaseModel):

    def __init__(self, target_size):
        self.checkpoint_path = "model_weights/vgg19_training_1/cp.ckpt"
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)

        img_augmentation = image_augmentation_model()

        inputs = Input(shape=(target_size, target_size, 3))
        x = img_augmentation(inputs)

        self.net = tf.keras.applications.VGG19(input_tensor=x,
                                               include_top=False,
                                               weights='imagenet',
                                               )

        self.net = freeze_first_n_percent(self.net, 100)

        self.model = tf.keras.Sequential([
            self.net,
            Flatten(),
            tf.keras.layers.Dense(4096, activation='relu'),
            tf.keras.layers.Dense(1024, activation='relu'),
            Dropout(0.5),
            tf.keras.layers.Dense(7, activation='softmax')
        ])

        print(self.model.summary())
        self.model.load_weights(self.checkpoint_path)

        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adamax',
                           metrics=['accuracy'])
