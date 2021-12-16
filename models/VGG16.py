import os
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.python.keras.regularizers import l2
from models.BaseModel import BaseModel
from .utils import *


class VGG16(BaseModel):

    def __init__(self):
        self.checkpoint_path = "vgg_training_1/cp.ckpt"
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)

        img_augmentation = image_augmentation_model()

        inputs = Input(shape=(160, 160, 3))
        x = img_augmentation(inputs)

        self.net = tf.keras.applications.VGG16(input_tensor=x,
                                               include_top=False,
                                               weights='imagenet'
                                               )

        self.net = freeze_first_n_percent(self.net, 100)

        self.model = tf.keras.Sequential([
            self.net,
            Conv2D(512, 3, padding='same', activation='relu'),
            Conv2D(256, 3, padding='same', activation='relu'),
            Conv2D(128, 3, padding='same', activation='relu'),
            Flatten(),
            Dense(1024, activation='relu', kernel_regularizer=l2(0.0001)),
            Dropout(0.4),
            tf.keras.layers.Dense(7, activation='softmax')
        ])

        # self.model.load_weights(self.checkpoint_path)

        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adamax',
                           metrics=['accuracy'],
                           )