import os

import tensorflow as tf
from tensorflow.keras.layers import *
from models.BaseModel import BaseModel
from utils import *


class Resnet50(BaseModel):

    def __init__(self):
        self.checkpoint_path = "model_weights/resnet_101__training_1/cp.ckpt"
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)

        img_augmentation = image_augmentation_model()

        inputs = Input(shape=(160, 160, 3))
        x = img_augmentation(inputs)

        self.net = tf.keras.applications.ResNet101V2(input_tensor=x,
                                                     include_top=False,
                                                     weights='imagenet'
                                                     )

        self.net = freeze_first_n_percent(self.net, 90)

        self.model = tf.keras.Sequential([
            self.net,
            GlobalAveragePooling2D(),
            Dense(7, activation='softmax', name='classifier')
        ])

        self.net.summary()
        # self.model.load_weights(self.checkpoint_path)

        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adamax',
                           metrics=['accuracy'])
