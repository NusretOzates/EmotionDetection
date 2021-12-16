import os

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import *
from .utils import *
from models.BaseModel import BaseModel


class MobileNet(BaseModel):

    def __init__(self, target_size):
        self.checkpoint_path = "model_weights/mobilenet_training_1/cp.ckpt"
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)

        img_augmentation = image_augmentation_model()

        inputs = Input(shape=(target_size, target_size, 3))
        X = img_augmentation(inputs)

        self.net = tf.keras.applications.MobileNetV2(input_tensor=X,
                                                     include_top=False,
                                                     weights='imagenet'
                                                     )

        for layer in self.net.layers:
            layer.trainable = False

        self.net = freeze_first_n_percent(self.net, 80)

        last_output = self.net.output
        X = GlobalAveragePooling2D()(last_output)
        X = Dense(7)(X)
        X = Activation('softmax', dtype='float32')(X)

        self.model = Model(self.net.input, X)

        # self.model.load_weights(self.checkpoint_path)

        self.model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                           optimizer=tf.keras.optimizers.Adam(),
                           metrics=['accuracy']
                           )

        self.model.summary()
