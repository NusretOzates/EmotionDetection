import os
import tensorflow as tf
from tensorflow.keras.layers import *
from .utils import *
from models.BaseModel import BaseModel


class Resnet50(BaseModel):

    def __init__(self, target_size):
        DROPOUT_RATE = 0.5
        self.checkpoint_path = "model_weights/resnet_50__training_1/cp.ckpt"
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)

        img_augmentation = image_augmentation_model()

        inputs = Input(shape=(target_size, target_size, 3))
        X = img_augmentation(inputs)

        self.net = tf.keras.applications.ResNet50V2(input_tensor=X,
                                                    include_top=False,
                                                    weights='imagenet'
                                                    )
        self.net = freeze_first_n_percent(self.net, 100)

        self.model = tf.keras.Sequential([
            self.net,
            Flatten(),
            Dropout(DROPOUT_RATE),
            Dense(512, activation='relu', name='Last_Layer6'),
            tf.keras.layers.Dense(7, activation='softmax')
        ])

        # self.model.load_weights(self.checkpoint_path)

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=tf.keras.optimizers.Adamax(learning_rate=0.0001),
                           metrics=['accuracy'],
                           )
