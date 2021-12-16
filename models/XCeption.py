import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import *
from models.BaseModel import BaseModel
from .utils import *
from tensorflow.keras.preprocessing import image
from keract import get_activations, display_heatmaps


class XCeption(BaseModel):

    def __init__(self, target_size):
        self.checkpoint_path = "model_weights/xception_training_1/cp.ckpt"
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)

        img_augmentation = image_augmentation_model()

        inputs = Input(shape=(target_size, target_size, 3))
        x = img_augmentation(inputs)

        self.net = tf.keras.applications.Xception(input_tensor=x,
                                                  include_top=False,
                                                  )

        self.net = freeze_first_n_percent(self.net, 100)

        self.model = tf.keras.Sequential([
            self.net,
            Flatten(),
            Dropout(0.3),
            tf.keras.layers.Dense(7, activation='softmax')
        ])

        # self.model.load_weights(self.checkpoint_path)

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=tf.keras.optimizers.Adamax(learning_rate=0.0001),
                           metrics=['accuracy'],
                           )

    def show_activation(self):
        img = image.load_img('test_pictures/nusret.png', target_size=(197, 197))
        arr = image.img_to_array(img)
        arr /= 255.0
        arr = np.expand_dims(arr, axis=0)
        images = np.vstack([arr])

        keract_inputs = images
        activations = get_activations(self.model, keract_inputs)
        display_heatmaps(activations, keract_inputs, save=False)
