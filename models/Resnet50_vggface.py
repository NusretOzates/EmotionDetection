import os

import tensorflow as tf
from keras_vggface.vggface import VGGFace
from tensorflow.keras import Model
from tensorflow.keras.layers import *
from models.BaseModel import BaseModel
from utils import *
from tensorflow.keras.preprocessing import image
import numpy as np
from keract import get_activations, display_heatmaps


class Resnet50_VGGFACE(BaseModel):

    def __init__(self, target_size):
        self.checkpoint_path = "model_weights/senet50_vggface_training_1/cp.ckpt"
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)

        DROPOUT_RATE = 0.5
        FROZEN_LAYER_NUM = 150

        img_augmentation = image_augmentation_model()

        inputs = Input(shape=(target_size, target_size, 3))
        x = img_augmentation(inputs)

        vgg_notop = VGGFace(model='resnet50', include_top=False, input_tensor=x,
                            pooling='avg')

        last_layer = vgg_notop.get_layer('avg_pool').output
        x = Flatten(name='flatten')(last_layer)
        x = Dropout(DROPOUT_RATE)(x)

        batch_norm_indices = [2, 6, 9, 13, 14, 18, 21, 24, 28, 31, 34, 38, 41, 45, 46, 53, 56, 60, 63, 66, 70, 73, 76,
                              80, 83, 87, 88, 92, 95, 98, 102, 105, 108, 112, 115, 118, 122, 125, 128, 132, 135, 138,
                              142, 145, 149, 150, 154, 157, 160, 164, 167, 170]
        for i in range(FROZEN_LAYER_NUM):
            if i not in batch_norm_indices:
                vgg_notop.layers[i].trainable = False

        out = Dense(7, activation='softmax', name='classifier')(x)

        self.model = Model(vgg_notop.input, out)

        # self.model.load_weights(self.checkpoint_path)

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=tf.keras.optimizers.Adamax(learning_rate=0.0001),
                           metrics=['accuracy'],
                           )

        self.model.summary()

    def show_activation(self):
        img = image.load_img('test_pictures/nusret.png', target_size=(197, 197))
        arr = image.img_to_array(img)
        arr /= 255.0
        arr = np.expand_dims(arr, axis=0)
        images = np.vstack([arr])

        keract_inputs = images
        activations = get_activations(self.model, keract_inputs)
        display_heatmaps(activations, keract_inputs, save=False)
