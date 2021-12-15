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


class VGG16_VGGFACE(BaseModel):

    def __init__(self, target_size):
        self.checkpoint_path = "model_weights/vgg16_vggface_training_1/cp.ckpt"
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)

        DROPOUT_RATE = 0.3

        img_augmentation = image_augmentation_model()

        inputs = Input(shape=(target_size, target_size, 3))
        x = img_augmentation(inputs)

        vgg_notop = VGGFace(model='vgg16', include_top=False, input_shape=x, pooling='avg')

        vgg_notop = freeze_first_n_percent(vgg_notop, 90)

        last_layer = vgg_notop.get_layer('pool5').output
        x = Flatten(name='flatten')(last_layer)
        x = Dropout(DROPOUT_RATE)(x)
        x = Dense(4096, activation='relu', name='fc6')(x)
        x = Dropout(DROPOUT_RATE)(x)
        x = Dense(1024, activation='relu', name='fc7')(x)

        out = Dense(7, activation='softmax', name='classifier')(x)

        self.model = Model(vgg_notop.input, out)

        self.model.load_weights(self.checkpoint_path)

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=tf.keras.optimizers.Adamax(learning_rate=0.0001),
                           metrics=['accuracy'])

    def show_activation(self):
        img = image.load_img('test_pictures/nusret.png', target_size=(197, 197))
        arr = image.img_to_array(img)
        arr /= 255.0
        arr = np.expand_dims(arr, axis=0)
        images = np.vstack([arr])

        keract_inputs = images
        activations = get_activations(self.model, keract_inputs)
        display_heatmaps(activations, keract_inputs, save=False)
