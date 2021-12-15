import os
import tensorflow as tf
from tensorflow.keras.layers import *
from utils import *
from models.BaseModel import BaseModel

class NasnetLarge(BaseModel):

    def __init__(self, target_size):
        self.checkpoint_path = "model_weights/nasnet_training_1/cp.ckpt"
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)

        img_augmentation = image_augmentation_model()

        inputs = Input(shape=(target_size, target_size, 3))
        x = img_augmentation(inputs)

        self.net = tf.keras.applications.NASNetLarge(input_tensor=x,
                                                     include_top=False,
                                                     weights='imagenet'
                                                     )
        # for layer in self.net.layers:
        #     layer.trainable = False

        freeze_first_n_percent(self.net, 70)

        last_layer = self.net.output
        x = GlobalAveragePooling2D()(last_layer)
        x = Flatten()(x)
        x = Dropout(0.3)(x)
        x = Dense(7, activation='softmax')(x)

        self.model = Model(self.net.input, x)

        self.model.load_weights(self.checkpoint_path)

        self.model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                           optimizer='adam',
                           metrics=['accuracy'],
                           )

        self.model.summary()
