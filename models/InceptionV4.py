import os
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import *
from utils import image_augmentation_model, freeze_first_n_percent
from models.BaseModel import BaseModel


class InceptionV4(BaseModel):

    def __init__(self, target_size: int):
        self.checkpoint_path = "model_weights/inceptionv4_training_1/cp.ckpt"
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)

        img_augmentation = image_augmentation_model()

        inputs = layers.Input(shape=(target_size, target_size, 3))
        x = img_augmentation(inputs)

        self.net = tf.keras.applications.InceptionResNetV2(input_tensor=x,
                                                           include_top=False,
                                                           weights='imagenet'
                                                           )
        # Freeze all layers
        for layer in self.net.layers:
            layer.trainable = False

        # Freeze first 70% of the layers
        self.net = freeze_first_n_percent(self.net, 70)

        last_output = self.net.output
        X = GlobalAveragePooling2D()(last_output)
        X = Dense(7)(X)
        X = Activation('softmax', dtype='float32')(X)

        self.model = Model(self.net.input, X)

        # Load saved weights
        #self.model.load_weights(self.checkpoint_path)

        self.model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                           optimizer=tf.keras.optimizers.SGD(nesterov=True),
                           metrics=['accuracy'],
                           )

        self.model.summary()
