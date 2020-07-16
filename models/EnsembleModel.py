import os

import tensorflow as tf
from tensorflow.keras.layers import *

from models.InceptionV4 import InceptionV4
from models.MobileNet import MobileNet
from models.MyModel import MyModel
from models.Resnet50 import Resnet50
from models.VGG16_vggface import VGG16_VGGFACE
from models.XCeption import XCeption


class EnsembleModel:

    def __init__(self, target_size):
        self.checkpoint_path = "model_weights/ensemble/cp.ckpt"
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)

        inception = InceptionV4(target_size).model
        VGG16 = VGG16_VGGFACE(target_size).model
        resnet50 = Resnet50(target_size).model
        mymodel = MyModel().model
        mobilenet = MobileNet(target_size).model
        xception = XCeption(target_size).model

        inception.trainable = False
        VGG16.trainable = False
        resnet50.trainable = False
        mymodel.trainable = False
        mobilenet.trainable = False
        xception.trainable = False

        input_layer = concatenate(
            [VGG16.output, inception.output, resnet50.output, mymodel.output, mobilenet.output, xception.output])
        dense_out = Dense(256, activation='relu')(input_layer)
        dense_out = Dropout(0.3)(dense_out)
        pred = Dense(7, activation='softmax')(dense_out)

        self.model = tf.keras.models.Model(
            inputs=[VGG16.input, inception.input, resnet50.input, mymodel.input, mobilenet.input, xception.input]
            , outputs=pred
            )

        # self.model.load_weights(self.checkpoint_path)

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=tf.keras.optimizers.Adamax(learning_rate=0.0005),
                           metrics=['accuracy'],
                           )

    def train(self, epochs, train_dataset, validation_dataset):
        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
                                                         save_weights_only=True,
                                                         save_best_only=True,
                                                         verbose=0)

        history = self.model.fit(x=train_dataset,
                                 steps_per_epoch=30336 / 16,
                                 validation_data=validation_dataset,
                                 validation_steps=3369 / 16,
                                 epochs=epochs,
                                 verbose=1,
                                 callbacks=[cp_callback]
                                 )
        return history
