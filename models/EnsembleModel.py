import os

import tensorflow as tf
from tensorflow.keras.layers import *

from models.InceptionV4 import InceptionV4
from models.MyModel import MyModel
from models.Resnet50 import Resnet50
from models.Simple_ExpertNet import Simple_ExpertNet
from models.XCeption import XCeption


class EnsembleModel:

    def __init__(self, target_size):
        self.checkpoint_path = "model_weights/ensemble/cp.ckpt"
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)

        inception = InceptionV4(target_size).model
        resnet50 = Resnet50(target_size).model
        mymodel = MyModel(target_size).model
        xception = XCeption(target_size).model
        expert = Simple_ExpertNet(target_size).model

        inception.trainable = False
        resnet50.trainable = False
        mymodel.trainable = False
        xception.trainable = False
        expert.trainable = False

        input_layer = concatenate(
            [inception.output, resnet50.output, mymodel.output, xception.output, expert.output])
        dense_out = Dense(128, activation='relu')(input_layer)
        dense_out = Dropout(0.3)(dense_out)
        dense_out = Dense(128, activation='relu')(dense_out)
        dense_out = Dropout(0.3)(dense_out)
        pred = Dense(7, activation='softmax')(dense_out)

        self.model = tf.keras.models.Model(
            inputs=[inception.input, resnet50.input, mymodel.input, xception.input, expert.input]
            , outputs=pred
        )

        # self.model.load_weights(self.checkpoint_path)

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=tf.keras.optimizers.Adamax(learning_rate=0.0001),
                           metrics=['accuracy'],
                           )

    def train(self, epochs, train_dataset, validation_dataset):
        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
                                                         save_weights_only=True,
                                                         save_best_only=True,
                                                         verbose=0)

        history = self.model.fit(x=train_dataset,
                                 steps_per_epoch=30134 / 16,
                                 validation_data=validation_dataset,
                                 validation_steps=3345 / 16,
                                 epochs=epochs,
                                 verbose=1,
                                 callbacks=[cp_callback]
                                 )
        return history
