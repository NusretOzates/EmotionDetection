import os

import tensorflow as tf
from tensorflow.keras.layers import *

from models.InceptionV4 import InceptionV4
from models.Resnet50 import Resnet50
from models.Resnet50_vggface import Resnet50_VGGFACE
from models.Senet50_vggface import Senet50_VGGFACE
from models.XCeption import XCeption


class EnsembleModel:

    def __init__(self, target_size):
        self.checkpoint_path = "model_weights/ensemble/cp.ckpt"
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)

        inception = InceptionV4(target_size).model
        resnet50 = Resnet50(target_size).model
        senet50 = Senet50_VGGFACE(target_size).model
        xception = XCeption(target_size).model
        resnet_vgg = Resnet50_VGGFACE(target_size).model

        inception.trainable = False
        resnet50.trainable = False
        senet50.trainable = False
        xception.trainable = False
        resnet_vgg.trainable = False

        inception_output = inception.get_layer('Last_Layer5').output
        resnet50_output = resnet50.get_layer('Last_Layer6').output
        mymodel_output = senet50.get_layer('Last_Layer12').output
        xception_output = xception.get_layer('Last_Layer7').output
        expert_output = resnet_vgg.get_layer('Last_Layer').output

        models = [inception, resnet50, senet50, xception, resnet_vgg]

        i = 0
        for model in models:
            for layer in model.layers:
                layer._name = layer._name + str(i)
                i += 1

        input_layer = concatenate(
            [inception_output, resnet50_output, mymodel_output, xception_output, expert_output])

        pred = Dense(7, activation='softmax')(input_layer)

        self.model = tf.keras.models.Model(
            inputs=[inception.input, resnet50.input, senet50.input, xception.input, resnet_vgg.input]
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
                                 steps_per_epoch=30134 / 8,
                                 validation_data=validation_dataset,
                                 validation_steps=3345 / 8,
                                 epochs=epochs,
                                 verbose=1,
                                 callbacks=[cp_callback]
                                 )
        return history
