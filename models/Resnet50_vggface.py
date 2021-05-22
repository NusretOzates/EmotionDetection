import os

import tensorflow as tf
from keras_vggface.vggface import VGGFace
from tensorflow.keras import Model
from tensorflow.keras.layers import *


# from deepface.basemodels import VGGFace

# from deepface.basemodels import VGGFace, OpenFace, Facenet, FbDeepFace, DeepID


class Resnet50_VGGFACE:

    def __init__(self, target_size):
        self.checkpoint_path = "model_weights/senet50_vggface_training_1/cp.ckpt"
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)

        DROPOUT_RATE = 0.5
        FROZEN_LAYER_NUM = 150

        vgg_notop = VGGFace(model='resnet50', include_top=False, input_shape=(target_size, target_size, 3),
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

        print(self.model.summary())

    def train(self, epochs, train_dataset, validation_dataset):
        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
                                                         save_weights_only=True,
                                                         save_best_only=True,
                                                         verbose=0)
        log_dir = "logs/fit/resnet50_vgg"
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        history = self.model.fit(x=train_dataset,
                                 validation_data=validation_dataset,
                                 epochs=epochs,
                                 verbose=1,
                                 callbacks=[cp_callback, tensorboard_callback]
                                 )
        return history

    def showActivation(self):
        from tensorflow.keras.preprocessing import image
        import numpy as np
        from keract import get_activations, display_heatmaps
        img = image.load_img('test_pictures/nusret.png', target_size=(197, 197))
        arr = image.img_to_array(img)
        arr /= 255.0
        arr = np.expand_dims(arr, axis=0)
        images = np.vstack([arr])

        keract_inputs = images
        activations = get_activations(self.model, keract_inputs)
        display_heatmaps(activations, keract_inputs, save=False)