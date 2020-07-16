import os

import tensorflow as tf
from keras_vggface.vggface import VGGFace
from tensorflow.keras import Model
from tensorflow.keras.layers import *


# from deepface.basemodels import VGGFace

# from deepface.basemodels import VGGFace, OpenFace, Facenet, FbDeepFace, DeepID


class Senet50_VGGFACE:

    def __init__(self, target_size):
        self.checkpoint_path = "model_weights/senet50_vggface_training_1/cp.ckpt"
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)
        # custom parameters
        nb_class = 7

        # vgg = VGGFace.loadModel()

        vgg_model = VGGFace(include_top=False, input_shape=(target_size, target_size, 3), model='senet50')
        vgg_model.trainable = True


        last_layer = vgg_model.get_layer('avg_pool').output
        x = Flatten()(last_layer)
        x = Dropout(0.3)(x)
        x = Dense(1024, activation='relu')(x)
        out = Dense(nb_class, activation='softmax', name='classifier')(x)
        self.model = Model(vgg_model.input, out)

        # self.model.load_weights(self.checkpoint_path)

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=tf.keras.optimizers.Adamax(learning_rate=0.0005),
                           metrics=['accuracy'],
                           )

        print(self.model.summary())

    def train(self, epochs, train_dataset, validation_dataset):
        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
                                                         save_weights_only=True,
                                                         save_best_only=False,
                                                         verbose=0)
        history = self.model.fit(x=train_dataset,
                                 validation_data=validation_dataset,
                                 epochs=epochs,
                                 verbose=1,
                                 callbacks=[cp_callback]
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