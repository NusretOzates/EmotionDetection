import os

import tensorflow as tf
from keras_vggface.vggface import VGGFace
from tensorflow.keras import Model
from tensorflow.keras.layers import *


# from deepface.basemodels import VGGFace

# from deepface.basemodels import VGGFace, OpenFace, Facenet, FbDeepFace, DeepID


class Senet50_VGGFACE:

    def __init__(self, target_size):
        # It is senet I know but there are some mistakes... It should stay like this!
        self.checkpoint_path = "model_weights/resnet50_vggface_training_1/cp.ckpt"
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)

        DROPOUT_RATE = 0.5
        FROZEN_LAYER_NUM = 201

        vgg_notop = VGGFace(model='senet50', include_top=False, input_shape=(target_size, target_size, 3),
                            pooling='avg')
        last_layer = vgg_notop.get_layer('avg_pool').output
        x = Flatten(name='flatten')(last_layer)
        x = Dropout(DROPOUT_RATE)(x)
        x = Dense(4096, activation='relu', name='fc6')(x)
        x = Dropout(DROPOUT_RATE)(x)
        x = Dense(1024, activation='relu', name='Last_Layer12')(x)
        x = Dropout(DROPOUT_RATE)(x)

        batch_norm_indices = [2, 6, 9, 12, 21, 25, 28, 31, 42, 45, 48, 59, 62, 65, 74, 78, 81, 84, 95, 98, 101, 112,
                              115, 118, 129, 132, 135, 144, 148, 151, 154, 165, 168, 171, 182, 185, 188, 199, 202, 205,
                              216, 219, 222, 233, 236, 239, 248, 252, 255, 258, 269, 272, 275]
        for i in range(FROZEN_LAYER_NUM):
            if i not in batch_norm_indices:
                vgg_notop.layers[i].trainable = False

        out = Dense(7, activation='softmax', name='classifier')(x)

        self.model = Model(vgg_notop.input, out)

        self.model.load_weights(self.checkpoint_path)

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=tf.keras.optimizers.Adamax(learning_rate=0.0001),
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
