import os

import tensorflow as tf
from keras_vggface.vggface import VGGFace
from tensorflow.keras import Model
from tensorflow.keras.layers import *


class VGG16_VGGFACE:

    def __init__(self, target_size):
        self.checkpoint_path = "model_weights/vgg16_vggface_training_1/cp.ckpt"
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)
        DROPOUT_RATE = 0.3
        print(target_size)
        self.net = VGGFace(input_shape=(target_size, target_size, 3),
                           include_top=False,
                           model='vgg16'
                           )

        self.net.trainable = True
        # Fine tune from this layer onwards
        fine_tune_at = 4

        # Freeze all the layers before the `fine_tune_at` layer
        for layer in self.net.layers[:-fine_tune_at]:
            layer.trainable = False

        last_layer = self.net.get_layer('pool5')
        last_output = last_layer.output

        # %60 test accuracy in 40 epoch , 76*76 image
        x = Flatten()(last_output)
        x = Dense(7, activation='softmax')(x)

        # %60 test accuracy in 40 epoch , 76*76 image
        # x = GlobalAveragePooling2D()(last_output)
        # x = Dense(7, activation='softmax')(x)

        self.model = Model(self.net.input, x)

        self.model.load_weights(self.checkpoint_path)

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=tf.keras.optimizers.Adamax(learning_rate=0.005),
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
