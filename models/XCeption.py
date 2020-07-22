import os

import tensorflow as tf
from tensorflow.keras.layers import *


class XCeption:

    def __init__(self,target_size):
        self.checkpoint_path = "model_weights/xception_training_1/cp.ckpt"
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)
        self.net = tf.keras.applications.Xception(input_shape=(target_size, target_size, 3),
                                                  include_top=False,
                                                  )

        self.net.trainable = True
        # Fine tune from this layer onwards
        fine_tune_at = 30

        # Freeze all the layers before the `fine_tune_at` layer
        for layer in self.net.layers[:fine_tune_at]:
            layer.trainable = False

        self.model = tf.keras.Sequential([
            self.net,
            Flatten(),
            Dropout(0.3),
            Dense(512, activation='relu', name='Last_Layer7'),
            tf.keras.layers.Dense(7, activation='softmax')
        ])

        self.model.load_weights(self.checkpoint_path)

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=tf.keras.optimizers.Adamax(learning_rate=0.0005),
                           metrics=['accuracy'],
                           )

    def train(self, epochs, train_dataset, validation_dataset):
        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=0)

        log_dir = "logs/fit/xception"
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