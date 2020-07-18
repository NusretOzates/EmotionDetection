import os

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import *
from tensorflow.python.keras.regularizers import l2


class MobileNet:

    def __init__(self, target_size):
        self.checkpoint_path = "model_weights/mobilenet_training_1/cp.ckpt"
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)
        DROPOUT_RATE = 0.5

        self.net = tf.keras.applications.MobileNetV2(input_shape=(target_size, target_size, 3),
                                                     include_top=False,
                                                     weights='imagenet'
                                                     )
        self.net.trainable = True
        # Fine tune from this layer onwards
        fine_tune_at = 80

        # Freeze all the layers before the `fine_tune_at` layer
        for layer in self.net.layers[:-fine_tune_at]:
            layer.trainable = False

        last_layer = self.net.get_layer('out_relu')
        last_output = last_layer.output
        x = Flatten()(last_output)
        x = Dropout(0.3)(x)
        x = Dense(512, activation='relu', kernel_regularizer=l2(l=0.001))(x)
        x = Dense(7, activation='softmax')(x)

        self.model = Model(self.net.input, x)

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
        log_dir = "logs/fit/mobilenet"
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        history = self.model.fit(x=train_dataset,
                                 validation_data=validation_dataset,
                                 epochs=epochs,
                                 verbose=2,
                                 callbacks=[cp_callback, tensorboard_callback]
                                 )
        return history
