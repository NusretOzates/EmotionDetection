import os

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import *
from tensorflow.python.keras.regularizers import l2


class MobileNet:

    def __init__(self, target_size):
        self.checkpoint_path = "model_weights/mobilenet_training_1/cp.ckpt"
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)

        from tensorflow.keras.layers.experimental import preprocessing
        from tensorflow.keras.models import Sequential
        from tensorflow.keras import layers

        img_augmentation = Sequential(
            [
                preprocessing.Rescaling(scale=1.0 / 255),
                preprocessing.RandomRotation(factor=0.15),
                preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
                preprocessing.RandomFlip(),
                preprocessing.RandomContrast(factor=0.1),
            ],
            name="img_augmentation",
        )

        inputs = layers.Input(shape=(target_size, target_size, 3))
        x = img_augmentation(inputs)

        self.net = tf.keras.applications.MobileNetV2(input_tensor=x,
                                                     include_top=False,
                                                     weights='imagenet'
                                                     )
        self.net.trainable = True
        # Fine tune from this layer onwards
        fine_tune_at = 50

        # Freeze all the layers before the `fine_tune_at` layer
        for layer in self.net.layers[:-fine_tune_at]:
            layer.trainable = False

        last_layer = self.net.get_layer('out_relu')
        last_output = last_layer.output
        x = Flatten()(last_output)
        x = Dropout(0.3)(x)
        x = Dense(512, activation='relu', kernel_regularizer=l2(l=0.001), name='Last_Layer4')(x)
        x = Dense(7, activation='softmax')(x)

        self.model = Model(self.net.input, x)

        self.model.load_weights(self.checkpoint_path)

        self.model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
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
        log_dir = "logs/fit/mobilenet"
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        history = self.model.fit(x=train_dataset,
                                 validation_data=validation_dataset,
                                 epochs=epochs,
                                 verbose=1,
                                 callbacks=[cp_callback, tensorboard_callback]
                                 )
        return history
