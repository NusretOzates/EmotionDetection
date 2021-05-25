import os

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import *
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential


class MobileNet:

    def __init__(self, target_size):
        self.checkpoint_path = "model_weights/mobilenet_training_1/cp.ckpt"
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)

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
        X = img_augmentation(inputs)

        self.net = tf.keras.applications.MobileNetV2(input_tensor=X,
                                                     include_top=False,
                                                     weights='imagenet'
                                                     )


        for layer in self.net.layers:
            layer.trainable = False

        # Fine tune from this layer onwards
        fine_tune_at = len(self.net.layers) * 20 // 100
        print(fine_tune_at)
        # Freeze all the layers before the `fine_tune_at` layer
        for layer in self.net.layers[:fine_tune_at]:
            layer.trainable = False
        for layer in self.net.layers[fine_tune_at:]:
            layer.trainable = True

        last_output = self.net.output
        X = GlobalAveragePooling2D()(last_output)

        X = Dense(7)(X)
        X = Activation('softmax', dtype='float32')(X)

        self.model = Model(self.net.input, X)

        # self.model.load_weights(self.checkpoint_path)

        self.model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                           optimizer=tf.keras.optimizers.SGD(nesterov=True),
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
                                 callbacks=[cp_callback, tensorboard_callback],
                                 class_weight={
                                     0: 3,
                                     1: 8,
                                     2: 3,
                                     3: 1,
                                     4: 2.6,
                                     5: 2.6,
                                     6: 3.4
                                 }
                                 )
        return history
