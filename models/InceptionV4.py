import os
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import *
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential


class InceptionV4:

    def __init__(self, target_size):
        self.checkpoint_path = "model_weights/inceptionv4_training_1/cp.ckpt"
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
        x = img_augmentation(inputs)

        self.net = tf.keras.applications.InceptionResNetV2(input_tensor=x,
                                                           include_top=False,
                                                           weights='imagenet'
                                                           )

        for layer in self.net.layers:
            layer.trainable = False

        #     # Fine tune from this layer onwards
        # fine_tune_at = len(self.net.layers) * 70 // 100
        # print(fine_tune_at)
        # # Freeze all the layers before the `fine_tune_at` layer
        # for layer in self.net.layers[:fine_tune_at]:
        #     layer.trainable = False
        # for layer in self.net.layers[fine_tune_at:]:
        #     layer.trainable = True

        last_output = self.net.output
        X = GlobalAveragePooling2D()(last_output)

        X = self.create_attention(X, X, X)
        X = self.create_attention(X, X, X)

        X = Dense(7)(X)
        X = Activation('softmax', dtype='float32')(X)

        self.model = Model(self.net.input, X)

        self.model.load_weights(self.checkpoint_path)

        self.model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                           optimizer=tf.keras.optimizers.SGD(nesterov=True),
                           metrics=['accuracy'],
                           )

        print(self.model.summary())

    def create_attention(self, X, key, value):
        K = MultiHeadAttention(8, X.shape[1], attention_axes=1)(key, value)

        X = Add()([X, K])
        add_norm_1 = LayerNormalization()(X)

        # C = Concatenate(axis=-1)([X, Y, Z])
        X = Dense(X.shape[1], activation='elu')(add_norm_1)
        X = Add()([X, add_norm_1])
        X = LayerNormalization()(X)

        X = Dense(512, activation='elu')(X)
        return X

    def train(self, epochs, train_dataset, validation_dataset):
        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
                                                         save_weights_only=True,
                                                         save_best_only=True,
                                                         verbose=0)

        log_dir = "logs/fit/inception"
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
        img = image.load_img('test_pictures/nusret.png', target_size=(196, 196))
        arr = image.img_to_array(img)
        arr /= 255.0
        arr = np.expand_dims(arr, axis=0)
        images = np.vstack([arr])

        keract_inputs = images
        activations = get_activations(self.model, keract_inputs)
        display_heatmaps(activations, keract_inputs, save=False)
