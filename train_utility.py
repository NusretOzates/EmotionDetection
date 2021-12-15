import typing

import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image_dataset_from_directory
from models import BaseModel

DATA_TEST = 'data/test'
DATA_VAL = 'data/val'
DATA_TRAIN = 'data/train'


def train_model(model: BaseModel, name, train_data, val_data, epochs=10):
    history = model.train(
        epochs=epochs,
        train=train_data,
        validation=val_data,
        name=name
    )

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(name + ' model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # # summarize history for loss
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title(name + ' model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()


def generate_train_dev_test(target_size: int, preprocessing_function=None, batch_size=16):
    datagen = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.2,
        shear_range=0.3,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode='nearest')

    datagen_dev = ImageDataGenerator()
    datagen_test = ImageDataGenerator()

    if preprocessing_function is not None:

        datagen.preprocessing_function = preprocessing_function
        datagen_dev.preprocessing_function = preprocessing_function
        datagen_test.preprocessing_function = preprocessing_function

    else:

        datagen.rescale = 1.0 / 255
        datagen_dev.rescale = 1.0 / 255
        datagen_test.rescale = 1.0 / 255

    return get_data(batch_size, datagen, datagen_dev, datagen_test, target_size)


def get_data(batch_size: int, datagen: ImageDataGenerator,
             datagen_dev: ImageDataGenerator, datagen_test: ImageDataGenerator, target_size: int):
    config = {
        'target_size': (target_size, target_size),
        'batch_size': batch_size,
        'class_mode': 'categorical',
        'interpolation': 'hamming',
        'shuffle': False
    }

    train = datagen.flow_from_directory(DATA_TRAIN, **config)
    val = datagen_dev.flow_from_directory(DATA_VAL, **config)
    test = datagen_test.flow_from_directory(DATA_TEST, **config)

    return train, val, test, datagen, datagen_dev


def train_dev_test(target_size: int, batch_size=16) -> typing.List[tf.data.Dataset]:
    config = {
        'label_mode': 'categorical',
        'seed': 42,
        'image_size': (target_size, target_size),
        'batch_size': batch_size
    }

    train_ds = image_dataset_from_directory(DATA_TRAIN, **config)
    val_ds = image_dataset_from_directory(DATA_VAL, **config)
    test_ds = image_dataset_from_directory(DATA_TEST, **config)

    class_names = train_ds.class_names
    print(class_names)

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    # # Use it when you have more RAM
    # train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    # val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    # test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return [train_ds, val_ds, test_ds]


def generate_generator_multiple(subset, target_size):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    config = {
        'directory': 'data/' + subset,
        'target_size': (target_size, target_size),
        'batch_size': 8,
        'class_mode': 'categorical',
        'interpolation': 'hamming',
        'seed': 7
    }

    generators = [image_dataset_from_directory(**config) for _ in range(5)]
    generators = [generator.cache().prefetch(buffer_size=AUTOTUNE) for generator in generators]

    while True:
        datas = [generator.next() for generator in generators]
        x = [data[0] for data in datas]
        y = datas[0][1]

        yield x, y  # Yield both images and their mutual label


def generate_labels(train):
    labels = '\n'.join(sorted(train.class_indices.keys()))

    with open('labels.txt', 'w') as f:
        f.write(labels)


def fix_gpu():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.compat.v1.InteractiveSession(config=config)
