import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATA_TEST = 'data/test'



def train_model(model, name, train_data, val_data, epochs=10):
    history = model.train(
        epochs=epochs,
        train_dataset=train_data,
        validation_dataset=val_data,
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


def generate_train_dev_test(target_size, preprocessing_function=None, batch_size=16):
    data_val = 'data/val'
    data_train = 'data/train'
    if preprocessing_function is not None:
        datagen = ImageDataGenerator(
            rotation_range=20,
            zoom_range=0.2,
            shear_range=0.3,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=False,
            fill_mode='nearest',
            preprocessing_function=preprocessing_function
        )

        datagen_dev = ImageDataGenerator(
            preprocessing_function=preprocessing_function
        )

        datagen_test = ImageDataGenerator(
            preprocessing_function=preprocessing_function
        )

        return get_data(batch_size, data_train, data_val, datagen, datagen_dev, datagen_test, target_size)
    else:

        datagen = ImageDataGenerator(
            rotation_range=20,
            zoom_range=0.2,
            shear_range=0.3,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=False,
            fill_mode='nearest',
            rescale=1.0 / 255
        )

        datagen_dev = ImageDataGenerator(
            rescale=1.0 / 255
        )

        datagen_test = ImageDataGenerator(
            rescale=1.0 / 255
        )

        return get_data(batch_size, data_train, data_val, datagen, datagen_dev, datagen_test, target_size)


def get_data(batch_size, data_train, data_val, datagen, datagen_dev, datagen_test, target_size):
    train = datagen.flow_from_directory(
        data_train,
        target_size=(target_size, target_size),
        batch_size=batch_size,
        class_mode='categorical',
        interpolation='hamming',
        shuffle=False
    )

    val = datagen_dev.flow_from_directory(
        data_val,
        target_size=(target_size, target_size),
        batch_size=batch_size,
        class_mode='categorical',
        interpolation='hamming',
        shuffle=False
    )
    test = datagen_test.flow_from_directory(
        DATA_TEST,
        target_size=(target_size, target_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        interpolation='hamming',
    )
    return train, val, test, datagen, datagen_dev


def train_dev_test(target_size, batch_size=16):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        'data/train',
        label_mode='categorical',
        seed=123,
        image_size=(target_size, target_size),
        batch_size=batch_size)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        'data/val',
        label_mode='categorical',
        seed=123,
        image_size=(target_size, target_size),
        batch_size=batch_size)

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        'data/test',
        label_mode='categorical',
        seed=123,
        image_size=(target_size, target_size),
        batch_size=batch_size)

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

    return train_ds, val_ds, test_ds


def generate_generator_multiple(subset, target_size):
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    genX1 = tf.keras.preprocessing.image_dataset_from_directory(
        'data/' + subset,
        target_size=(target_size, target_size),
        batch_size=8,
        class_mode='categorical',
        interpolation='hamming',
        seed=7
    ).prefetch(buffer_size=AUTOTUNE)

    genX2 = tf.keras.preprocessing.image_dataset_from_directory(
        'data/' + subset,
        target_size=(target_size, target_size),
        batch_size=8,
        class_mode='categorical',
        interpolation='hamming',
        seed=7
    ).prefetch(buffer_size=AUTOTUNE)

    genX3 = tf.keras.preprocessing.image_dataset_from_directory(
        'data/' + subset,
        target_size=(target_size, target_size),
        batch_size=8,
        class_mode='categorical',
        interpolation='hamming',
        seed=7
    ).prefetch(buffer_size=AUTOTUNE)

    genX4 = tf.keras.preprocessing.image_dataset_from_directory(
        'data/' + subset,
        target_size=(target_size, target_size),
        batch_size=8,
        class_mode='categorical',
        interpolation='hamming',
        seed=7
    ).prefetch(buffer_size=AUTOTUNE)

    genX5 = tf.keras.preprocessing.image_dataset_from_directory(
        'data/' + subset,
        target_size=(target_size, target_size),
        batch_size=8,
        class_mode='categorical',
        interpolation='hamming',
        seed=7
    ).prefetch(buffer_size=AUTOTUNE)

    while True:
        X1i = genX1.next()
        X2i = genX2.next()
        X3i = genX3.next()
        X4i = genX4.next()
        X5i = genX5.next()

        yield [X1i[0], X2i[0], X3i[0], X4i[0], X5i[0]], X2i[1]  # Yield both images and their mutual label


def generate_test_generator_multiple(target_size):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    genX1 = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_TEST,
        target_size=(target_size, target_size),
        batch_size=8,
        class_mode='categorical',
        interpolation='hamming',
        seed=7
    ).prefetch(buffer_size=AUTOTUNE)

    genX2 = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_TEST,
        target_size=(target_size, target_size),
        batch_size=8,
        class_mode='categorical',
        interpolation='hamming',
        seed=7
    ).prefetch(buffer_size=AUTOTUNE)

    genX3 = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_TEST,
        target_size=(target_size, target_size),
        batch_size=8,
        class_mode='categorical',
        interpolation='hamming',
        seed=7
    ).prefetch(buffer_size=AUTOTUNE)
    genX4 = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_TEST,
        target_size=(target_size, target_size),
        batch_size=8,
        class_mode='categorical',
        interpolation='hamming',
        seed=7
    ).prefetch(buffer_size=AUTOTUNE)
    genX5 = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_TEST,
        target_size=(target_size, target_size),
        batch_size=8,
        class_mode='categorical',
        interpolation='hamming',
        seed=7
    ).prefetch(buffer_size=AUTOTUNE)

    while True:
        X1i = genX1.next()
        X2i = genX2.next()
        X3i = genX3.next()
        X4i = genX4.next()
        X5i = genX5.next()

        yield [X1i[0], X2i[0], X3i[0], X4i[0], X5i[0]], X2i[1]  # Yield both images and their mutual label


def generat_labels(train):
    labels = '\n'.join(sorted(train.class_indices.keys()))

    with open('labels.txt', 'w') as f:
        f.write(labels)


def fix_gpu():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.compat.v1.InteractiveSession(config=config)
