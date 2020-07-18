import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def train_model(model, name, train_data, val_data):
    epochs = 25

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
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(name + ' model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def generate_train_dev_test(target_size):
    datagen = ImageDataGenerator(
        rescale=1 / 255,
        rotation_range=20,
        zoom_range=0.2,
        shear_range=0.3,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode='nearest',
        validation_split=0.1
    )

    datagen_dev = ImageDataGenerator(
        rescale=1 / 255,
    )

    train = datagen.flow_from_directory(
        'data/train',
        target_size=(target_size, target_size),
        batch_size=32,
        subset='training',
        class_mode='categorical',
        interpolation='hamming'
    )

    val = datagen.flow_from_directory(
        'data/train',
        target_size=(target_size, target_size),
        batch_size=32,
        subset='validation',
        class_mode='categorical',
        interpolation='hamming'
    )

    test = datagen_dev.flow_from_directory(
        'data/dev',
        target_size=(197, 197),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )

    return train, val, test, datagen, datagen_dev


def generate_generator_multiple(generator, subset, target_size):
    genX1 = generator.flow_from_directory(
        'data/train',
        target_size=(target_size, target_size),
        batch_size=16,
        subset=subset,
        class_mode='categorical',
        interpolation='hamming',
        seed=7
    )

    genX2 = generator.flow_from_directory(
        'data/train',
        target_size=(target_size, target_size),
        batch_size=16,
        subset=subset,
        class_mode='categorical',
        interpolation='hamming',
        seed=7
    )

    genX3 = generator.flow_from_directory(
        'data/train',
        target_size=(target_size, target_size),
        batch_size=16,
        subset=subset,
        class_mode='categorical',
        interpolation='hamming',
        seed=7
    )

    genX4 = generator.flow_from_directory(
        'data/train',
        target_size=(target_size, target_size),
        batch_size=16,
        subset=subset,
        class_mode='categorical',
        interpolation='hamming',
        seed=7
    )

    genX5 = generator.flow_from_directory(
        'data/train',
        target_size=(target_size, target_size),
        batch_size=16,
        subset=subset,
        class_mode='categorical',
        interpolation='hamming',
        seed=7
    )
    genX6 = generator.flow_from_directory(
        'data/train',
        target_size=(target_size, target_size),
        batch_size=16,
        subset=subset,
        class_mode='categorical',
        interpolation='hamming',
        seed=7
    )

    while True:
        X1i = genX1.next()
        X2i = genX2.next()
        X3i = genX3.next()
        X4i = genX4.next()
        X5i = genX5.next()
        X6i = genX5.next()

        yield [X1i[0], X2i[0], X3i[0], X4i[0], X5i[0], X6i[0]], X2i[1]  # Yield both images and their mutual label


def generate_test_generator_multiple(generator, target_size):
    genX1 = generator.flow_from_directory(
        'data/dev',
        target_size=(target_size, target_size),
        batch_size=16,
        class_mode='categorical',
        interpolation='hamming',
        seed=7
    )

    genX2 = generator.flow_from_directory(
        'data/dev',
        target_size=(target_size, target_size),
        batch_size=16,
        class_mode='categorical',
        interpolation='hamming',
        seed=7
    )

    genX3 = generator.flow_from_directory(
        'data/dev',
        target_size=(target_size, target_size),
        batch_size=16,
        class_mode='categorical',
        interpolation='hamming',
        seed=7
    )
    genX4 = generator.flow_from_directory(
        'data/dev',
        target_size=(target_size, target_size),
        batch_size=16,
        class_mode='categorical',
        interpolation='hamming',
        seed=7
    )
    genX5 = generator.flow_from_directory(
        'data/dev',
        target_size=(target_size, target_size),
        batch_size=16,
        class_mode='categorical',
        interpolation='hamming',
        seed=7
    )

    genX6 = generator.flow_from_directory(
        'data/dev',
        target_size=(target_size, target_size),
        batch_size=16,
        class_mode='categorical',
        interpolation='hamming',
        seed=7
    )

    while True:
        X1i = genX1.next()
        X2i = genX2.next()
        X3i = genX3.next()
        X4i = genX4.next()
        X5i = genX5.next()
        X6i = genX5.next()

        yield [X1i[0], X2i[0], X3i[0], X4i[0], X5i[0], X6i[0]], X2i[1]  # Yield both images and their mutual label


def generateLabels(train):
    labels = '\n'.join(sorted(train.class_indices.keys()))

    with open('labels.txt', 'w') as f:
        f.write(labels)


def fix_gpu():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.InteractiveSession(config=config)
