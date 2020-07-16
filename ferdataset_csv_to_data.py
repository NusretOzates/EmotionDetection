import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

x = pd.read_csv('data/fer2013.csv')
# print(x.values.shape)
# print(x.head())

data = x.values
# print(data[0, 0])

# emotions
y = data[:, 0]
# print(y.shape)

# pixels
pixels = data[:, 1]

# print(type(pixels))
# print(len(pixels[0]))

# Ornek sayisi kadar satir, resim boyutu kadar sutun
X = np.zeros((pixels.shape[0], 48 * 48))

# Iterate on all rows
for samp in range(X.shape[0]):
    # pixels of 1 row
    p = pixels[samp].split(' ')
    for col in range(X.shape[1]):
        # p[col] value of col'th pixel
        X[samp, col] = int(p[col])

x = X
# for ix in range(4):
#     plt.figure(ix)
#     plt.imshow(x[ix].reshape((48, 48)), interpolation='none', cmap='gray')
#
# plt.show()

y = to_categorical(y, 7)

x_train = x[0:32300, :]
y_train = y[0:32300]

# print(x_train.shape)
# print(y_train.shape)

x_dev_set = x[32300:34100, :]
y_dev_set = y[32300:34100]

x_test_set = x[34100:, :]
y_test_set = y[34100:]

x_train = x_train.reshape((x_train.shape[0], 48, 48, 1))
x_dev_set = x_dev_set.reshape((x_dev_set.shape[0], 48, 48, 1))
x_test_set = x_test_set.reshape((x_test_set.shape[0], 48, 48, 1))

datagen = ImageDataGenerator(
    rescale=1 / 255,

    rotation_range=10,
    zoom_range=0.1,
    shear_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False,
    fill_mode='nearest'
)

train_generator = datagen.flow(x_train, y_train)
test_generator = datagen.flow(x_test_set, y_test_set)
dev_generator = datagen.flow(x_dev_set, y_dev_set)
