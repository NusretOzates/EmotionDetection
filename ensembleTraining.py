import numpy as np
from sklearn.metrics import accuracy_score
from train_utility import generate_train_dev_test
from models.InceptionV4 import *
from models.MobileNet import *
from models.Resnet50 import Resnet50
from models.Resnet50_vggface import Resnet50_VGGFACE
from models.Senet50_vggface import Senet50_VGGFACE
from models.VGG16_vggface import VGG16_VGGFACE
from models.XCeption import *

from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras.applications import inception_resnet_v2
from tensorflow.keras.applications import resnet_v2
from tensorflow.keras.applications import xception

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

target_size = 197
targetsize = 197

resnet_vgg = Resnet50_VGGFACE(targetsize).model
senet_vgg = Senet50_VGGFACE(targetsize).model
mobilenet = MobileNet(targetsize).model
inception = InceptionV4(targetsize).model
resnet50 = Resnet50(targetsize).model
xception_model = XCeption(targetsize).model
vgg16 = VGG16_VGGFACE(targetsize).model

train, val, test, datagen, datagen_dev = generate_train_dev_test(target_size, mobilenet_v2.preprocess_input, 8)
mobiley = mobilenet.predict(test)

train, val, test, datagen, datagen_dev = generate_train_dev_test(target_size, inception_resnet_v2.preprocess_input, 8)
inceptiony = inception.predict(test)

train, val, test, datagen, datagen_dev = generate_train_dev_test(target_size, resnet_v2.preprocess_input, 8)
resnet50y = resnet50.predict(test)

train, val, test, datagen, datagen_dev = generate_train_dev_test(target_size, xception.preprocess_input, 8)
xceptiony = xception_model.predict(test)

train, val, test, datagen, datagen_dev = generate_train_dev_test(target_size, batch_size=8)
vgg16y = vgg16.predict(test)
resnet_vggy = resnet_vgg.predict(test)
senet_vggy = senet_vgg.predict(test)

yhats = np.zeros((6, len(test.filenames), 7))
print(yhats.shape)

yhats[0, :, :] = senet_vggy
yhats[1, :, :] = xceptiony
yhats[2, :, :] = resnet50y
yhats[3, :, :] = resnet_vggy

# yhats[4, :, :] = experty
# yhats[6, :, :] = mobiley
yhats[4, :, :] = vgg16y
yhats[5, :, :] = inceptiony

summed = np.sum(yhats, axis=0)
result = np.argmax(summed, axis=1)

train, val, test, datagen, datagen_dev = generate_train_dev_test(target_size, mobilenet_v2.preprocess_input, 8)
print('MobileNet Accuracy score is coming!')
mobilenet.evaluate(test)

train, val, test, datagen, datagen_dev = generate_train_dev_test(target_size, inception_resnet_v2.preprocess_input, 8)
print('Inception Accuracy score is coming!')
inception.evaluate(test)

train, val, test, datagen, datagen_dev = generate_train_dev_test(target_size, resnet_v2.preprocess_input, 8)
print('Resnet Accuracy score is coming!')
resnet50.evaluate(test)

train, val, test, datagen, datagen_dev = generate_train_dev_test(target_size, xception.preprocess_input, 8)
print('Xception Accuracy score is coming!')
xception_model.evaluate(test)

train, val, test, datagen, datagen_dev = generate_train_dev_test(target_size, batch_size=8)

print('Resnet50-VGGFACE Accuracy score is coming!')
resnet_vgg.evaluate(test)

print('VGG16-VGGFACE Accuracy score is coming!')
vgg16.evaluate(test)

print('Senet-VGGFACE Accuracy score is coming!')
senet_vgg.evaluate(test)

print('Accuracy score is coming!')
print(accuracy_score(test.labels, result))
