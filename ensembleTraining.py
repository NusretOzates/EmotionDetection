import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from models.InceptionV4 import *
from models.MobileNet import *
from models.MyModel import MyModel
from models.Resnet50 import Resnet50
from models.Resnet50_vggface import Resnet50_VGGFACE
from models.Senet50_vggface import Senet50_VGGFACE
from models.Simple_ExpertNet import Simple_ExpertNet
from models.VGG16_vggface import VGG16_VGGFACE
from models.XCeption import *

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

datagen_test = ImageDataGenerator(
    rescale=1 / 255,

)

targetsize = 197

test = datagen_test.flow_from_directory(
    'data/dev',
    target_size=(targetsize, targetsize),
    batch_size=8,
    class_mode='categorical',
    shuffle=False,
    interpolation='hamming'
)

yhats = np.zeros((6, len(test.filenames), 7))
print(yhats.shape)

resnet_vgg = Resnet50_VGGFACE(targetsize).model
senet_vgg = Senet50_VGGFACE(targetsize).model
mobilenet = MobileNet(targetsize).model
inception = InceptionV4(targetsize).model
resnet50 = Resnet50(targetsize).model
xception = XCeption(targetsize).model
expert = Simple_ExpertNet(targetsize).model
mymodel = MyModel(targetsize).model
vgg16 = VGG16_VGGFACE(targetsize).model

mobiley = mobilenet.predict(test)
inceptiony = inception.predict(test)
resnet50y = resnet50.predict(test)
xceptiony = xception.predict(test)
experty = expert.predict(test)
vgg16y = vgg16.predict(test)
resnet_vggy = resnet_vgg.predict(test)
senet_vggy = senet_vgg.predict(test)

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

print('MobileNet Accuracy score is coming!')
mobilenet.evaluate(test)

print('Inception Accuracy score is coming!')
inception.evaluate(test)

print('Resnet Accuracy score is coming!')
resnet50.evaluate(test)

print('Xception Accuracy score is coming!')
xception.evaluate(test)

print('My Model Accuracy score is coming!')
mymodel.evaluate(test)

print('Expertnet Accuracy score is coming!')
expert.evaluate(test)

print('Resnet50-VGGFACE Accuracy score is coming!')
resnet_vgg.evaluate(test)

print('VGG16-VGGFACE Accuracy score is coming!')
vgg16.evaluate(test)

print('Senet-VGGFACE Accuracy score is coming!')
senet_vgg.evaluate(test)

print('Accuracy score is coming!')
print(accuracy_score(test.labels, result))
