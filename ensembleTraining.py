import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from models.InceptionV4 import *
from models.MobileNet import *
from models.MyModel import MyModel
from models.Resnet50 import Resnet50
from models.Simple_ExpertNet import Simple_ExpertNet
from models.XCeption import *

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

datagen_test = ImageDataGenerator(
    rescale=1 / 255,


)

test = datagen_test.flow_from_directory(
    'data/dev',
    target_size=(128, 128),
    batch_size=8,
    class_mode='categorical',
    shuffle=False,
    interpolation='hamming'
)

yhats = np.zeros((5, len(test.filenames), 7))
print(yhats.shape)

targetsize = 128

mobilenet = MobileNet(targetsize).model
inception = InceptionV4(targetsize).model
resnet50 = Resnet50(targetsize).model
xception = XCeption(targetsize).model
expert = Simple_ExpertNet(targetsize).model
mymodel = MyModel(targetsize).model
#
#
#
mobiley = mobilenet.predict(test)
inceptiony = inception.predict(test)
resnet50y = resnet50.predict(test)
xceptiony = xception.predict(test)
mymodely = mymodel.predict(test)
experty = expert.predict(test)

yhats[0, :, :] = mymodely
yhats[1, :, :] = inceptiony
yhats[2, :, :] = xceptiony
yhats[3, :, :] = resnet50y
yhats[4, :, :] = experty
# yhats[5, :, :] = mobiley

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

print('Accuracy score is coming!')
print(accuracy_score(test.labels, result))
