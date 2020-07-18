import tensorflow as tf

from Train_Utility import fix_gpu

#
# mymodel = tf.keras.applications.ResNet101V2(include_top=False, input_shape=(160, 160, 3),weights='imagenet')
# print(len(mymodel.layers))

fix_gpu()

mymodel = tf.keras.applications.ResNet50V2(include_top=False, input_shape=(128, 128, 3), weights='imagenet')
print(len(mymodel.layers))

mymodel = tf.keras.applications.InceptionResNetV2(include_top=False, input_shape=(128, 128, 3), weights='imagenet')
print(len(mymodel.layers))

mymodel = tf.keras.applications.MobileNetV2(include_top=False, input_shape=(128, 128, 3), weights='imagenet')
print(len(mymodel.layers))

mymodel = tf.keras.applications.Xception(include_top=False, input_shape=(128, 128, 3), weights='imagenet')
print(len(mymodel.layers))
