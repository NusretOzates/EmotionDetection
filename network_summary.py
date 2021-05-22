from keras.applications import NASNetLarge
from keras_vggface import VGGFace

from Train_Utility import fix_gpu

#
# mymodel = tf.keras.applications.ResNet101V2(include_top=False, input_shape=(160, 160, 3),weights='imagenet')
# print(len(mymodel.layers))

fix_gpu()

mymodel = NASNetLarge(include_top=False, input_shape=(331, 331, 3), weights='imagenet')
for i, layer in enumerate(mymodel.layers):
   print(i, layer.name)
