from keras_vggface import VGGFace

from Train_Utility import fix_gpu

#
# mymodel = tf.keras.applications.ResNet101V2(include_top=False, input_shape=(160, 160, 3),weights='imagenet')
# print(len(mymodel.layers))

fix_gpu()

mymodel = VGGFace(model='vgg16', include_top=False, input_shape=(197, 197, 3),
                  pooling='avg')
print(len(mymodel.layers))
