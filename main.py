import tensorflow as tf
from tensorflow.keras import mixed_precision

from Train_Utility import fix_gpu, train_dev_test
from Train_Utility import train_model
from models.MobileNet import MobileNet

fix_gpu()
# only uncomment if you use Nvida gpu
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_global_policy(policy)

# print('Compute dtype: %s' % policy.compute_dtype)
# print('Variable dtype: %s' % policy.variable_dtype)

target_size = 48

tf.get_logger().setLevel('ERROR')

train, val, test = train_dev_test(target_size, batch_size=64)

# mymodel = NasnetLarge(target_size)
# train_model(mymodel, 'Nasnet Large Model', train, val, 30)

mymodel = MobileNet(target_size)
train_model(mymodel, 'MobileNet Model', train, val, 10)

# train, val, test, datagen, datagen_dev = generate_train_dev_test(target_size, inception_resnet_v2.preprocess_input)
# mymodel = InceptionV4(target_size)
# train_model(mymodel, 'Inception Model', train, val, 10)
#
# train, val, test, datagen, datagen_dev = generate_train_dev_test(target_size, resnet_v2.preprocess_input,32)
# mymodel = Resnet50(target_size)
# train_model(mymodel, 'Resnet Model', train, val, 10)
# #
# #
# # train, val, test, datagen, datagen_dev = generate_train_dev_test(target_size, xception.preprocess_input,8)
# mymodel = XCeption(target_size)
# train_model(mymodel, 'Xception Model', train, val, 10)
# #
# # train, val, test, datagen, datagen_dev = generate_train_dev_test(target_size)
# #
# mymodel = Senet50_VGGFACE(target_size)
# train_model(mymodel, 'Senet-VGGFACE', train, val, 10)
# #
# mymodel = Resnet50_VGGFACE(target_size)
# train_model(mymodel, 'Resnet50-VGGFACE', train, val, 10)
# #
# # mymodel = VGG16_VGGFACE(target_size)
# # train_model(mymodel, 'VGG16-VGGFACE', train, val, 35)
# #
# # mymodel = Simple_ExpertNet(target_size)
# # train_model(mymodel, 'ExpertNet', train, val, 30)
# #
# # mymodel = MyModel(target_size)
# # train_model(mymodel, 'My Model', train, val, 40)
#
#
# train_generator = generate_generator_multiple('train', target_size)
# validation_generator = generate_generator_multiple('val', target_size)
# test_generator = generate_test_generator_multiple(target_size)
#
# mymodel = EnsembleModel(target_size)
# train_model(mymodel, "Ensemble", train_generator, test_generator)
# # mymodel.model.evaluate(test_generator, steps=len(test.filenames) / 16)
