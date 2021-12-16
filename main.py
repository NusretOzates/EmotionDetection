import tensorflow as tf
from tensorflow.keras.applications import inception_resnet_v2
from tensorflow.keras import mixed_precision

from train_utility import fix_gpu, train_dev_test, generate_train_dev_test
from train_utility import train_model
from models import MobileNet, InceptionV4, Resnet50, \
    XCeption, NasnetLarge, VGG16, VGG19, Resnet101

fix_gpu()

target_size = 128

tf.get_logger().setLevel('ERROR')

train, val, test = train_dev_test(target_size, batch_size=64)

mymodel = NasnetLarge(target_size)
train_model(mymodel, 'Nasnet Large Model', train, val, 10)

mymodel = MobileNet(target_size)
train_model(mymodel, 'MobileNet Model', train, val, 10)

# train, val, test, datagen, datagen_dev = generate_train_dev_test(target_size, inception_resnet_v2.preprocess_input)
mymodel = InceptionV4(target_size)
train_model(mymodel, 'Inception Model', train, val, 10)

# train, val, test, datagen, datagen_dev = generate_train_dev_test(target_size, resnet_v2.preprocess_input,32)
mymodel = Resnet50(target_size)
train_model(mymodel, 'Resnet Model', train, val, 10)

# # train, val, test, datagen, datagen_dev = generate_train_dev_test(target_size, xception.preprocess_input,8)
mymodel = XCeption(target_size)
train_model(mymodel, 'Xception Model', train, val, 10)

mymodel = Resnet101()
train_model(mymodel, 'Resnet101 Model', train, val, 10)

mymodel = VGG16()
train_model(mymodel, 'VGG16 Model', train, val, 10)

mymodel = VGG19(target_size)
train_model(mymodel, 'VGG19 Model', train, val, 10)

# train_generator = generate_generator_multiple('train', target_size)
# validation_generator = generate_generator_multiple('val', target_size)
# test_generator = generate_generator_multiple('test', target_size)
#
# mymodel = EnsembleModel(target_size)
# train_model(mymodel, "Ensemble", train_generator, test_generator)
# # mymodel.model.evaluate(test_generator, steps=len(test.filenames) / 16)
