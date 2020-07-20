from Train_Utility import fix_gpu
from Train_Utility import generate_generator_multiple
from Train_Utility import generate_test_generator_multiple
from Train_Utility import generate_train_dev_test
from Train_Utility import train_model
from models.EnsembleModel import EnsembleModel
from models.InceptionV4 import InceptionV4
from models.MyModel import MyModel
from models.Resnet50 import Resnet50
from models.Resnet50_vggface import Resnet50_VGGFACE
from models.Simple_ExpertNet import Simple_ExpertNet
from models.VGG16_vggface import VGG16_VGGFACE
from models.XCeption import XCeption

target_size = 197

fix_gpu()

train, val, test, datagen, datagen_dev = generate_train_dev_test(target_size)

train_generator = generate_generator_multiple(datagen, 'training', target_size)
validation_generator = generate_generator_multiple(datagen, 'validation', target_size)
test_generator = generate_test_generator_multiple(datagen_dev, target_size)

mymodel = MyModel(target_size)  # %63 test max
train_model(mymodel, 'My Model', train, val)

mymodel = Resnet50_VGGFACE(target_size)
train_model(mymodel, 'Resnet50-VGGFACE', train, val)

mymodel = VGG16_VGGFACE(target_size)
train_model(mymodel, 'VGG16-VGGFACE', train, val)

mymodel = Simple_ExpertNet(target_size)
train_model(mymodel, 'ExpertNet', train, val)

# mymodel = MobileNet(target_size)  # %95 train accuracy , %64 test accuracy
# train_model(mymodel, 'MobileNet Model', train, val)

mymodel = InceptionV4(target_size)  # %66 test max
train_model(mymodel, 'Inception Model', train, val)

mymodel = Resnet50(target_size)  # %95 train accuracy , %64 test accuracy
train_model(mymodel, 'Resnet Model', train, val)

mymodel = XCeption(target_size)
train_model(mymodel, 'Xception Model', train, val)

mymodel = EnsembleModel(target_size)
train_model(mymodel, "Ensemble", train_generator, test_generator)

mymodel.model.evaluate(test_generator, steps=len(test.filenames) / 16)
