import os

from PIL import Image
from autocrop import Cropper

cropper = Cropper(face_percent=100)
path = 'C:/Users/nozat/Desktop/train/surprise_head'
files = os.listdir(path)
for file in files:
    # Get a Numpy array of the cropped image
    cropped_array = cropper.crop(path + '/' + file)
    # Save the cropped image with PIL
    if cropped_array is None:
        print("Kafa bulunamadi: " + file)
    else:
        cropped_image = Image.fromarray(cropped_array)
        cropped_image.save(path + '/' + file)
        print("Kafa bulundu: " + file)
