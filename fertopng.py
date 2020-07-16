import numpy as np
import pandas as pd
from PIL import Image

df = pd.read_csv('data/fer2013.csv')
emotions = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}
df['emotion'] = [emotions[x] for x in df['emotion'] if x in emotions]

count = 0
for emotion, image_pixels, usage in zip(df['emotion'], df['pixels'], df['Usage']):
    image_string = image_pixels.split(' ')  # pixels are separated by spaces
    image_data = np.asarray(image_string, dtype=np.uint8).reshape(48, 48)
    img = Image.fromarray(image_data)  # final image\
    count_string = str(count).zfill(6)

    print(emotion)
    path = ''
    if usage == 'Training':
        path = 'data/train/'
    elif usage == 'PublicTest':
        path = 'data/dev/'
    elif usage == 'PrivateTest':
        path = 'data/test/'
    else:
        print("Exception!")

    # train/2 fear/fear-000001.png
    img.save(path + f'{emotion}/' + f'{emotion}-{count_string}.png')
    count += 1
