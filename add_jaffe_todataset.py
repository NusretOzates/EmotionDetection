from pathlib import Path

import pandas as pd

#https://www.kaggle.com/andrewmvd/japanese-female-facial-expression-dataset-jaffe/data?select=data.csv

df = pd.read_csv('jaffe/data.csv')

for path,usage in zip(df['filepath'],df['facial_expression']):

    imagename = path.split('/')[1]
    path = 'jaffe/'+path
    filepath = 'data/train/'+usage+'/'+imagename

    Path(path).rename(filepath)


