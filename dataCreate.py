import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os
import csv
source = "charlies"

FAST_RUN = False
IMAGE_WIDTH=128
IMAGE_HEIGHT=128
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3
def createCsv():
    f = open('data.csv', 'w')
    with f:
        fHeaders = ['imgName', 'isCharlie']
        writer = csv.DictWriter(f, fieldnames=fHeaders)
        filenames = os.listdir("charlies")
        for filename in filenames:
            print('filename', filename)
            category = filename.split('_')[0]
            print('category', category)
            if category == 'yes':
                writer.writerow({'imgName': filename, 'isCharlie': '1'})
            else:
                writer.writerow({'imgName': filename, 'isCharlie': '0'})
    # def getDataFrame():
#     filenames = os.listdir("charlies")
#     categories = []
#     for filename in filenames:
#         category = filename.split('_')[0]
#         if category == 'yes':
#             categories.append(1)
#         else:
#             categories.append(0)
#
#     df = pd.DataFrame({
#         'filename': filenames,
#         'category': categories
#     })
#     print(df)
    # df['category'].value_counts().plot.bar()
    #plt.show()

    # sample = random.choice(filenames)
    # image = load_img(source + "/" + sample)
    # plt.imshow(image)
    #plt.show()

    # df["category"] = df["category"].replace({0: 'non', 1: 'oui'})
    # return df





