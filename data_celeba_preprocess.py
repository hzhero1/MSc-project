import os
import shutil
import glob
import random
import numpy as np
import pandas as pd

path_attr_celeba = '../dataset/celeba/list_attr_celeba.txt'
feature = 'Gray_Hair'
celeba_subset_src = "../dataset/celeba/celeba_subset/subset"
folder_pos = '../dataset/celeba/celeba_subset/hair_color/Gray_Hair'
folder_neg = ''
images_list = glob.glob(celeba_subset_src + '/*.jpg')

attr_list = pd.read_csv(path_attr_celeba, delim_whitespace=True)

feature_col = attr_list[feature]

for image in images_list:
    file_name = os.path.split(image)[1]
    if feature_col[file_name] == 1:
        # move image to positive folder
        if not os.path.exists(folder_pos):
            os.makedirs(folder_pos)
        shutil.move(image, folder_pos + '/' + file_name)
    # else:
    #     # move image to negative folder
    #     if not os.path.exists(folder_neg):
    #         os.makedirs(folder_neg)
    #     shutil.move(image, folder_neg + '/' + file_name)

# print(attr_list[feature]['000010.jpg'])

