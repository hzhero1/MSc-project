import os
import shutil
import glob
import random
import numpy as np
import pandas as pd

path_attr_celeba = 'list_attr_celeba.txt'
feature = 'Blond_Hair'
celeba_subset_src = "../dataset/celeba/align_celeba/img_align_celeba"
folder_pos = '../dataset/celeba_val_test_2/Blond_Hair'
# folder_neg = '../dataset/celeba/celeba_subset/gender/female'
images_list = glob.glob(celeba_subset_src + '/*.jpg')

attr_list = pd.read_csv(path_attr_celeba, delim_whitespace=True)

feature_col = attr_list[feature]

num_images = 0
for image in images_list:
    file_name = os.path.split(image)[1]
    if feature_col[file_name] == 1:
        # move image to positive folder
        if not os.path.exists(folder_pos):
            os.makedirs(folder_pos)
        shutil.copy(image, folder_pos + '/' + file_name)
        num_images += 1
    if num_images == 1000:
        break
    # else:
    #     # move image to negative folder
    #     if not os.path.exists(folder_neg):
    #         os.makedirs(folder_neg)
    #     shutil.copy(image, folder_neg + '/' + file_name)

# print(attr_list[feature]['000010.jpg'])

