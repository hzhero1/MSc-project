import os
import shutil
import glob
import random
import numpy as np

label = "Gray_Hair"
src_dir = "../dataset/celeba/celeba_subset/hair_color/train/" + label
tar_dir = "../dataset/celeba/celeba_subset/hair_color/val/" + label

celeba_src_dir = "../dataset/celeba/align_celeba/img_align_celeba"
celeba_tar_dir = "../dataset/celeba/celeba_subset/subset"

total = len([name for name in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, name))])
print(total)
to_be_moved = random.sample(glob.glob(src_dir + '/*.jpg'), int(np.floor(0.3 * total)))

for f in to_be_moved:
    file_name = os.path.split(f)[1]

    if not os.path.exists(tar_dir):
        os.makedirs(tar_dir)
    shutil.move(f, tar_dir + '/' + file_name)

# total = len([name for name in os.listdir(celeba_src_dir) if os.path.isfile(os.path.join(celeba_src_dir, name))])
# print(total)
# to_be_moved = random.sample(glob.glob(celeba_src_dir + '/*.jpg'), 10000)
#
# for f in to_be_moved:
#     file_name = os.path.split(f)[1]
#
#     if not os.path.exists(celeba_tar_dir):
#         os.makedirs(celeba_tar_dir)
#     shutil.move(f, celeba_tar_dir + '/' + file_name)
