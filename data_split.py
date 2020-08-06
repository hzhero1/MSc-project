import os
import shutil
import glob
import random
import numpy as np

label = "Gray_Hair"
src_dir = "../dataset/celeba/celeba_subset/hair_color/val/" + label
tar_dir = "../dataset/celeba/celeba_subset/hair_color/test/" + label

celeba_src_dir = "../dataset/celeba/align_celeba/img_align_celeba"
celeba_tar_dir = "../dataset/celeba/celeba_subset/subset"

source = "/home/hzhero23/pytorch-CycleGAN-and-pix2pix/results/black2gray_cyclegan/test_latest/fake"
target = "/home/hzhero23/dataset/celeba/celeba_subset/hair_color_cyclegan_aug/train/Gray_Hair"

#total = len([name for name in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, name))])
#print(total)
#to_be_moved = random.sample(glob.glob(src_dir + '/*.jpg'), int(np.floor(0.5 * total)))

#for f in to_be_moved:
 #   file_name = os.path.split(f)[1]

  #  if not os.path.exists(tar_dir):
   #     os.makedirs(tar_dir)
    #shutil.move(f, tar_dir + '/' + file_name)

total = len([name for name in os.listdir(source) if os.path.isfile(os.path.join(source, name))])
print(total)
to_be_moved = random.sample(glob.glob(source + '/*.png'), 1434)

for f in to_be_moved:
     file_name = os.path.split(f)[1]

     if not os.path.exists(target):
         os.makedirs(target)
     shutil.move(f, target + '/' + file_name)
