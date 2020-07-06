import os
import shutil
import glob
import random
import numpy as np

label = "Hispanic"
src_dir = "../dataset/The_CNBC_Face_Database_split/val/" + label
tar_dir = "../dataset/The_CNBC_Face_Database_split/test/" + label

total = len([name for name in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, name))])
print(total)
to_be_moved = random.sample(glob.glob(src_dir + '/*.jpg'), int(np.floor(0.5 * total)))

for f in to_be_moved:
    file_name = os.path.split(f)[1]

    if not os.path.exists(tar_dir):
        os.makedirs(tar_dir)
    shutil.move(f, tar_dir + '/' + file_name)
