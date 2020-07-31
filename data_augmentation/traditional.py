import Augmentor

p = Augmentor.Pipeline('/home/hzhero23/dataset/celeba/celeba_subset/hair_color_DCGAN_aug/train/gan_gray/Gray_Hair')

p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
p.zoom(probability=0.5, min_factor=1.1, max_factor=1.2)
p.flip_left_right(probability=0.5)

p.sample(1000)