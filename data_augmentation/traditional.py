import Augmentor

p = Augmentor.Pipeline('../../dataset/The_CNBC_Face_Database_aug_tra/train/Asian')

p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
p.zoom(probability=0.5, min_factor=1.1, max_factor=1.3)
p.flip_left_right(probability=0.5)

p.sample(1455)