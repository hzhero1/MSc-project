import Augmentor

p = Augmentor.Pipeline('/home/hzhero23/stargan/data/RaFD/train/Gray_Hair')

p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
p.zoom(probability=0.3, min_factor=1.1, max_factor=1.2)
p.flip_left_right(probability=0.5)

p.sample(1000)