import torch
from model.DCGAN import Generator
from torchvision.utils import save_image

num_image = 2006
noise_size = 100
ngpu = 1
model_dir = '../model_weights/DCGAN_Blond_Hair.ckpt'
target_dir = '../../dataset/The_CNBC_Face_Database_aug_dcgan/train/Multiracial/'

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

noise = torch.randn(num_image, noise_size, 1, 1, device=device)

model = Generator(ngpu).to(device)
model.load_state_dict(torch.load(model_dir))
model.eval()
fake_images = model(noise)

for i, image in enumerate(fake_images):
    save_image(image, target_dir + "fake_%d.jpg" % i, normalize=True)
    print('Save %d image' % (i + 1))
