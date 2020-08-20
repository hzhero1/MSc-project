import torch
from torchvision.utils import save_image
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

model_dir = '/home/hzhero23/CelebA_cDCGAN_results/CelebA_cDCGAN_generator_param.pkl'
target_dir = '/home/hzhero23/dataset/celeba/celeba_subset/hair_color_cdcgan_aug/train/Gray_Hair/'

num_images = 1434


class generator(nn.Module):
    def __init__(self, d=128):
        super(generator, self).__init__()
        self.deconv1_1 = nn.ConvTranspose2d(100, d * 4, 4, 1, 0)
        self.deconv1_1_bn = nn.BatchNorm2d(d * 4)
        self.deconv1_2 = nn.ConvTranspose2d(2, d * 4, 4, 1, 0)
        self.deconv1_2_bn = nn.BatchNorm2d(d * 4)
        self.deconv2 = nn.ConvTranspose2d(d * 8, d * 4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d * 4)
        self.deconv3 = nn.ConvTranspose2d(d * 4, d * 2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d * 2)
        # self.deconv4 = nn.ConvTranspose2d(d, 3, 4, 2, 1)
        self.deconv4 = nn.ConvTranspose2d(d * 2, d, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, 3, 4, 2, 1)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input, label):
        x = F.leaky_relu(self.deconv1_1_bn(self.deconv1_1(input)), 0.2)
        y = F.leaky_relu(self.deconv1_2_bn(self.deconv1_2(label)), 0.2)
        x = torch.cat([x, y], 1)
        x = F.leaky_relu(self.deconv2_bn(self.deconv2(x)), 0.2)
        x = F.leaky_relu(self.deconv3_bn(self.deconv3(x)), 0.2)
        # x = F.tanh(self.deconv4(x))
        x = F.leaky_relu(self.deconv4_bn(self.deconv4(x)), 0.2)
        x = torch.tanh(self.deconv5(x))

        return x


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


# label preprocess
img_size = 64
onehot = torch.zeros(2, 2)
onehot = onehot.scatter_(1, torch.LongTensor([0, 1]).view(2, 1), 1).view(2, 2, 1, 1)
fill = torch.zeros([2, 2, img_size, img_size])
for i in range(2):
    fill[i, i, :, :] = 1

G = generator(128)
G.load_state_dict(torch.load(model_dir))
G.cuda()
G.eval()
# test_images = G(fixed_z_, fixed_y_label_)
G.train()

z_ = torch.randn((num_images, 100)).view(-1, 100, 1, 1)
y_ = (torch.ones(num_images, 1)).type(torch.LongTensor).squeeze()
y_label_ = onehot[y_]
z_, y_label_ = Variable(z_.cuda()), Variable(y_label_.cuda())

fake_images = G(z_, y_label_)

for i, image in enumerate(fake_images):
    save_image(image, target_dir + "fake_%d.jpg" % i, normalize=True)
    print('Save %d image' % (i + 1))
