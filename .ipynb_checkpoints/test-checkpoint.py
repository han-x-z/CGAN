import numpy as np
import os
import argparse
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import random
from torch.utils.data import DataLoader
import logging
from torchvision import datasets
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils

# from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar10', help='cifar10 | lsun | mnist')
parser.add_argument('--dataroot', default='./data', help='path to data')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=32, help='image size input')
parser.add_argument('--channels', type=int, default=3, help='number of channels')
parser.add_argument('--latentdim', type=int, default=100, help='size of latent vector')
parser.add_argument('--n_classes', type=int, default=10, help='number of classes in data set')
parser.add_argument('--epoch', type=int, default=200, help='number of epoch')
parser.add_argument('--lrate', type=float, default=0.0002, help='learning rate')
parser.add_argument('--beta', type=float, default=0.5, help='beta for adam optimizer')
parser.add_argument('--beta1', type=float, default=0.999, help='beta1 for adam optimizer')
parser.add_argument('--output', default='./output', help='folder to output images and model checkpoints')
parser.add_argument('--randomseed', type=int, help='seed')

opt = parser.parse_args()

img_shape = (opt.channels, opt.imageSize, opt.imageSize)

cuda = True if torch.cuda.is_available() else False

os.makedirs(opt.output, exist_ok=True)

if opt.randomseed is None:
    opt.randomseed = random.randint(1, 10000)
random.seed(opt.randomseed)
torch.manual_seed(opt.randomseed)

# preprocessing for mnist, lsun, cifar10
if opt.dataset == 'mnist':
    dataset = datasets.MNIST(root=opt.dataroot, train=True, download=True,
                             transform=transforms.Compose([transforms.Resize(opt.imageSize),
                                                           transforms.ToTensor(),
                                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

elif opt.dataset == 'lsun':
    dataset = datasets.LSUN(root=opt.dataroot, train=True, download=True,
                            transform=transforms.Compose([transforms.Resize(opt.imageSize),
                                                          transforms.CenterCrop(opt.imageSize),
                                                          transforms.ToTensor(),
                                                          transforms.Normalize((0.5,), (0.5,))]))

elif opt.dataset == 'cifar10':
    dataset = datasets.CIFAR10(root=opt.dataroot, train=True, download=True,
                               transform=transforms.Compose([transforms.Resize(opt.imageSize),
                                                             transforms.ToTensor(),
                                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True)


# building generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.label_embed = nn.Embedding(opt.n_classes, opt.n_classes)
        self.depth = 128

        def init(input, output, normalize=True):
            layers = [nn.Linear(input, output)]
            if normalize:
                layers.append(nn.BatchNorm1d(output, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.generator = nn.Sequential(

            *init(opt.latentdim + opt.n_classes, self.depth),
            *init(self.depth, self.depth * 2),
            *init(self.depth * 2, self.depth * 4),
            *init(self.depth * 4, self.depth * 8),
            nn.Linear(self.depth * 8, int(np.prod(img_shape))),
            nn.Tanh()

        )

    # torchcat needs to combine tensors
    def forward(self, noise, labels):
        gen_input = torch.cat((self.label_embed(labels), noise), -1)
        img = self.generator(gen_input)
        img = img.view(img.size(0), *img_shape)
        return img
class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()
		self.label_embed1 = nn.Embedding(opt.n_classes, opt.n_classes)
		self.dropout = 0.4
		self.depth = 512

		def init(input, output, normalize=True):
			layers = [nn.Linear(input, output)]
			if normalize:
				layers.append(nn.Dropout(self.dropout))
			layers.append(nn.LeakyReLU(0.2, inplace=True))
			return layers

		self.discriminator = nn.Sequential(
			*init(opt.n_classes+int(np.prod(img_shape)), self.depth, normalize=False),
			*init(self.depth, self.depth),
			*init(self.depth, self.depth),
			nn.Linear(self.depth, 1),
			nn.Sigmoid()
			)

	def forward(self, img, labels):
		imgs = img.view(img.size(0),-1)
		inpu = torch.cat((imgs, self.label_embed1(labels)), -1)
		validity = self.discriminator(inpu)
		return validity

generator = Generator()

discriminator = Discriminator()

# 加载已经训练好的模型参数
g_model_path = "./output/generator_epoch_199.pth"  # 替换为训练好的模型的路径
d_model_path = "./output/discriminator_epoch_199.pth"
generator.load_state_dict(torch.load(g_model_path))
discriminator.load_state_dict(torch.load(d_model_path))
# 生成噪声和标签
noise = torch.randn(opt.batchSize, opt.latentdim)
labels = torch.randint(0, opt.n_classes, (opt.batchSize,))

# 将生成器设置为评估模式
generator.eval()
discriminator.eval()
# 使用生成器生成图像
with torch.no_grad():
    generated_images = generator(noise, labels)
    # 将生成的图像输入鉴别器并转换为类别概率
with torch.no_grad():
    probabilities = discriminator(generated_images, labels)

save_path = os.path.join("tests", "test.png")
print("生成的图像的类别概率:", probabilities)

vutils.save_image(generated_images, save_path, normalize=True)

