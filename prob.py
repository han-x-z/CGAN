import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from resnet18_32x32 import ResNet18_32x32

# 加载训练好的模型
model = ResNet18_32x32()
model.load_state_dict(torch.load('./resnet/model.pth'))
output = "./output_prob"
# 加载CIFAR-10数据集
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                         shuffle=False, num_workers=2)

# 定义CIFAR-10类别标签
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 对测试集进行预测
model.eval()
tensor_list = []
all_generated_probs = []

with torch.no_grad():
    for images, labels in testloader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.softmax(outputs, dim=1)
        probabilities_cifar10 = probabilities[:, :10]
        # 归一化概率分布
        normalized_probabilities_cifar10 = probabilities_cifar10 / torch.sum(probabilities_cifar10, dim=1, keepdim=True)
        tensor_list.append(normalized_probabilities_cifar10) #训练数据集

print("Size of tensor_list:", len(tensor_list))
#定义CGAN

class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super(Generator, self).__init__()
        self.label_embed = nn.Embedding(num_classes, latent_dim)
        self.generator = nn.Sequential(
            nn.Linear(latent_dim * 2, 128),  # 将噪音维度加入生成器输入维度
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, noise, labels):
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        class_probs = self.model(gen_input)
        return class_probs
class Discriminator(nn.Module):
    def __init__(self, num_classes):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(num_classes, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, class_probs):
        validity = self.discriminator(class_probs)
        return validity

# 定义参数

latent_dim = 100
num_classes = 10
lr = 0.0002
batch_size = 1
num_epochs = 20
# 创建生成器和判别器实例
generator = Generator(latent_dim, num_classes)
discriminator = Discriminator(num_classes)

# 定义损失函数和优化器
adversarial_loss = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=lr)
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

# 训练生成对抗网络
for epoch in range(num_epochs):
    for index in range (len(tensor_list)):
        real_tensor = tensor_list[index]
        # 生成器生成类别概率分布
        gen_labels = torch.randint(0, num_classes, (batch_size,))
        noise = torch.randn(batch_size, latent_dim)
        gen_class_probs = generator(gen_labels, noise)
        # 训练判别器
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        optimizer_D.zero_grad()

        # 判别器判断真实类别概率分布
        real_validity = discriminator(real_tensor)
        real_loss = adversarial_loss(real_validity, real_labels)

        # 判别器判断生成的类别概率分布
        fake_validity = discriminator(gen_class_probs.detach())
        fake_loss = adversarial_loss(fake_validity, fake_labels)

        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        # 训练生成器
        optimizer_G.zero_grad()

        # 生成器生成类别概率分布，并判别器判断生成的类别概率分布
        gen_validity = discriminator(gen_class_probs)
        g_loss = adversarial_loss(gen_validity, real_labels)

        g_loss.backward()
        optimizer_G.step()
    all_generated_probs.append(gen_class_probs.detach().cpu().numpy())
    # 打印损失
    print(f"[Epoch {epoch + 1}/{num_epochs}] Generated Probs: {gen_class_probs.detach().cpu().numpy()}")
    print(f"[Epoch {epoch + 1}/{num_epochs}] D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")
    torch.save(generator.state_dict(), '%s/generator_epoch_%d.pth' % (output, epoch))
    torch.save(discriminator.state_dict(), '%s/discriminator_epoch_%d.pth' % (output, epoch))

all_generated_probs = np.array(all_generated_probs)
np.save('generated_probs.npy', all_generated_probs)