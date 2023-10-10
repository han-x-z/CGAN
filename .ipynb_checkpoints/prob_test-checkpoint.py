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


device = torch.device("cuda" if torch.cuda.is_available() else False)

# 加载训练好的模型
model = ResNet18_32x32().to(device)
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

# 加载训练集和测试集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=4)


# 定义CIFAR-10类别标签
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 对测试集进行预测
model.eval()
tensor_list = []
all_generated_probs = []

with torch.no_grad():
    for images, labels in testloader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.softmax(outputs, dim=1) 
        tensor_list.append((predicted, probabilities)) #训练数据集

print("Size of tensor_list:", len(tensor_list))
for i in range(5):
    predicted, probabilities = tensor_list[i]
    print(f"Element {i+1}:")
    print("Predicted Label:", predicted)
    print("Probabilities:", probabilities)
    print()
#参数加载
latent_dim = 100
lr = 0.0002
batch_size = 1
num_epochs = 100
num_classes = 10
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

    def forward(self, labels, noise):
        gen_input = self.label_embed(labels)
        gen_input_with_noise = torch.cat((gen_input, noise), dim=1)  # 将噪音和标签嵌入向量连接起来
        class_probs = self.generator(gen_input_with_noise)
        return class_probs



class Discriminator(nn.Module):
    def __init__(self, num_classes):
        super(Discriminator, self).__init__()
        self.label_embed = nn.Embedding(num_classes, num_classes)
        self.discriminator = nn.Sequential(
            nn.Linear(num_classes * 2, 512),  # 输入维度为类别数乘以2
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, class_probs, labels):
        label_embed = self.label_embed(labels)
        input_tensor = torch.cat((class_probs, label_embed), dim=1)
        validity = self.discriminator(input_tensor)
        return validity


# 创建生成器和判别器实例
generator = Generator(latent_dim, num_classes).to(device)
discriminator = Discriminator(num_classes).to(device)

generator.load_state_dict(torch.load('generator4.pth'))
discriminator.load_state_dict(torch.load('discriminato4.pth'))


# 定义损失函数和优化器
adversarial_loss = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=lr)
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)


d_losses = []  # 存储判别器损失值
g_losses = []  # 存储生成器损失值
# 训练生成对抗网络
for epoch in range(num_epochs):
    for index in range (len(tensor_list)):
        predicted, probabilities = tensor_list[index]
        # 生成器生成类别概率分布
        gen_labels = torch.randint(0, num_classes, (batch_size,)).to(device)
        noise = torch.randn(batch_size, latent_dim).to(device)
        gen_class_probs = generator(gen_labels, noise)
        
        # 训练判别器
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        
        optimizer_D.zero_grad()
        # 判别器判断真实类别概率分布
        real_validity = discriminator(probabilities.to(device), predicted.to(device))
        real_loss = adversarial_loss(real_validity, real_labels)

        # 判别器判断生成的类别概率分布
        fake_validity = discriminator(gen_class_probs.detach(), gen_labels)
        fake_loss = adversarial_loss(fake_validity, fake_labels)

        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()
        
        
        # 训练生成器
        optimizer_G.zero_grad()

        # 生成器生成类别概率分布，并判别器判断生成的类别概率分布
        gen_validity = discriminator(gen_class_probs, gen_labels)
        g_loss = adversarial_loss(gen_validity, real_labels)

        g_loss.backward()
        optimizer_G.step()

    d_losses.append(d_loss.item())
    g_losses.append(g_loss.item())
    print(f"[Epoch {epoch + 1}/{num_epochs}] Label: {gen_labels.item()}")
    print(f"[Epoch {epoch + 1}/{num_epochs}] Generated Probs: {gen_class_probs.detach().cpu().numpy()}")
    print(f"[Epoch {epoch + 1}/{num_epochs}] D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")
# 保存生成器和判别器的状态字典
torch.save(generator.state_dict(), 'generator5.pth')
torch.save(discriminator.state_dict(), 'discriminato5.pth')