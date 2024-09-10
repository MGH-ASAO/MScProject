import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as D
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 1. 数据加载
transform = transforms.Compose([transforms.ToTensor()])
mnist_dataset = MNIST(root='./data', train=True, transform=transform, download=True)
dataloader = DataLoader(mnist_dataset, batch_size=128, shuffle=True)


# 2. 构建正常化流模型
class CouplingLayer(nn.Module):
    def __init__(self, num_features):
        super(CouplingLayer, self).__init__()
        self.scale_net = nn.Sequential(
            nn.Linear(num_features // 2, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_features // 2),
            nn.Tanh()
        )
        self.translate_net = nn.Sequential(
            nn.Linear(num_features // 2, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_features // 2)
        )

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        s = self.scale_net(x1)
        t = self.translate_net(x1)
        y1 = x1
        y2 = x2 * torch.exp(s) + t
        y = torch.cat([y1, y2], dim=1)
        log_det_jacobian = s.sum(-1)
        return y, log_det_jacobian

    def inverse(self, y):
        y1, y2 = y.chunk(2, dim=1)
        s = self.scale_net(y1)
        t = self.translate_net(y1)
        x1 = y1
        x2 = (y2 - t) * torch.exp(-s)
        x = torch.cat([x1, x2], dim=1)
        return x


class RealNVP(nn.Module):
    def __init__(self, num_layers=24, num_features=784):  # 28*28=784
        super(RealNVP, self).__init__()

        self.num_layers = num_layers
        self.coupling_layers = nn.ModuleList([CouplingLayer(num_features) for _ in range(num_layers)])
        self.prior = D.Normal(torch.zeros(num_features), torch.ones(num_features))

    def forward(self, x):
        log_det_jacobian = 0
        for layer in self.coupling_layers:
            x, log_det_j = layer(x)
            log_det_jacobian += log_det_j
        return x, log_det_jacobian

    def inverse(self, z):
        for layer in reversed(self.coupling_layers):
            z = layer.inverse(z)
        return z

    def log_prob(self, x):
        z, log_det_j = self.forward(x)
        log_prob = self.prior.log_prob(z).sum(-1) + log_det_j
        return log_prob


# 3. 模型训练
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = RealNVP().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def train(model, dataloader, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            x, _ = batch
            x = x.view(x.size(0), -1).to(device)  # 展平图像到784维

            # Ensure prior tensor is on the same device
            model.prior.loc = model.prior.loc.to(device)
            model.prior.scale = model.prior.scale.to(device)

            optimizer.zero_grad()
            log_prob = model.log_prob(x)
            loss = -log_prob.mean()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader)}')


train(model, dataloader, optimizer, num_epochs=100)


# 4. 生成新图像
def generate_images(model, num_images=10):
    model.eval()
    z = model.prior.sample((num_images,)).to(device)
    with torch.no_grad():
        samples = model.inverse(z).cpu().view(-1, 1, 28, 28)
    return samples


# 生成10张手写数字图片
samples = generate_images(model, num_images=10)

# 显示生成的图片
for i, sample in enumerate(samples):
    plt.subplot(2, 5, i + 1)
    plt.imshow(sample.squeeze(), cmap='gray')
    plt.axis('off')
plt.show()