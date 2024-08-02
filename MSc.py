import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import numpy as np
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

class RealNVPCouplingLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(RealNVPCouplingLayer, self).__init__()
        self.scale_net = nn.Sequential(
            nn.Linear(input_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim // 2)
        )
        self.translate_net = nn.Sequential(
            nn.Linear(input_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim // 2)
        )
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, reverse=False):
        x_a, x_b = x.chunk(2, dim=1)
        if reverse:
            scale = self.scale_net(x_a)
            translate = self.translate_net(x_a)
            x_b = (x_b - translate) * torch.exp(-scale)
        else:
            scale = self.scale_net(x_a)
            translate = self.translate_net(x_a)
            x_b = x_b * torch.exp(scale) + translate
        return torch.cat([x_a, x_b], dim=1), scale

class RealNVP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_coupling_layers):
        super(RealNVP, self).__init__()
        self.coupling_layers = nn.ModuleList([RealNVPCouplingLayer(input_dim, hidden_dim) for _ in range(num_coupling_layers)])

    def forward(self, x, reverse=False):
        log_det_jacobian = 0
        if reverse:
            for layer in reversed(self.coupling_layers):
                x, scale = layer(x, reverse)
                log_det_jacobian -= scale.sum(dim=1)
        else:
            for layer in self.coupling_layers:
                x, scale = layer(x)
                log_det_jacobian += scale.sum(dim=1)
        return x, log_det_jacobian

# Update the transformation pipeline
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),  # Normalize the images to [-1, 1]
    transforms.Lambda(lambda x: x.view(-1))  # Flatten the images
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

def loss_fn(z, log_det_jacobian):
    prior_log_prob = -0.5 * torch.sum(z ** 2 + torch.log(torch.tensor(2 * torch.pi)), dim=1)
    return -(prior_log_prob + log_det_jacobian).mean()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RealNVP(input_dim=784, hidden_dim=512, num_coupling_layers=8).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)  # Add weight decay (L2 regularization)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # Add learning rate scheduler

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        z, log_det_jacobian = model(data)
        loss = loss_fn(z, log_det_jacobian)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print(f'Epoch {epoch}, Loss: {train_loss / len(train_loader.dataset)}')
    scheduler.step()

def test():
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            z, log_det_jacobian = model(data)
            loss = loss_fn(z, log_det_jacobian)
            test_loss += loss.item()
    print(f'Test Loss: {test_loss / len(test_loader.dataset)}')

for epoch in range(1, 31):  # Increase the number of epochs
    train(epoch)

test()
