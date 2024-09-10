# handwritten_digit_generation/experiments/train_diffusion.py

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse
import logging
from datetime import datetime
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import CosineAnnealingLR

# 获取项目根目录的路径并添加到 Python 路径中
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from handwritten_digit_generation.models.diffusion import DiffusionModel, forward_diffusion_sample, \
    linear_beta_schedule
from handwritten_digit_generation.utils.file_utils import save_to_results, get_project_root
from handwritten_digit_generation.utils.training_utils import to_cpu, print_progress, validate, T

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)



# 参数解析
parser = argparse.ArgumentParser(description='Train Diffusion model')
parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
parser.add_argument('--save_interval', type=int, default=5, help='save model every n epochs')
args = parser.parse_args()

# 配置
CONFIG = {
    'batch_size': args.batch_size,
    'lr': 1e-4,
    'n_epochs': 500,
    'save_interval': 10,
    'device': torch.device("mps" if torch.backends.mps.is_available() else "cpu"),
    'input_dim': 784
}


def main():
    logging.info(f"Starting Diffusion training script at {datetime.now()}")
    logging.info(f"Using device: {CONFIG['device']}")
    logging.info(f"Configuration: {CONFIG}")

    # 数据加载
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root=os.path.join(project_root, 'data'), train=True, download=True,
                                   transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)

    # 初始化模型
    model = DiffusionModel(input_channels=1, hidden_dim=64).to(CONFIG['device'])
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG['n_epochs'], eta_min=1e-6)

    # 检查点路径
    checkpoint_path = save_to_results('diffusion_checkpoint.pth', subdirectory='diffusion')

    # 最终模型路径
    final_model_path = save_to_results('diffusion_model.pth', subdirectory='diffusion')

    # 恢复训练（如果存在检查点）
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=CONFIG['device'])
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1
            logging.info(f"Resuming training from epoch {start_epoch}")
        except RuntimeError as e:
            logging.warning(f"Failed to load checkpoint: {e}")
            logging.info("Starting training from scratch")
            start_epoch = 0

    # 训练循环
    try:
        for epoch in range(start_epoch, CONFIG['n_epochs']):
            model.train()
            total_loss = 0
            for i, (imgs, _) in enumerate(train_loader):
                imgs = imgs.to(CONFIG['device'])

                optimizer.zero_grad()

                # Sample t uniformly for every example in the batch
                t = torch.randint(0, T, (imgs.shape[0],), device=CONFIG['device']).long()

                # Get noisy image
                x_noisy, noise = forward_diffusion_sample(imgs, t, CONFIG['device'])

                # 确保 x_noisy 和 noise 的形状一致
                if x_noisy.dim() == 2:
                    x_noisy = x_noisy.view(-1, 1, 28, 28)
                if noise.dim() == 2:
                    noise = noise.view(-1, 1, 28, 28)

                # Get predicted noise
                noise_pred = model(x_noisy, t)

                # 确保 noise 和 noise_pred 的形状一致
                noise = noise.view(noise.shape[0], -1)
                noise_pred = noise_pred.view(noise_pred.shape[0], -1)

                # Compute loss
                loss = F.mse_loss(noise, noise_pred)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                if i % 100 == 0:
                    print_progress(epoch, CONFIG['n_epochs'], i, len(train_loader), loss.item())

            avg_loss = total_loss / len(train_loader)
            logging.info(f'Epoch {epoch + 1}/{CONFIG["n_epochs"]}, Average loss: {avg_loss:.4f}')

            # 学习率调度
            scheduler.step()

            # 保存模型
            if (epoch + 1) % CONFIG['save_interval'] == 0:
                with to_cpu(model) as m:
                    torch.save(m.state_dict(),
                               save_to_results(f'diffusion_model_epoch_{epoch + 1}.pth', subdirectory='diffusion'))

            # 保存检查点
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, checkpoint_path)

        # 保存最终模型
        torch.save(model.state_dict(), final_model_path)
        logging.info(f"Final model saved to {final_model_path}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        # 保存检查点
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, checkpoint_path)

    # 训练完成
    logging.info(f"Diffusion training completed at {datetime.now()}")


if __name__ == "__main__":
    main()