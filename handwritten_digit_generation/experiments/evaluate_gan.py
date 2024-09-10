# handwritten_digit_generation/experiments/evaluate_gan.py

import os
import sys
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from handwritten_digit_generation.models.gan import Generator, Discriminator
from handwritten_digit_generation.utils.file_utils import save_to_results, get_project_root
from handwritten_digit_generation.utils.visualization import plot_confusion_matrix

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


def evaluate_gan():
    # 加载测试数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_dataset = datasets.MNIST(root=os.path.join(project_root, 'data'), train=False, download=True,
                                  transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # 加载模型
    latent_dim = 100
    img_shape = (1, 28, 28)
    generator = Generator(latent_dim, img_shape).to(device)
    discriminator = Discriminator(img_shape).to(device)

    generator_path = save_to_results('gan_generator.pth', subdirectory='gan')
    discriminator_path = save_to_results('gan_discriminator.pth', subdirectory='gan')

    generator.load_state_dict(torch.load(generator_path, map_location=device, weights_only=True))
    discriminator.load_state_dict(torch.load(discriminator_path, map_location=device, weights_only=True))

    generator.eval()
    discriminator.eval()

    # 生成图像
    n_samples = 25
    z = torch.randn(n_samples, latent_dim).to(device)
    with torch.no_grad():
        generated_images = generator(z)

    # 保存生成的图像
    grid = torchvision.utils.make_grid(generated_images.cpu(), nrow=5, normalize=True)
    samples_path = save_to_results('gan_generated_images.png', subdirectory='gan')
    torchvision.utils.save_image(grid, samples_path)

    # 评估生成器和判别器
    total_fake_score = 0
    total_real_score = 0
    n_evaluated = 0

    with torch.no_grad():
        for real_images, _ in test_loader:
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            # 评估真实图像
            real_score = discriminator(real_images).mean().item()
            total_real_score += real_score * batch_size

            # 生成假图像并评估
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_images = generator(z)
            fake_score = discriminator(fake_images).mean().item()
            total_fake_score += fake_score * batch_size

            n_evaluated += batch_size

    avg_real_score = total_real_score / n_evaluated
    avg_fake_score = total_fake_score / n_evaluated

    print(f'Average discriminator score for real images: {avg_real_score:.4f}')
    print(f'Average discriminator score for fake images: {avg_fake_score:.4f}')

    z1 = torch.randn(1, latent_dim, device=device)
    z2 = torch.randn(1, latent_dim, device=device)
    alphas = torch.linspace(0, 1, steps=10).to(device)

    interpolated_images = []
    for alpha in alphas:
        z = alpha * z1 + (1 - alpha) * z2
        with torch.no_grad():
            img = generator(z)
        interpolated_images.append(img)

    interpolated_images = torch.cat(interpolated_images, dim=0)

    # 保存插值图像
    grid = torchvision.utils.make_grid(interpolated_images, nrow=10, normalize=True)
    torchvision.utils.save_image(grid, save_to_results('gan_latent_interpolation.png', subdirectory='gan'))

    # 保存评估结果
    result_path = save_to_results('gan_evaluation_result.txt', subdirectory='gan')
    with open(result_path, 'w') as f:
        f.write(f'Average discriminator score for real images: {avg_real_score:.4f}\n')
        f.write(f'Average discriminator score for fake images: {avg_fake_score:.4f}\n')
        f.write(f"Generated images saved to {samples_path}\n")

    print(f"Evaluation results saved to {result_path}")

    return avg_real_score, avg_fake_score


if __name__ == "__main__":
    evaluate_gan()