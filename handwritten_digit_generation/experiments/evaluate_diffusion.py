import os
import sys
import torch
from keras.src.utils.module_utils import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from scipy import linalg
from tqdm import tqdm
import torch.nn.functional as F

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from handwritten_digit_generation.models.diffusion import DiffusionModel
from handwritten_digit_generation.utils.file_utils import save_to_results
from handwritten_digit_generation.utils.training_utils import T

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


# Load pre-trained Inception v3 model
def load_inception_model():
    inception_model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
    inception_model.fc = torch.nn.Identity()
    inception_model = inception_model.to(device)
    inception_model.eval()
    return inception_model


# Calculate FID
def calculate_fid(real_features, fake_features):
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)

    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = linalg.sqrtm(sigma1.dot(sigma2))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


# Calculate Inception Score
def calculate_inception_score(preds, n_split=10, eps=1E-16):
    scores = []
    for i in range(n_split):
        part = preds[(i * preds.shape[0] // n_split):((i + 1) * preds.shape[0] // n_split), :]
        kl = part * (np.log(part + eps) - np.log(np.expand_dims(np.mean(part, 0), 0) + eps))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)


def sample_from_model(model, n_samples=10000, device=device):
    model.eval()
    samples = []
    with torch.no_grad():
        for _ in tqdm(range(n_samples // 100)):
            x = torch.randn((100, 1, 28, 28)).to(device)
            for i in reversed(range(T)):
                t = torch.full((100,), i, device=device, dtype=torch.long)
                x = model.p_sample(x, t)
            samples.append(x.cpu())
    return torch.cat(samples, dim=0)


def preprocess_for_inception(images):
    images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
    images = images.repeat(1, 3, 1, 1)  # Convert grayscale to RGB
    return images


def evaluate_diffusion():
    # Load test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST(root=os.path.join(project_root, 'data'), train=False, download=True,
                                  transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

    # Load model
    model = DiffusionModel(input_channels=1, time_dim=256, hidden_dim=64, device=device).to(device)
    model_path = save_to_results('diffusion_model.pth', subdirectory='diffusion')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load Inception model
    inception_model = load_inception_model()

    # Calculate real features
    real_features = []
    for images, _ in tqdm(test_loader, desc="Processing real images"):
        images = images.to(device)
        images = preprocess_for_inception(images)
        with torch.no_grad():
            features = inception_model(images).cpu().numpy()
        real_features.append(features)
    real_features = np.concatenate(real_features, axis=0)

    # Generate samples and calculate fake features
    generated_samples = sample_from_model(model)
    fake_features = []
    fake_preds = []
    for i in tqdm(range(0, len(generated_samples), 100), desc="Processing generated images"):
        batch = generated_samples[i:i + 100].to(device)
        batch = preprocess_for_inception(batch)
        with torch.no_grad():
            features = inception_model(batch).cpu().numpy()
            preds = F.softmax(inception_model(batch), dim=1).cpu().numpy()
        fake_features.append(features)
        fake_preds.append(preds)
    fake_features = np.concatenate(fake_features, axis=0)
    fake_preds = np.concatenate(fake_preds, axis=0)

    # Calculate FID
    fid = calculate_fid(real_features, fake_features)

    # Calculate Inception Score
    is_mean, is_std = calculate_inception_score(fake_preds)

    # Save results
    result_path = save_to_results('diffusion_evaluation_result.txt', subdirectory='diffusion')

    n_steps = 10
    x1 = torch.randn(1, 1, 28, 28, device=device)
    x2 = torch.randn(1, 1, 28, 28, device=device)

    interpolated_images = []
    with torch.no_grad():
        for i in range(n_steps):
            t = torch.full((1,), i * 100, device=device, dtype=torch.long)
            x = (n_steps - i) / n_steps * x1 + i / n_steps * x2
            img = model.p_sample(x, t)
            interpolated_images.append(img)

    interpolated_images = torch.cat(interpolated_images, dim=0)

    # 保存插值图像
    grid = torchvision.utils.make_grid(interpolated_images, nrow=n_steps, normalize=True)
    torchvision.utils.save_image(grid, save_to_results('diffusion_noise_interpolation.png', subdirectory='diffusion'))

    with open(result_path, 'w') as f:
        f.write(f'FID: {fid:.4f}\n')
        f.write(f'Inception Score: {is_mean:.4f} ± {is_std:.4f}\n')

    print(f"Evaluation results saved to {result_path}")

    return fid, is_mean, is_std


if __name__ == "__main__":
    evaluate_diffusion()



# handwritten_digit_generation/experiments/evaluate_diffusion.py

# import os
# import sys
# import torch
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
# import numpy as np
# import matplotlib.pyplot as plt
# from skimage.metrics import structural_similarity as ssim
#
# from handwritten_digit_generation.utils.training_utils import T
#
# project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(project_root)
#
# from handwritten_digit_generation.models.diffusion import DiffusionModel, betas, alphas_cumprod, posterior_variance
# from handwritten_digit_generation.utils.file_utils import save_to_results, get_project_root
# from handwritten_digit_generation.utils.visualization import plot_confusion_matrix
#
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# print(f"Using device: {device}")
#
# def sample_from_model(model, n_samples=16, device=device):
#     model.eval()
#     with torch.no_grad():
#         x = torch.randn((n_samples, 1, 28, 28)).to(device)
#         for i in reversed(range(T)):
#             t = torch.full((n_samples,), i, device=device, dtype=torch.long)
#             noise_pred = model(x, t)
#             if noise_pred.dim() == 2:
#                 noise_pred = noise_pred.view(-1, 1, 28, 28)
#             x = x - betas[i] * noise_pred / torch.sqrt(1 - alphas_cumprod[i])
#             if i > 0:
#                 noise = torch.randn_like(x)
#                 x = x + torch.sqrt(posterior_variance[i]) * noise
#     model.train()
#     x = torch.clamp(x, -1, 1)
#     return x.cpu()
#
#
# def visualize_reconstruction(original, reconstructed, n_samples=5):
#     fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 2, 4))
#     for i in range(n_samples):
#         axes[0, i].imshow(original[i].squeeze().cpu().numpy(), cmap='gray')
#         axes[0, i].axis('off')
#         axes[0, i].set_title('Original')
#
#         axes[1, i].imshow(reconstructed[i].squeeze().cpu().numpy(), cmap='gray')
#         axes[1, i].axis('off')
#         axes[1, i].set_title('Reconstructed')
#
#     plt.tight_layout()
#     reconstruction_path = save_to_results('diffusion_reconstruction_comparison.png', subdirectory='diffusion')
#     plt.savefig(reconstruction_path)
#     plt.close()
#     return reconstruction_path
#
#
#
# def evaluate_diffusion():
#     # 加载测试数据
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.1307,), (0.3081,))
#     ])
#     test_dataset = datasets.MNIST(root=os.path.join(project_root, 'data'), train=False, download=True,
#                                   transform=transform)
#     test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
#
#     # 加载模型
#     input_dim = 784
#     model = DiffusionModel(input_channels=1, time_dim=256, hidden_dim=64, device=device).to(device)
#     model_path = save_to_results('diffusion_model.pth', subdirectory='diffusion')
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.eval()
#
#     # 生成样本
#     n_samples = 16
#     samples = sample_from_model(model, n_samples)
#
#     # 绘制生成的样本
#     fig, axes = plt.subplots(4, 4, figsize=(10, 10))
#     for i, ax in enumerate(axes.flatten()):
#         ax.imshow(samples[i].squeeze().numpy(), cmap='gray')  # 直接使用 .numpy()
#         ax.axis('off')
#
#     plt.tight_layout()
#
#     # 保存结果
#     samples_path = save_to_results('diffusion_generated_samples.png', subdirectory='diffusion')
#     plt.savefig(samples_path)
#     plt.close()
#
#     # 计算重构误差和SSIM
#     total_mse = 0
#     total_ssim = 0
#     n_evaluated = 0
#
#     with torch.no_grad():
#         for images, _ in test_loader:
#             images = images.to(device)
#             reconstructed = sample_from_model(model, images.size(0))
#
#             # 确保值域一致
#             images = (images + 1) / 2  # 从[-1, 1]转换到[0, 1]
#             reconstructed = (reconstructed + 1) / 2
#
#             mse = torch.mean((images.cpu() - reconstructed) ** 2)
#             total_mse += mse.item() * images.size(0)
#
#             # 计算SSIM
#             for i in range(images.size(0)):
#                 orig = images[i].squeeze().cpu().numpy()
#                 recon = reconstructed[i].squeeze().numpy()
#                 total_ssim += ssim(orig, recon, data_range=1)
#
#             n_evaluated += images.size(0)
#
#     avg_mse = total_mse / n_evaluated
#     avg_ssim = total_ssim / n_evaluated
#     print(f'Average MSE: {avg_mse:.4f}')
#     print(f'Average SSIM: {avg_ssim:.4f}')
#
#     # 在评估函数中调用
#     reconstruction_path = visualize_reconstruction(images[:5], reconstructed[:5])
#
#     # 保存评估结果
#     result_path = save_to_results('diffusion_evaluation_result.txt', subdirectory='diffusion')
#     with open(result_path, 'w') as f:
#         f.write(f'Average MSE: {avg_mse:.4f}\n')
#         f.write(f'Average SSIM: {avg_ssim:.4f}\n')
#         f.write(f"Generated samples saved to {samples_path}\n")
#         f.write(f"Reconstruction comparison saved to {reconstruction_path}\n")
#
#     print(f"Evaluation results saved to {result_path}")
#
#     return avg_mse, avg_ssim
#
# if __name__ == "__main__":
#     evaluate_diffusion()