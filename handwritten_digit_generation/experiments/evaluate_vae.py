import os
import sys
import torch
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from handwritten_digit_generation.models.vae import ConvVAE
from handwritten_digit_generation.utils.file_utils import save_to_results
from handwritten_digit_generation.utils.visualization import plot_confusion_matrix

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

def evaluate_vae():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    test_dataset = datasets.MNIST(root=os.path.join(project_root, 'data'), train=False, download=True,
                                  transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    latent_dim = 32
    beta = 0.1
    model = ConvVAE(latent_dim, beta).to(device)
    model_path = save_to_results('vae_model.pth', subdirectory='vae')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    total_loss = 0
    total_bce = 0
    total_kld = 0
    total_mse = 0
    n_evaluated = 0

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            recon_batch, mu, logvar = model(images)
            loss, bce, kld = model.loss_function(recon_batch, images, mu, logvar)
            total_loss += loss.item()
            total_bce += bce.item()
            total_kld += kld.item()

            mse = F.mse_loss(recon_batch, images, reduction='sum')
            total_mse += mse.item()

            n_evaluated += images.size(0)

    avg_loss = total_loss / n_evaluated
    avg_bce = total_bce / n_evaluated
    avg_kld = total_kld / n_evaluated
    avg_mse = total_mse / n_evaluated
    print(f'Average loss on test set: {avg_loss:.4f}')
    print(f'Average BCE on test set: {avg_bce:.4f}')
    print(f'Average KLD on test set: {avg_kld:.4f}')
    print(f'Average MSE on test set: {avg_mse:.4f}')

    n_samples = 25
    z = torch.randn(n_samples, latent_dim).to(device)
    with torch.no_grad():
        generated_images = model.decode(z)

    samples_path = save_to_results('vae_generated_images.png', subdirectory='vae')
    torchvision.utils.save_image(generated_images.cpu(), samples_path, nrow=5, normalize=True)


    result_path = save_to_results('vae_evaluation_result.txt', subdirectory='vae')

    z1 = torch.randn(1, model.latent_dim, device=device)
    z2 = torch.randn(1, model.latent_dim, device=device)
    alphas = torch.linspace(0, 1, steps=10).to(device)

    interpolated_images = []
    for alpha in alphas:
        z = alpha * z1 + (1 - alpha) * z2
        with torch.no_grad():
            img = model.decode(z)
        interpolated_images.append(img)

    interpolated_images = torch.cat(interpolated_images, dim=0)

    # 保存插值图像
    grid = torchvision.utils.make_grid(interpolated_images, nrow=10, normalize=True)
    torchvision.utils.save_image(grid, save_to_results('vae_latent_interpolation.png', subdirectory='vae'))

    with open(result_path, 'w') as f:
        f.write(f'Average loss: {avg_loss:.4f}\n')
        f.write(f'Average BCE: {avg_bce:.4f}\n')
        f.write(f'Average KLD: {avg_kld:.4f}\n')
        f.write(f'Average MSE: {avg_mse:.4f}\n')
        f.write(f"Generated images saved to {samples_path}\n")

    print(f"Evaluation results saved to {result_path}")

    return avg_loss, avg_bce, avg_kld, avg_mse

if __name__ == "__main__":
    evaluate_vae()