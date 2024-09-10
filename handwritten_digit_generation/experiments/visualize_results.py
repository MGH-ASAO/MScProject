# handwritten_digit_generation/experiments/visualize_results.py

import os
import sys
import torch
from torchvision import datasets, transforms

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from handwritten_digit_generation.experiments.evaluate_cnn import evaluate_cnn
from handwritten_digit_generation.experiments.evaluate_diffusion import evaluate_diffusion
from handwritten_digit_generation.experiments.evaluate_gan import evaluate_gan
from handwritten_digit_generation.experiments.evaluate_normalizing_flow import evaluate_normalizing_flow
from handwritten_digit_generation.experiments.evaluate_vae import evaluate_vae
from handwritten_digit_generation.utils.visualization import plot_comparison
from handwritten_digit_generation.utils.file_utils import save_to_results
import matplotlib.pyplot as plt
import torch
from handwritten_digit_generation.utils.file_utils import save_to_results

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# print(f"Using device: {device}")





def get_real_images():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST(root=os.path.join(project_root, 'data'), train=False, download=True, transform=transform)
    return torch.stack([dataset[i][0] for i in range(5)])

def main():
    real_images = get_real_images()

    # Evaluate CNN
    cnn_accuracy, cnn_class_correct, cnn_class_total = evaluate_cnn()
    print(f"CNN overall accuracy: {cnn_accuracy:.2f}%")

    # Evaluate Diffusion model
    diffusion_mse = evaluate_diffusion()
    print(f"Diffusion model average MSE: {diffusion_mse:.4f}")

    # Evaluate GAN
    gan_real_score, gan_fake_score = evaluate_gan()
    print(f"GAN average discriminator score for real images: {gan_real_score:.4f}")
    print(f"GAN average discriminator score for fake images: {gan_fake_score:.4f}")

    # Evaluate Normalizing Flow
    nf_log_likelihood, nf_mse = evaluate_normalizing_flow()
    print(f"Normalizing Flow average log-likelihood: {nf_log_likelihood:.4f}")
    print(f"Normalizing Flow average MSE: {nf_mse:.4f}")

    # Evaluate VAE
    vae_loss, vae_mse = evaluate_vae()
    print(f"VAE average loss: {vae_loss:.4f}")
    print(f"VAE average MSE: {vae_mse:.4f}")

    # Visual comparison
    plot_comparison(real_images,
                    save_to_results('diffusion_samples.png', subdirectory='diffusion'),
                    save_to_results('gan_generated_images.png', subdirectory='gan'),
                    save_to_results('normalizing_flow_samples.png', subdirectory='normalizing_flow'),
                    save_to_results('vae_generated_images.png', subdirectory='vae'),
                    "Real vs Generated Images Comparison",
                    save_to_results('model_comparison.png'))

if __name__ == "__main__":
    main()