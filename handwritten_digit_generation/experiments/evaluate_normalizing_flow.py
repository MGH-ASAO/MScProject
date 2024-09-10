import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from keras.src.utils.module_utils import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.manifold import TSNE

from handwritten_digit_generation.utils.file_utils import save_to_results
from handwritten_digit_generation.models.normalizing_flow import NormalizingFlow

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def preprocess(x):
    return x.view(x.size(0), -1)


def compute_loss(z, log_det):
    prior_ll = -0.5 * torch.sum(z ** 2, dim=1) - 0.5 * z.shape[1] * np.log(2 * np.pi)
    log_likelihood = prior_ll + log_det
    return -torch.mean(log_likelihood)


def evaluate_normalizing_flow(model_path, batch_size=256):
    # Load model
    state_dict = torch.load(model_path, map_location=device)

    # Extract model configuration
    input_dim = 784  # MNIST image size
    n_flows = len([key for key in state_dict.keys() if
                   key.startswith('flows')]) // 7  # 7 is the number of components in each flow

    print(f"Extracted config: input_dim={input_dim}, n_flows={n_flows}")

    # Create model
    model = NormalizingFlow(dim=input_dim, n_flows=n_flows).to(device)

    # Check if the model structure matches the state dict
    model_state = model.state_dict()
    if model_state.keys() != state_dict.keys():
        print("Warning: Model structure does not match the saved state dict.")
        print("Model keys:", model_state.keys())
        print("State dict keys:", state_dict.keys())
        missing_keys = set(model_state.keys()) - set(state_dict.keys())
        extra_keys = set(state_dict.keys()) - set(model_state.keys())
        if missing_keys:
            print("Missing keys in state dict:", missing_keys)
        if extra_keys:
            print("Extra keys in state dict:", extra_keys)
        return

    model.load_state_dict(state_dict)
    model.eval()

    # Prepare data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Evaluation
    total_loss = 0
    all_log_likelihoods = []
    all_latents = []
    all_labels = []

    with torch.no_grad():
        for data, labels in test_loader:
            data = preprocess(data).to(device)
            z, log_det = model(data)
            loss = compute_loss(z, log_det)
            total_loss += loss.item() * data.size(0)
            all_log_likelihoods.extend((-loss.item() * data.size(0) / np.prod(data.shape[1:])).repeat(data.size(0)))
            all_latents.append(z.cpu().numpy())
            all_labels.extend(labels.numpy())

    avg_loss = total_loss / len(test_dataset)
    avg_log_likelihood = np.mean(all_log_likelihoods)
    avg_bits_per_dim = -avg_log_likelihood / np.log(2)

    print(f'Average loss: {avg_loss:.4f}')
    print(f'Average log-likelihood: {avg_log_likelihood:.4f}')
    print(f'Average bits per dimension: {avg_bits_per_dim:.4f}')

    # Plot log-likelihood distribution
    plt.figure(figsize=(10, 6))
    plt.hist(all_log_likelihoods, bins=50)
    plt.title('Distribution of Log-Likelihoods')
    plt.xlabel('Log-Likelihood')
    plt.ylabel('Count')
    plt.savefig(save_to_results('log_likelihood_distribution.png', subdirectory='normalizing_flow'))
    plt.close()

    # Generate samples
    n_samples = 100
    z = torch.randn(n_samples, input_dim).to(device)
    with torch.no_grad():
        generated_samples = model.inverse(z)
    generated_samples = generated_samples.view(-1, 1, 28, 28)
    generated_samples = (generated_samples - generated_samples.min()) / (
                generated_samples.max() - generated_samples.min())

    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.imshow(generated_samples[i, 0].cpu().numpy(), cmap='gray')
        plt.axis('off')
    plt.savefig(save_to_results('generated_samples.png', subdirectory='normalizing_flow'))
    plt.close()

    n_samples = 10
    z1 = torch.randn(1, input_dim, device=device)
    z2 = torch.randn(1, input_dim, device=device)
    alphas = torch.linspace(0, 1, steps=n_samples).to(device)

    interpolated_samples = []
    with torch.no_grad():
        for alpha in alphas:
            z = alpha * z1 + (1 - alpha) * z2
            samples = model.inverse(z)
            interpolated_samples.append(samples)

    interpolated_samples = torch.cat(interpolated_samples, dim=0)

    # 保存插值图像
    grid = torchvision.utils.make_grid(interpolated_samples.view(-1, 1, 28, 28), nrow=n_samples, normalize=True)
    torchvision.utils.save_image(grid, save_to_results('normalizing_flow_latent_interpolation.png',
                                                       subdirectory='normalizing_flow'))


if __name__ == "__main__":
    model_path = save_to_results('best_normalizing_flow_model.pth', subdirectory='normalizing_flow')
    evaluate_normalizing_flow(model_path)