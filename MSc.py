import os
import sys
import torch
import torch.nn.functional as F
import torchvision
from scipy.stats import entropy
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from scipy import linalg
from tqdm import tqdm


project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from handwritten_digit_generation.models.gan import Generator as GANGenerator
from handwritten_digit_generation.models.vae import ConvVAE
from handwritten_digit_generation.models.normalizing_flow import NormalizingFlow
from handwritten_digit_generation.models.diffusion import DiffusionModel
from handwritten_digit_generation.utils.file_utils import save_to_results


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_inception_model():
    inception_model = torchvision.models.inception_v3(weights=torchvision.models.Inception_V3_Weights.IMAGENET1K_V1)
    inception_model.fc = torch.nn.Identity()
    inception_model = inception_model.to(device)
    inception_model.eval()
    return inception_model


def calculate_fid(real_features, fake_features):
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)

    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = linalg.sqrtm(sigma1.dot(sigma2))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def calculate_inception_score(pred, num_splits=10, eps=1e-16):
    scores = []
    for i in range(num_splits):
        part = pred[i * (len(pred) // num_splits): (i + 1) * (len(pred) // num_splits), :]
        py = np.mean(part, axis=0)
        scores.append(np.exp(np.mean(np.sum(part * (np.log(part + eps) - np.log(py + eps)), axis=1))))
    return np.mean(scores), np.std(scores)


def generate_samples(model, model_type, num_samples=10000):
    model.eval()
    samples = []
    with torch.no_grad():
        for _ in tqdm(range(num_samples // 100), desc=f"Generating {model_type} samples"):
            if model_type == 'gan':
                z = torch.randn(100, 100).to(device)
                sample = model(z)
            elif model_type == 'vae':
                z = torch.randn(100, model.latent_dim).to(device)
                sample = model.decode(z)
            elif model_type == 'normalizing_flow':
                z = torch.randn(100, 784).to(device)
                sample = model.inverse(z)
                sample = sample.view(-1, 1, 28, 28)
            elif model_type == 'diffusion':
                sample = model.sample(100, device)

            samples.append(sample.cpu())

    samples = torch.cat(samples, dim=0)
    return samples


def preprocess_samples(samples, model_type):
    if model_type == 'diffusion':
        samples = (samples + 1) / 2  # 假设 Diffusion 模型输出范围是 [-1, 1]
    samples = torch.clamp(samples, 0, 1)
    return samples


def preprocess_for_inception(images):
    images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
    images = images.repeat(1, 3, 1, 1)
    images = images * 2 - 1  # 将 [0, 1] 范围转换为 [-1, 1]
    return images


def debug_output(tensor, name):
    print(f"{name} - Shape: {tensor.shape}")
    print(f"{name} - Min: {tensor.min().item():.4f}, Max: {tensor.max().item():.4f}")
    print(f"{name} - Mean: {tensor.mean().item():.4f}, Std: {tensor.std().item():.4f}")


def evaluate_models():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_dataset = datasets.MNIST(root=os.path.join(project_root, 'data'), train=False, download=True,
                                  transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

    inception_model = load_inception_model()

    latent_dim = 100
    img_shape = (1, 28, 28)

    gan_generator = GANGenerator(latent_dim, img_shape).to(device)
    gan_generator.load_state_dict(
        torch.load(save_to_results('gan_generator.pth', subdirectory='gan'), map_location=device))

    vae = ConvVAE(latent_dim=32).to(device)
    vae.load_state_dict(torch.load(save_to_results('vae_model.pth', subdirectory='vae'), map_location=device))

    normalizing_flow = NormalizingFlow(dim=784, n_flows=16).to(device)
    normalizing_flow.load_state_dict(
        torch.load(save_to_results('best_normalizing_flow_model.pth', subdirectory='normalizing_flow'),
                   map_location=device))

    diffusion = DiffusionModel(input_channels=1, time_dim=256, hidden_dim=64, device=device).to(device)
    diffusion.load_state_dict(
        torch.load(save_to_results('diffusion_model.pth', subdirectory='diffusion'), map_location=device))

    models = {
        'gan': gan_generator,
        'vae': vae,
        'normalizing_flow': normalizing_flow,
        'diffusion': diffusion
    }

    real_features = []
    for images, _ in tqdm(test_loader, desc="Processing real images"):
        images = images.to(device)
        images = preprocess_for_inception(images)
        with torch.no_grad():
            features = inception_model(images).cpu().numpy()
        real_features.append(features)
    real_features = np.concatenate(real_features, axis=0)

    results = {}

    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")


        generated_samples = generate_samples(model, model_name)
        generated_samples = preprocess_samples(generated_samples, model_name)

        debug_output(generated_samples, f"{model_name} generated samples")

        # Calculate the characteristics of the generated sample
        fake_features = []
        fake_preds = []
        for i in tqdm(range(0, len(generated_samples), 100), desc=f"Processing {model_name} generated images"):
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

        results[model_name] = {
            'FID': fid,
            'IS_mean': is_mean,
            'IS_std': is_std
        }

        # Save the generated sample
        torchvision.utils.save_image(generated_samples[:100],
                                     save_to_results(f'{model_name}_samples.png', subdirectory=model_name),
                                     nrow=10, normalize=True)

    # print results
    for model_name, scores in results.items():
        print(f"\nResults for {model_name}:")
        print(f"FID: {scores['FID']:.2f}")
        print(f"Inception Score: {scores['IS_mean']:.2f} ± {scores['IS_std']:.2f}")

    # Save results to file
    with open(save_to_results('evaluation_results.txt', subdirectory='evaluation'), 'w') as f:
        for model_name, scores in results.items():
            f.write(f"Results for {model_name}:\n")
            f.write(f"FID: {scores['FID']:.2f}\n")
            f.write(f"Inception Score: {scores['IS_mean']:.2f} ± {scores['IS_std']:.2f}\n\n")


if __name__ == "__main__":
    evaluate_models()