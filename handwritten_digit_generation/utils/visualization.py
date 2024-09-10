# handwritten_digit_generation/utils/visualization.py

import matplotlib.pyplot as plt
import torch
import numpy as np
from torchvision.utils import make_grid
from sklearn.metrics import confusion_matrix
import seaborn as sns
from PIL import Image

from handwritten_digit_generation.utils.file_utils import save_to_results


def save_samples(samples, filename):
    samples = samples.cpu().numpy()
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(samples[i, 0], cmap='gray')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_to_results(filename, subdirectory='normalizing_flow'))
    plt.close()

def plot_images(images, title, save_path):
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(images[i].squeeze().cpu(), cmap='gray')
        ax.axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_comparison(real_images, *generated_image_paths, title, save_path):
    fig, axes = plt.subplots(len(generated_image_paths) + 1, 5, figsize=(15, 6 * (len(generated_image_paths) + 1)))

    # Plot real images
    for i in range(5):
        axes[0, i].imshow(real_images[i].squeeze().cpu(), cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title('Real')

    # Plot generated images
    for row, image_path in enumerate(generated_image_paths, start=1):
        generated_images = Image.open(image_path)
        generated_images = np.array(generated_images)

        if len(generated_images.shape) == 3:  # If it's a grid of images
            n_images = generated_images.shape[1] // generated_images.shape[0]
            image_size = generated_images.shape[0]
            for i in range(5):
                img = generated_images[:, i * image_size:(i + 1) * image_size]
                axes[row, i].imshow(img, cmap='gray')
                axes[row, i].axis('off')
        else:  # If it's a single image
            axes[row, 0].imshow(generated_images, cmap='gray')
            axes[row, 0].axis('off')

        axes[row, 0].set_ylabel(f'Generated\n({image_path.split("/")[-2]})', rotation=0, labelpad=40, va='center')

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()


def plot_loss_curve(losses, title, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig(save_path)
    plt.close()