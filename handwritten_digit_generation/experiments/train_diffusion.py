# handwritten_digit_generation/experiments/train_diffusion.py

import os
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse
import logging
from datetime import datetime
from torch.optim.lr_scheduler import CosineAnnealingLR

# Get the path to the project root directory and add it to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from handwritten_digit_generation.models.diffusion import DiffusionModel, forward_diffusion_sample
from handwritten_digit_generation.utils.file_utils import save_to_results
from handwritten_digit_generation.utils.training_utils import to_cpu, print_progress, T

# Set up log
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Parameter analysis
parser = argparse.ArgumentParser(description='Train Diffusion model')
parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--epochs', type=int, default=500, help='number of epochs to train')
parser.add_argument('--save_interval', type=int, default=10, help='save model every n epochs')
args = parser.parse_args()

#Configuration
CONFIG = {
    'batch_size': args.batch_size,
    'lr': args.lr,
    'n_epochs': args.epochs,
    'save_interval': args.save_interval,
    'device': torch.device("mps" if torch.backends.mps.is_available() else "cpu"),
    'input_dim': 784
}

def load_checkpoint(model, optimizer, checkpoint_path, device):
    if os.path.exists(checkpoint_path):
        logging.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model' in checkpoint and 'optimizer' in checkpoint and 'epoch' in checkpoint:
            try:
                model.load_state_dict(checkpoint['model'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                start_epoch = checkpoint['epoch'] + 1
                logging.info(f"Resuming training from epoch {start_epoch}")
                return start_epoch
            except RuntimeError as e:
                logging.warning(f"Could not load checkpoint due to: {e}")
                logging.info("Starting training from scratch.")
        else:
            logging.warning("Checkpoint format not recognized. Starting training from scratch.")
    else:
        logging.info("No checkpoint found. Starting training from scratch.")
    return 0

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_path):
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': loss
    }, checkpoint_path)
    logging.info(f'Checkpoint saved at epoch {epoch}')

def main():
    logging.info(f"Starting Diffusion training script at {datetime.now()}")
    logging.info(f"Using device: {CONFIG['device']}")
    logging.info(f"Configuration: {CONFIG}")

    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root=os.path.join(project_root, 'data'), train=True, download=True,
                                   transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)

    #Initialize the model
    model = DiffusionModel(input_channels=1, hidden_dim=64).to(CONFIG['device'])
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG['n_epochs'], eta_min=1e-6)

    # Checkpoint path
    checkpoint_path = save_to_results('diffusion_checkpoint.pth', subdirectory='diffusion')

    #Final model path
    final_model_path = save_to_results('diffusion_model.pth', subdirectory='diffusion')

    # Resume training (if checkpoint exists)
    start_epoch = load_checkpoint(model, optimizer, checkpoint_path, CONFIG['device'])

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

                # Make sure the shapes of x_noisy and noise are consistent
                if x_noisy.dim() == 2:
                    x_noisy = x_noisy.view(-1, 1, 28, 28)
                if noise.dim() == 2:
                    noise = noise.view(-1, 1, 28, 28)

                # Get predicted noise
                noise_pred = model(x_noisy, t)

                # Ensure that the shapes of noise and noise_pred are consistent
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

            scheduler.step()

            # Save checkpoint
            save_checkpoint(model, optimizer, epoch, avg_loss, checkpoint_path)

            if (epoch + 1) % CONFIG['save_interval'] == 0:
                with to_cpu(model) as m:
                    torch.save(m.state_dict(),
                               save_to_results(f'diffusion_model_epoch_{epoch + 1}.pth', subdirectory='diffusion'))

        torch.save(model.state_dict(), final_model_path)
        logging.info(f"Final model saved to {final_model_path}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        # Save checkpoint
        save_checkpoint(model, optimizer, epoch, avg_loss, checkpoint_path)

    logging.info(f"Diffusion training completed at {datetime.now()}")


if __name__ == "__main__":
    main()