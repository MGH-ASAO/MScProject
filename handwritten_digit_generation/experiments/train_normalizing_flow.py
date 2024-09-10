import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import logging
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
from datetime import datetime

from handwritten_digit_generation.models.normalizing_flow import NormalizingFlow, initialize_weights
from handwritten_digit_generation.utils.file_utils import save_to_results
from handwritten_digit_generation.utils.training_utils import (
    to_cpu, print_progress, preprocess, save_checkpoint, load_checkpoint,
    get_device, set_seed, EarlyStopping, validate
)
from handwritten_digit_generation.utils.visualization import save_samples

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

parser = argparse.ArgumentParser(description='Train Normalizing Flow model')
parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
parser.add_argument('--save_interval', type=int, default=10, help='save model every n epochs')
parser.add_argument('--seed', type=int, default=42, help='random seed')
args = parser.parse_args()

CONFIG = {
    'batch_size': args.batch_size,
    'lr': args.lr,
    'n_epochs': 500,
    'save_interval': args.save_interval,
    'device': get_device(),
    'input_dim': 784,
    'n_flows': 16,
    'weight_decay': 1e-5,
}

def compute_loss(z, log_det):
    prior_ll = -0.5 * torch.sum(z ** 2, dim=1) - 0.5 * z.shape[1] * np.log(2 * np.pi)
    log_likelihood = prior_ll + log_det
    return -torch.mean(log_likelihood)

def train_flow(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch, _ in data_loader:
        batch = preprocess(batch).to(device)
        optimizer.zero_grad()
        z, log_det = model(batch)
        loss = compute_loss(z, log_det)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

def main():
    set_seed(args.seed)
    logging.info(f"Starting Normalizing Flow training script at {datetime.now()}")
    logging.info(f"Using device: {CONFIG['device']}")
    logging.info(f"Configuration: {CONFIG}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4)

    val_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4)

    model = NormalizingFlow(CONFIG['input_dim'], CONFIG['n_flows']).to(CONFIG['device'])
    model.apply(initialize_weights)

    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG['n_epochs'], eta_min=1e-6)

    checkpoint_path = save_to_results('normalizing_flow_checkpoint.pth', subdirectory='normalizing_flow')
    best_model_path = save_to_results('best_normalizing_flow_model.pth', subdirectory='normalizing_flow')

    start_epoch = load_checkpoint(model, optimizer, checkpoint_path, CONFIG['device'])

    early_stopping = EarlyStopping(patience=20, verbose=True)

    try:
        for epoch in range(start_epoch, CONFIG['n_epochs']):
            train_loss = train_flow(model, train_loader, optimizer, CONFIG['device'])
            val_loss = validate(model, val_loader, compute_loss, CONFIG['device'])
            scheduler.step()

            logging.info(f'Epoch {epoch + 1}/{CONFIG["n_epochs"]}, '
                         f'Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

            save_checkpoint({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss
            }, checkpoint_path)

            early_stopping(val_loss, model, best_model_path)
            if early_stopping.early_stop:
                logging.info("Early stopping")
                break

            if (epoch + 1) % CONFIG['save_interval'] == 0:
                with to_cpu(model) as m:
                    torch.save(m.state_dict(),
                               save_to_results(f'normalizing_flow_model_epoch_{epoch + 1}.pth',
                                               subdirectory='normalizing_flow'))

            if epoch % 10 == 0:
                with torch.no_grad():
                    z = torch.randn(16, CONFIG['input_dim']).to(CONFIG['device'])
                    samples = model.inverse(z)
                    samples = samples.view(-1, 1, 28, 28)
                    save_samples(samples, f'samples_epoch_{epoch}.png')

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        save_checkpoint({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss
        }, checkpoint_path)

    logging.info(f"Normalizing Flow training completed at {datetime.now()}")

if __name__ == "__main__":
    main()