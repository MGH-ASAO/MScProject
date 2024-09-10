import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import logging
from torch.optim.lr_scheduler import CosineAnnealingLR
from handwritten_digit_generation.models.normalizing_flow import NormalizingFlow, initialize_weights
from handwritten_digit_generation.utils.file_utils import save_to_results
from handwritten_digit_generation.utils.visualization import save_samples

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CONFIG = {
    'batch_size': 128,
    'lr': 1e-4,
    'n_epochs': 400,
    'device': torch.device("mps" if torch.backends.mps.is_available() else "cpu"),
    'input_dim': 784,
    'n_flows': 16,
    'weight_decay': 1e-5,
}

logging.info(f"Using device: {CONFIG['device']}")

def preprocess(x):
    return x.view(x.size(0), -1)

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

def validate(model, val_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch, _ in val_loader:
            batch = preprocess(batch).to(device)
            z, log_det = model(batch)
            loss = compute_loss(z, log_det)
            total_loss += loss.item()
    return total_loss / len(val_loader)

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

def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, checkpoint_path):
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss
    }, checkpoint_path)
    logging.info(f'Checkpoint saved at epoch {epoch}')

def main():
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

    start_epoch = load_checkpoint(model, optimizer, checkpoint_path, CONFIG['device'])

    best_loss = float('inf')
    for epoch in range(start_epoch, CONFIG['n_epochs'] + 1):
        train_loss = train_flow(model, train_loader, optimizer, CONFIG['device'])
        val_loss = validate(model, val_loader, CONFIG['device'])
        scheduler.step()

        logging.info(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

        save_checkpoint(model, optimizer, epoch, train_loss, val_loss, checkpoint_path)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(),
                       save_to_results('best_normalizing_flow_model.pth', subdirectory='normalizing_flow'))
            logging.info(f'New best model saved with validation loss: {val_loss:.4f}')

        if epoch % 10 == 0:
            with torch.no_grad():
                z = torch.randn(16, CONFIG['input_dim']).to(CONFIG['device'])
                samples = model.inverse(z)
                samples = samples.view(-1, 1, 28, 28)
                save_samples(samples, f'samples_epoch_{epoch}.png')

if __name__ == "__main__":
    main()