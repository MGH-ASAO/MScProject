import os
import sys
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse
import logging
from datetime import datetime
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from handwritten_digit_generation.models.vae import ConvVAE
from handwritten_digit_generation.utils.file_utils import save_to_results
from handwritten_digit_generation.utils.training_utils import to_cpu, print_progress

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

parser = argparse.ArgumentParser(description='Train VAE model')
parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
parser.add_argument('--save_interval', type=int, default=10, help='save model every n epochs')
parser.add_argument('--latent_dim', type=int, default=32, help='dimension of latent space')
parser.add_argument('--beta', type=float, default=0.1, help='beta value for beta-VAE')
args = parser.parse_args()

CONFIG = {
    'batch_size': args.batch_size,
    'lr': args.lr,
    'n_epochs': 200,
    'save_interval': args.save_interval,
    'device': torch.device("mps" if torch.backends.mps.is_available() else "cpu"),
    'latent_dim': args.latent_dim,
    'beta': args.beta
}


def main():
    logging.info(f"Starting VAE training script at {datetime.now()}")
    logging.info(f"Using device: {CONFIG['device']}")
    logging.info(f"Configuration: {CONFIG}")

    # Data augmentation
    transform = transforms.Compose([
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.MNIST(root=os.path.join(project_root, 'data'), train=True, download=True,
                                   transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)

    val_dataset = datasets.MNIST(root=os.path.join(project_root, 'data'), train=False, download=True,
                                 transform=transforms.ToTensor())
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

    model = ConvVAE(CONFIG['latent_dim'], CONFIG['beta']).to(CONFIG['device'])
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True)

    checkpoint_path = save_to_results('vae_checkpoint.pth', subdirectory='vae')
    final_model_path = save_to_results('vae_model.pth', subdirectory='vae')

    writer = SummaryWriter(log_dir=save_to_results('tensorboard_logs', subdirectory='vae'))

    start_epoch = 0
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=CONFIG['device'])
        if 'model' in checkpoint:
            try:
                model.load_state_dict(checkpoint['model'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                start_epoch = checkpoint['epoch'] + 1
                logging.info(f"Resuming training from epoch {start_epoch}")
            except RuntimeError as e:
                logging.warning(f"Could not load old checkpoint due to model architecture change: {e}")
                logging.info("Starting training from scratch with the new model architecture.")
        else:
            logging.warning("Checkpoint format not recognized. Starting training from scratch.")
    else:
        logging.info("No checkpoint found. Starting training from scratch.")

    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0

    try:
        for epoch in range(start_epoch, CONFIG['n_epochs']):
            model.train()
            train_loss = 0
            train_bce = 0
            train_kld = 0
            for i, (imgs, _) in enumerate(train_loader):
                imgs = imgs.to(CONFIG['device'])
                optimizer.zero_grad()
                recon_batch, mu, logvar = model(imgs)
                loss, bce, kld = model.loss_function(recon_batch, imgs, mu, logvar)
                loss.backward()
                train_loss += loss.item()
                train_bce += bce.item()
                train_kld += kld.item()
                optimizer.step()

                if i % 100 == 0:
                    print_progress(epoch, CONFIG['n_epochs'], i, len(train_loader), loss.item() / len(imgs))

            avg_train_loss = train_loss / len(train_loader.dataset)
            avg_train_bce = train_bce / len(train_loader.dataset)
            avg_train_kld = train_kld / len(train_loader.dataset)
            logging.info(f'Epoch {epoch + 1}/{CONFIG["n_epochs"]}, '
                         f'Average train loss: {avg_train_loss:.4f}, '
                         f'BCE: {avg_train_bce:.4f}, KLD: {avg_train_kld:.4f}')
            writer.add_scalar('Loss/train', avg_train_loss, epoch)
            writer.add_scalar('BCE/train', avg_train_bce, epoch)
            writer.add_scalar('KLD/train', avg_train_kld, epoch)

            # Validation
            model.eval()
            val_loss = 0
            val_bce = 0
            val_kld = 0
            with torch.no_grad():
                for imgs, _ in val_loader:
                    imgs = imgs.to(CONFIG['device'])
                    recon_batch, mu, logvar = model(imgs)
                    loss, bce, kld = model.loss_function(recon_batch, imgs, mu, logvar)
                    val_loss += loss.item()
                    val_bce += bce.item()
                    val_kld += kld.item()

            avg_val_loss = val_loss / len(val_loader.dataset)
            avg_val_bce = val_bce / len(val_loader.dataset)
            avg_val_kld = val_kld / len(val_loader.dataset)
            logging.info(f'Epoch {epoch + 1}/{CONFIG["n_epochs"]}, '
                         f'Average validation loss: {avg_val_loss:.4f}, '
                         f'BCE: {avg_val_bce:.4f}, KLD: {avg_val_kld:.4f}')
            writer.add_scalar('Loss/validation', avg_val_loss, epoch)
            writer.add_scalar('BCE/validation', avg_val_bce, epoch)
            writer.add_scalar('KLD/validation', avg_val_kld, epoch)

            scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), final_model_path)
                logging.info(f"New best model saved with validation loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logging.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break

            if (epoch + 1) % CONFIG['save_interval'] == 0:
                with to_cpu(model) as m:
                    torch.save(m.state_dict(), save_to_results(f'vae_model_epoch_{epoch + 1}.pth', subdirectory='vae'))

            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, checkpoint_path)

        logging.info(f"Final model saved to {final_model_path}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, checkpoint_path)

    writer.close()
    logging.info(f"VAE training completed at {datetime.now()}")


if __name__ == "__main__":
    main()