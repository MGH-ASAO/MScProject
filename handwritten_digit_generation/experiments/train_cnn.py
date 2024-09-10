# handwritten_digit_generation/experiments/train_cnn.py

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse
import logging
from datetime import datetime
from torch.optim.lr_scheduler import StepLR

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from handwritten_digit_generation.models.cnn import CNN
from handwritten_digit_generation.utils.file_utils import save_to_results, get_project_root
from handwritten_digit_generation.utils.training_utils import to_cpu, print_progress, validate

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

parser = argparse.ArgumentParser(description='Train CNN model')
parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
parser.add_argument('--save_interval', type=int, default=5, help='save model every n epochs')
args = parser.parse_args()

CONFIG = {
    'batch_size': args.batch_size,
    'lr': args.lr,
    'n_epochs': args.epochs,
    'save_interval': args.save_interval,
    'device': torch.device("mps" if torch.backends.mps.is_available() else "cpu")
}


def main():
    logging.info(f"Starting CNN training script at {datetime.now()}")
    logging.info(f"Using device: {CONFIG['device']}")
    logging.info(f"Configuration: {CONFIG}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root=os.path.join(project_root, 'data'), train=True, download=True,
                                   transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_dataset = datasets.MNIST(root=os.path.join(project_root, 'data'), train=False, download=True,
                                 transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

    model = CNN().to(CONFIG['device'])
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    criterion = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    checkpoint_path = save_to_results('cnn_checkpoint.pth', subdirectory='cnn')

    final_model_path = save_to_results('cnn_model.pth', subdirectory='cnn')

    # Resume training (if checkpoint exists)
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=CONFIG['device'], weights_only=True)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        logging.info(f"Resuming training from epoch {start_epoch}")

    try:
        for epoch in range(start_epoch, CONFIG['n_epochs']):
            model.train()
            total_loss = 0
            for i, (imgs, labels) in enumerate(train_loader):
                imgs, labels = imgs.to(CONFIG['device']), labels.to(CONFIG['device'])

                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                if i % 100 == 0:
                    print_progress(epoch, CONFIG['n_epochs'], i, len(train_loader), loss.item())

            avg_loss = total_loss / len(train_loader)
            logging.info(f'Epoch {epoch + 1}/{CONFIG["n_epochs"]}, Average loss: {avg_loss:.4f}')

            val_loss, accuracy = validate(model, val_loader, criterion, CONFIG['device'])
            logging.info(f'Validation loss: {val_loss:.4f}, accuracy: {accuracy}')

            scheduler.step()

            if (epoch + 1) % CONFIG['save_interval'] == 0:
                with to_cpu(model) as m:
                    torch.save(m.state_dict(), save_to_results(f'cnn_model_epoch_{epoch + 1}.pth', subdirectory='cnn'))

            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, checkpoint_path)

        torch.save(model.state_dict(), final_model_path)
        logging.info(f"Final model saved to {final_model_path}")

    except Exception as e:
        logging.exception(f"An error occurred:")
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, checkpoint_path)

    logging.info(f"CNN training completed at {datetime.now()}")


if __name__ == "__main__":
    main()