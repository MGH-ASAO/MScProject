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

from handwritten_digit_generation.models.gan import Generator, Discriminator
from handwritten_digit_generation.utils.file_utils import save_to_results, get_project_root
from handwritten_digit_generation.utils.training_utils import to_cpu, print_progress

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

parser = argparse.ArgumentParser(description='Train GAN model')
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
    'device': torch.device("mps" if torch.backends.mps.is_available() else "cpu"),
    'latent_dim': 100,
    'img_shape': (1, 28, 28)
}


def main():
    logging.info(f"Starting GAN training script at {datetime.now()}")
    logging.info(f"Using device: {CONFIG['device']}")
    logging.info(f"Configuration: {CONFIG}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    train_dataset = datasets.MNIST(root=os.path.join(project_root, 'data'), train=True, download=True,
                                   transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)

    generator = Generator(CONFIG['latent_dim'], CONFIG['img_shape']).to(CONFIG['device'])
    discriminator = Discriminator(CONFIG['img_shape']).to(CONFIG['device'])

    adversarial_loss = nn.BCELoss()

    optimizer_G = optim.Adam(generator.parameters(), lr=CONFIG['lr'], betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=CONFIG['lr'], betas=(0.5, 0.999))

    try:
        for epoch in range(CONFIG['n_epochs']):
            for i, (imgs, _) in enumerate(train_loader):
                valid = torch.ones((imgs.size(0), 1), requires_grad=False).to(CONFIG['device'])
                fake = torch.zeros((imgs.size(0), 1), requires_grad=False).to(CONFIG['device'])

                real_imgs = imgs.to(CONFIG['device'])

                optimizer_G.zero_grad()
                z = torch.randn((imgs.size(0), CONFIG['latent_dim'])).to(CONFIG['device'])
                gen_imgs = generator(z)
                g_loss = adversarial_loss(discriminator(gen_imgs), valid)
                g_loss.backward()
                optimizer_G.step()

                optimizer_D.zero_grad()
                real_loss = adversarial_loss(discriminator(real_imgs), valid)
                fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                optimizer_D.step()

                if i % 100 == 0:
                    print_progress(epoch, CONFIG['n_epochs'], i, len(train_loader), d_loss.item(), g_loss.item())

            logging.info(
                f'Epoch {epoch + 1}/{CONFIG["n_epochs"]}, D loss: {d_loss.item():.4f}, G loss: {g_loss.item():.4f}')

            if (epoch + 1) % CONFIG['save_interval'] == 0:
                with to_cpu(generator) as g:
                    torch.save(g.state_dict(),
                               save_to_results(f'gan_generator_epoch_{epoch + 1}.pth', subdirectory='gan'))
                with to_cpu(discriminator) as d:
                    torch.save(d.state_dict(),
                               save_to_results(f'gan_discriminator_epoch_{epoch + 1}.pth', subdirectory='gan'))

            torch.save({
                'epoch': epoch,
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_D': optimizer_D.state_dict(),
            }, save_to_results(f'gan_checkpoint_epoch_{epoch + 1}.pth', subdirectory='gan'))

        torch.save(generator.state_dict(), save_to_results('gan_generator.pth', subdirectory='gan'))
        torch.save(discriminator.state_dict(), save_to_results('gan_discriminator.pth', subdirectory='gan'))
        logging.info("Final model saved")

    except Exception as e:
        logging.exception("An error occurred:")
        torch.save({
            'epoch': epoch,
            'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict(),
            'optimizer_G': optimizer_G.state_dict(),
            'optimizer_D': optimizer_D.state_dict(),
        }, save_to_results(f'gan_checkpoint_epoch_{epoch + 1}.pth', subdirectory='gan'))

    logging.info(f"GAN training completed at {datetime.now()}")


if __name__ == "__main__":
    main()
