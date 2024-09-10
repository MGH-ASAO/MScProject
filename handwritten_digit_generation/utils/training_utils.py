# handwritten_digit_generation/utils/training_utils.py

import torch
import torch.nn.functional as F
import contextlib
import logging
import numpy as np

@contextlib.contextmanager
def to_cpu(model):
    """临时将模型移到CPU的上下文管理器"""
    device = next(model.parameters()).device
    model.cpu()
    yield model
    model.to(device)

def print_progress(epoch, n_epochs, batch, n_batches, loss):
    """打印训练进度"""
    logging.info(f"[Epoch {epoch + 1}/{n_epochs}] [Batch {batch + 1}/{n_batches}] [Loss: {loss:.4f}]")

def preprocess(x):
    """将输入数据预处理到 [-1, 1] 范围内"""
    return (x.view(x.size(0), -1) / 255.0 - 0.5) * 2

def validate(model, dataloader, compute_loss, device):
    """验证模型性能"""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, _ in dataloader:
            data = preprocess(data).to(device)
            z, log_det = model(data)
            loss = compute_loss(z, log_det)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def save_checkpoint(state, filename):
    """保存检查点"""
    torch.save(state, filename)
    logging.info(f"Checkpoint saved to {filename}")

def load_checkpoint(model, optimizer, filename, device):
    """加载检查点"""
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch'] + 1
    logging.info(f"Checkpoint loaded from {filename}")
    return start_epoch

def get_device():
    """获取可用的设备"""
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def set_seed(seed):
    """设置随机种子以确保可重复性"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            logging.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss

def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

def get_index_from_list(vals, t, x_shape):
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def forward_diffusion_sample(x_0, t, device="mps"):
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x_0.shape)

    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
           + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise

# Define beta schedule
T = 300
betas = linear_beta_schedule(timesteps=T).view(-1)

# Pre-calculate different terms for closed form
alphas = (1. - betas).view(-1)
alphas_cumprod = torch.cumprod(alphas, dim=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)