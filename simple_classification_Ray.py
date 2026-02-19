# Python 3.13 + PyTorch 2.6
# Uses sklearn digits (8x8) and a small ResNet with configurable number of residual blocks.

import argparse
import os
import time

os.environ["RAY_DEDUP_LOGS"] = "0"

import numpy as np
import ray
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils import summary


# -----------------------------
# Residual Block + Small ResNet
# -----------------------------
class BasicBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        return out


class ResNet(nn.Module):
    """
    Simple ResNet-like model:
      stem: 1x8x8 -> Cx8x8
      blocks: N residual blocks at constant width C
      head: global average pooling -> linear classifier
    """
    def __init__(self, num_blocks: int, width: int, num_classes: int = 10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[BasicBlock(width) for _ in range(num_blocks)])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(width, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x


# -----------------------------
# Data + Train/Eval
# -----------------------------
def load_digit_data():
    digits = load_digits()  # 1797 samples, 8x8, labels 0..9
    X = digits.data.astype(np.float32)         # (N, 64)
    y = digits.target.astype(np.int64)         # (N,)

    # Standardize features to make optimization stable
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    # reshape to image: (N, 1, 8, 8)
    X = X.reshape(-1, 1, 8, 8)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train = torch.from_numpy(X_train)
    y_train = torch.from_numpy(y_train)
    X_val = torch.from_numpy(X_val)
    y_val = torch.from_numpy(y_val)
    return X_train, y_train, X_val, y_val


def train(num_blocks: int, width: int, epochs: int, batch_size: int, device: str):
    X_train, y_train, X_val, y_val = load_digit_data()

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    model = ResNet(num_blocks=num_blocks, width=width, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    start = time.perf_counter()
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    elapsed = time.perf_counter() - start

    # Eval
    model.eval()
    with torch.no_grad():
        logits = model(X_val.to(device))
        acc = (logits.argmax(dim=1) == y_val.to(device)).float().mean().item()

    # Report params for sanity (blocks scaling)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"num_blocks={num_blocks} | width={width} | params={n_params:,}")
    print(f"train_time={elapsed:.4f} sec | val_acc={acc:.4f}")

    return elapsed, acc


# Each Ray task uses exactly 1 GPU.
@ray.remote(num_gpus=1)
def ray_run(num_blocks: int, width: int, epochs: int, batch_size: int):
    # Ray will set CUDA_VISIBLE_DEVICES for this worker to a single GPU id.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Optional: print to verify GPU isolation
    print(f"[PID {os.getpid()}] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} device={device}")

    elapsed, acc = train(
        num_blocks=num_blocks,
        width=width,
        epochs=epochs,
        batch_size=batch_size,
        device=device,
    )
    return {"num_blocks": num_blocks, "elapsed": elapsed, "acc": acc, "Used_GPU_num": os.environ.get('CUDA_VISIBLE_DEVICES')}


if __name__ == "__main__":
    ray.init()  # auto-detects resources (GPUs/CPUs)
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--width", type=int, default=32, help="Channels in the residual trunk")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    block_configs = list(range(4, 81, 4))

    futures = [ray_run.remote(nb, args.width, args.epochs, args.batch_size) for nb in block_configs]
    results = ray.get(futures)
    
    summary(results)    