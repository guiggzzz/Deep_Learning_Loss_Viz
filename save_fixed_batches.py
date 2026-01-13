# save_fixed_batches_with_utils.py
import torch
from utils.data import get_cifar10_loader, extract_fixed_batches

device = "cpu"   # ou "cuda"

# Charger DataLoader CIFAR-10
train_loader = get_cifar10_loader(batch_size=128, shuffle=True)

# Extraire et sauvegarder 5 batches fixes
fixed_batches = extract_fixed_batches(
    loader=train_loader,
    n_batches=5,
    save_path="data/fixed_batches.pt",
    device=device
)