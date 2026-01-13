import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_cifar10_loader(batch_size=128, shuffle=True, num_workers=4):
    """Retourne un DataLoader pour CIFAR-10 avec normalisation standard."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616)
        )
    ])
    dataset = datasets.CIFAR10(
        root="data/cifar10",
        train=True,
        download=True,
        transform=transform
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=True
    )
    return loader

def extract_fixed_batches(loader, n_batches=5, save_path="data/fixed_batches.pt", device="cpu"):
    """
    Extrait un nombre fixe de batches et les sauvegarde sur disque pour reuse.
    """
    fixed_batches = []
    it = iter(loader)
    for _ in range(n_batches):
        x, y = next(it)
        fixed_batches.append((x.to(device), y.to(device)))
    torch.save(fixed_batches, save_path)
    print(f"[INFO] Saved {n_batches} fixed batches to {save_path}")
    return fixed_batches
