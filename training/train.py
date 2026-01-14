import torch
import torch.nn as nn
import torch.optim as optim
import sys, os
import yaml
import random
import numpy as np

# --- Ajouter racine du projet au path ---
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.data import get_cifar10_loader, extract_fixed_batches
from utils.checkpoint import save_checkpoint
from models.architecture import CustomCNN

# ---------------------------
# Reproductibilité (IMPORTANT)
# ---------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# ---------------------------
# Charger config YAML
# ---------------------------
config_name = input("Entrez le nom du fichier de configuration YAML (sans .yaml): ")

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

config_path = os.path.join(
    ROOT_DIR, "configs", f"{config_name}.yaml"
)
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Dataset
batch_size = config["dataset"]["batch_size"]
num_workers = config["dataset"]["num_workers"]
n_fixed_batches = config["dataset"]["n_fixed_batches"]

# Training
epochs = config["training"]["epochs"]
lr = config["training"]["lr"]
momentum = config["training"]["momentum"]

# -------- Device --------
device = config["training"]["device"]
if device == "cuda" and not torch.cuda.is_available():
    device = "cpu"

# Model params (LOSS LANDSCAPE)
model_cfg = config["model"]

# Paths
checkpoint_path = f"models/{config_name}.pt"

# ---------------------------
# Dataset
# ---------------------------
train_loader = get_cifar10_loader(
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers
)

# ---------------------------
# Modèle
# ---------------------------
model = CustomCNN(
    depth=model_cfg["depth"],
    width=model_cfg["width"],
    use_skip=model_cfg["use_skip"],
    use_bn=model_cfg["use_bn"],
    activation=model_cfg["activation"],
    dropout=model_cfg["dropout"]
).to(device)

# ---------------------------
# Optimisation
# ---------------------------
optimizer = optim.SGD(
    model.parameters(),
    lr=lr,
    momentum=momentum
)

criterion = nn.CrossEntropyLoss()

# ---------------------------
# Entraînement
# ---------------------------
model.train()
for epoch in range(epochs):
    running_loss = 0.0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"[Epoch {epoch+1}/{epochs}] Loss: {avg_loss:.4f}")

# ---------------------------
# Sauvegarde
# ---------------------------
os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
save_checkpoint(model, optimizer, checkpoint_path)


print("[INFO] Training finished.")
print(f"[INFO] Model saved to {checkpoint_path}")