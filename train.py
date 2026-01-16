import torch
import torch.nn as nn
import torch.optim as optim
import sys, os
import yaml
import random
import numpy as np
from sklearn.metrics import f1_score

from utils.data import get_cifar10_loader, extract_fixed_batches
from utils.checkpoint import save_checkpoint
from architecture import CustomCNN


# ---------------------------
# Reproductibilité (IMPORTANT)
# ---------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    config_name = sys.argv[1]
    print(f"Paramètre reçu : {config_name}")

    set_seed(42)

    ROOT_DIR = os.path.dirname(__file__)

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
    nesterov = config["training"]["nesterov"]
    weight_decay = config["training"]["weight_decay"]

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
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        nesterov=nesterov,
        weight_decay=weight_decay
    )

    criterion = nn.CrossEntropyLoss()
    f1_history = []
    loss_history = []
    # ---------------------------
    # Entraînement
    # ---------------------------
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        all_preds = []
        all_targets = []

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.detach().cpu())
            all_targets.append(y.detach().cpu())

        avg_loss = running_loss / len(train_loader)
        loss_history.append(avg_loss)

        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()

        f1 = f1_score(all_targets, all_preds, average="macro")
        f1_history.append(f1)

        print(
            f"[Epoch {epoch+1}/{epochs}] "
            f"Loss: {avg_loss:.4f} | F1 (macro): {f1:.4f}"
        )
    # ---------------------------
    # Sauvegarde
    # ---------------------------
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    save_checkpoint(model, optimizer, checkpoint_path)


    print("[INFO] Training finished.")
    print(f"[INFO] Model saved to {checkpoint_path}")