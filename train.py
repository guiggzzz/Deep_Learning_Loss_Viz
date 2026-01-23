import torch
import torch.nn as nn
import torch.optim as optim
import sys, os
import yaml
import random
import numpy as np
import csv
from sklearn.metrics import f1_score

from utils.data import get_cifar10_loader, extract_fixed_batches
from utils.checkpoint import save_checkpoint
from architecture import build_model


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
    model = build_model(
        resnet=model_cfg["resnet"],
        num_config=model_cfg["num_config"],
        use_skip=model_cfg["use_skip"],
        activation=model_cfg["activation"],
        dropout=model_cfg.get("dropout", 0.0)
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

        if epoch % 5 == 0:
            f1 = f1_score(all_targets, all_preds, average="macro")
            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f} - F1 Macro: {f1:.4f}")
    f1 = f1_score(all_targets, all_preds, average="macro")

    # ---------------------------
    # Sauvegarde
    # ---------------------------
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    save_checkpoint(model, optimizer, checkpoint_path)

    # ---------------------------
    # Sauvegarde des caractéristiques du modèle en CSV
    # ---------------------------
    csv_path = checkpoint_path.replace(".pt", "_summary.csv")

    summary_data = [
        # Dataset
        ("dataset_batch_size", batch_size),
        ("dataset_num_workers", num_workers),
        ("dataset_n_fixed_batches", n_fixed_batches),

        # Training
        ("training_epochs", epochs),
        ("training_lr", lr),
        ("training_momentum", momentum),
        ("training_nesterov", nesterov),
        ("training_weight_decay", weight_decay),
        ("training_device", device),

        # Model
        ("model_resnet", model_cfg["resnet"]),
        ("model_num_config", model_cfg["num_config"]),
        ("model_use_skip", model_cfg["use_skip"]),
        ("model_activation", model_cfg["activation"]),
        ("model_dropout", model_cfg.get("dropout", 0.0)),

        # Metrics
        ("final_loss", loss_history[-1]),
        ("final_f1_macro", f1),
    ]

    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["parameter", "value"])
        writer.writerows(summary_data)

    print(f"[INFO] Model summary saved to {csv_path}")


