import os
import sys
import yaml
import torch
import numpy as np
from architecture import build_model
from utils.checkpoint import load_checkpoint
from utils.data import get_cifar10_loader, extract_fixed_batches

# ============================================================
# Utils : flatten / unflatten paramètres
# ============================================================

def flatten_params(params):
    return torch.cat([p.detach().flatten() for p in params])

def get_shapes_and_sizes(params):
    shapes = [p.shape for p in params]
    sizes = [p.numel() for p in params]
    return shapes, sizes

def set_params_from_flat(model, theta_flat, shapes, sizes):
    idx = 0
    with torch.no_grad():
        for p, shape, size in zip(model.parameters(), shapes, sizes):
            p.copy_(theta_flat[idx:idx + size].view(shape))
            idx += size

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    # ---------------------------
    # Config
    # ---------------------------
    config_name = sys.argv[1]
    print(f"[INFO] Config : {config_name}")

    ROOT_DIR = os.path.dirname(__file__)
    config_path = os.path.join(ROOT_DIR, "configs", f"{config_name}.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]
    train_cfg = config["training"]
    dataset_cfg = config["dataset"]

    device = train_cfg["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    print(f"[INFO] Device : {device}")

    # ---------------------------
    # Dataset
    # ---------------------------
    train_loader = get_cifar10_loader(
        batch_size=dataset_cfg["batch_size"],
        shuffle=True,
        num_workers=1  # Pas besoin de parallélisme pour plot
    )

    # Prendre seulement un sous-ensemble fixe pour le plot
    fixed_batches = extract_fixed_batches(train_loader, dataset_cfg["n_fixed_batches"])
    fixed_batches = [(x.to(device), y.to(device)) for x, y in fixed_batches]
    print(f"[INFO] Using {len(fixed_batches)} fixed batches for plot")

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

    # Charger checkpoint si besoin
    checkpoint_path = f"models/{config_name}.pt"
    model, _ = load_checkpoint(model, checkpoint_path)
    model.eval()

    # ---------------------------
    # Loss
    # ---------------------------
    loss_fn = torch.nn.CrossEntropyLoss()

    # ---------------------------
    # Directions pour loss landscape
    # ---------------------------
    directions_path = f"plot_resources/directions_{config_name}.pt"
    directions = torch.load(directions_path)
    delta = flatten_params(directions["delta"]).to(device)
    eta   = flatten_params(directions["eta"]).to(device)

    # Flatten paramètres actuels
    theta_star = flatten_params(model.parameters()).to(device)
    shapes, sizes = get_shapes_and_sizes(model.parameters())

    # ---------------------------
    # Grid pour landscape
    # ---------------------------
    n_points = 21
    length = 10
    alphas = np.linspace(-length, length, n_points)
    betas  = np.linspace(-length, length, n_points)
    Z = np.zeros((n_points, n_points))

    # ---------------------------
    # Calcul du landscape
    # ---------------------------
    print("[INFO] Computing loss landscape...")
    with torch.no_grad():
        for i, a in enumerate(alphas):
            for j, b in enumerate(betas):
                theta = theta_star + a * delta + b * eta
                set_params_from_flat(model, theta, shapes, sizes)

                loss = 0.0
                for x, y in fixed_batches:
                    loss += loss_fn(model(x), y).item()

                Z[i, j] = loss / len(fixed_batches)

            if i == 0 or (i + 1) % 5 == 0:
                print(f"[INFO] Row {i+1}/{n_points} done")

    # ---------------------------
    # Sauvegarde
    # ---------------------------
    output_path = f"plot_resources/Z_{config_name}.npy"
    os.makedirs("plot_resources", exist_ok=True)
    np.save(output_path, Z)
    print(f"[INFO] Loss landscape saved to {output_path}")
