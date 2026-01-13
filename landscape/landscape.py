import os
import sys
import yaml
import torch
import numpy as np

# --- Ajouter racine du projet ---
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from models.architecture import CustomCNN
from utils.checkpoint import load_checkpoint

# ---------------------------
# Utilitaire : setter des poids
# ---------------------------
def set_params(model, base, delta, eta, alpha, beta):
    with torch.no_grad():
        for p, p0, d, e in zip(model.parameters(), base, delta, eta):
            p.copy_(p0 + alpha * d + beta * e)

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":

    config_name = input("Entrez le nom du fichier de configuration YAML (sans .yaml): ")

    ROOT_DIR = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )

    config_path = os.path.join(
        ROOT_DIR, "configs", f"{config_name}.yaml"
    )
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]
    train_cfg = config["training"]

    # -------- Device --------
    device = train_cfg["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # -------- Paths --------
    checkpoint_path = f"models/{config_name}.pt"
    fixed_batches_path = f"data/fixed_batches.pt"
    directions_path = f"landscape/directions_{config_name}.pt"
    output_path = f"landscape/Z_{config_name}.npy"

    # -------- Hyperparams paysage --------
    n_points = 25
    alpha_range = (-1.0, 1.0)
    beta_range = (-1.0, 1.0)

    # -------- Model --------
    model = CustomCNN(
        depth=model_cfg["depth"],
        width=model_cfg["width"],
        use_skip=model_cfg["use_skip"],
        use_bn=model_cfg["use_bn"],
        activation=model_cfg["activation"],
        dropout=model_cfg["dropout"]
    ).to(device)

    model, _ = load_checkpoint(model, checkpoint_path)
    model.eval()

    # -------- Fixed batches --------
    fixed_batches = torch.load(fixed_batches_path)
    fixed_batches = [(x.to(device), y.to(device)) for x, y in fixed_batches]

    # -------- Directions --------
    directions = torch.load(directions_path)
    delta = directions["delta"]
    eta   = directions["eta"]

    # -------- θ* --------
    theta_star = [p.detach().clone() for p in model.parameters()]

    # -------- Loss --------
    loss_fn = torch.nn.CrossEntropyLoss()

    # -------- Grid --------
    alphas = np.linspace(alpha_range[0], alpha_range[1], n_points)
    betas  = np.linspace(beta_range[0], beta_range[1], n_points)
    Z = np.zeros((n_points, n_points))

    print("[INFO] Computing loss landscape...")
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            set_params(model, theta_star, delta, eta, a, b)

            loss = 0.0
            with torch.no_grad():
                for x, y in fixed_batches:
                    loss += loss_fn(model(x), y).item()

            Z[i, j] = loss / len(fixed_batches)

        if (i + 1) % 5 == 0 or i == 0:
            print(f"[INFO] Row {i+1}/{n_points} done")

    # -------- Save --------
    os.makedirs("landscape", exist_ok=True)
    np.save(output_path, Z)

    print(f"[INFO] Loss landscape saved to {output_path}")
