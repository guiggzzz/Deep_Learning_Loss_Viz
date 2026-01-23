import os
import sys
import yaml
import torch
import numpy as np

# --- Ajouter racine du projet ---
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from architecture import build_model
from utils.checkpoint import load_checkpoint

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

    # ---------------------------
    # Device
    # ---------------------------
    device = train_cfg["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    print(f"[INFO] Device : {device}")

    # ---------------------------
    # Paths
    # ---------------------------
    checkpoint_path = f"models/{config_name}.pt"
    fixed_batches_path = "data/fixed_batches.pt"
    directions_path = f"plot_resources/directions_{config_name}.pt"
    output_path = f"plot_resources/Z_{config_name}.npy"

    os.makedirs("plot_resources", exist_ok=True)

    # ---------------------------
    # Hyperparams landscape
    # ---------------------------
    n_points = 21                 # 21x21 = 441 points
    length = 0.5
    alpha_range = (-length, length)
    beta_range  = (-length, length)

    # ---------------------------
    # Model
    # ---------------------------
    model = build_model(
        resnet=model_cfg["resnet"],
        num_config=model_cfg["num_config"],
        use_skip=model_cfg["use_skip"],
        activation=model_cfg["activation"],
        dropout=model_cfg.get("dropout", 0.0)
    ).to(device)

    model, _ = load_checkpoint(model, checkpoint_path)
    model.eval()

    # ---------------------------
    # Fixed batches
    # ---------------------------
    fixed_batches = torch.load(fixed_batches_path)
    fixed_batches = [(x.to(device), y.to(device)) for x, y in fixed_batches]

    # 👉 Suffisant pour un landscape propre
    fixed_batches = fixed_batches[:2]

    print(f"[INFO] Using {len(fixed_batches)} fixed batches")

    # ---------------------------
    # Directions
    # ---------------------------
    directions = torch.load(directions_path)
    delta = directions["delta"]
    eta   = directions["eta"]

    # ---------------------------
    # Flatten everything
    # ---------------------------
    theta_star = flatten_params(model.parameters()).to(device)
    delta = flatten_params(delta).to(device)
    eta   = flatten_params(eta).to(device)

    shapes, sizes = get_shapes_and_sizes(model.parameters())

    # ---------------------------
    # Loss
    # ---------------------------
    loss_fn = torch.nn.CrossEntropyLoss()

    # ---------------------------
    # Grid
    # ---------------------------
    alphas = np.linspace(alpha_range[0], alpha_range[1], n_points)
    betas  = np.linspace(beta_range[0], beta_range[1], n_points)

    Z = np.zeros((n_points, n_points))

    # ---------------------------
    # Landscape computation
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
    # Save
    # ---------------------------
    np.save(output_path, Z)
    print(f"[INFO] Loss landscape saved to {output_path}")
