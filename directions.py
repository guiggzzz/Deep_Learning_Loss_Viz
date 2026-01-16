import os
import sys
import yaml
import torch


from architecture import CustomCNN
from utils.checkpoint import load_checkpoint

# ---------------------------
# Filter-wise random directions
# ---------------------------
def random_directions_filterwise(model):
    direction = []

    for p in model.parameters():
        if p.ndim <= 1:
            # biais, BN, etc.
            direction.append(torch.zeros_like(p))
            continue

        # Génère direction brute
        d = torch.randn_like(p)

        # reshape : (out_channels, -1)
        d_flat = d.view(d.size(0), -1)
        p_flat = p.view(p.size(0), -1)

        # normes par filtre
        d_norm = torch.norm(d_flat, dim=1, keepdim=True)
        p_norm = torch.norm(p_flat, dim=1, keepdim=True)

        # filter-wise normalization
        d_flat = d_flat / (d_norm + 1e-10) * p_norm

        direction.append(d_flat.view_as(p))

    return direction

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    config_name = sys.argv[1]
    print(f"Paramètre reçu : {config_name}")

    ROOT_DIR = os.path.dirname(__file__)

    config_path = os.path.join(
        ROOT_DIR, "configs", f"{config_name}.yaml"
    )
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]

    # -------- Device --------
    device = config["training"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # -------- Model --------
    model = CustomCNN(
        depth=model_cfg["depth"],
        width=model_cfg["width"],
        use_skip=model_cfg["use_skip"],
        use_bn=model_cfg["use_bn"],
        activation=model_cfg["activation"],
        dropout=model_cfg["dropout"]
    ).to(device)

    # -------- Checkpoint --------
    checkpoint_path = f"models/{config_name}.pt"
    model, _ = load_checkpoint(model, checkpoint_path)
    model.eval()

    # -------- Directions (FIXED) --------
    os.makedirs("plot_resources", exist_ok=True)
    directions_path = f"plot_resources/directions_{config_name}.pt"

    if os.path.exists(directions_path):
        print("[INFO] Loading fixed directions...")
        directions = torch.load(directions_path, map_location=device)
        delta = directions["delta"]
        eta   = directions["eta"]
    else:
        print("[INFO] Generating fixed filter-wise random directions...")
        torch.manual_seed(0)  # reproductibilité
        delta = random_directions_filterwise(model)
        eta   = random_directions_filterwise(model)

        torch.save(
            {"delta": delta, "eta": eta},
            directions_path
        )
        print(f"[INFO] Directions saved to {directions_path}")

    print("[INFO] Directions ready (consistent across configs)")
