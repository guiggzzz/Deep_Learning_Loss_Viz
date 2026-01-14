import os
import sys
import yaml
import torch

# --- Ajouter racine du projet ---
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from models.architecture import CustomCNN
from utils.checkpoint import load_checkpoint

# ---------------------------
# Filter-wise random directions
# ---------------------------
def random_directions_filterwise(model):
    """
    Génère une direction aléatoire filter-wise
    (Li et al., 2018).
    """
    direction = []
    for p in model.parameters():
        if p.ndim > 1:
            r = torch.randn_like(p)
            r *= p.norm() / (r.norm() + 1e-10)
        else:
            r = torch.zeros_like(p)
        direction.append(r)
    return direction

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    config_name = sys.argv[1]
    print(f"Paramètre reçu : {config_name}")

    ROOT_DIR = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )

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
    os.makedirs("landscape", exist_ok=True)
    directions_path = f"landscape/directions_{config_name}.pt"

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
