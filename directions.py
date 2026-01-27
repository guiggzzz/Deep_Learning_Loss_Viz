import os
import sys
import yaml
import torch
from architecture import build_model
from utils.checkpoint import load_checkpoint

# ---------------------------
# Filter-wise random directions 
# ---------------------------
def random_directions_filterwise(model):
    """
    Génère des directions aléatoires normalisées filter-wise.
    
    Args:
        model: Le modèle PyTorch
    """
    direction = []

    for name, p in model.named_parameters():
        # Vérifier si c'est un paramètre BatchNorm ou biais
        is_bn_or_bias = ('bn' in name.lower() or 
                         'bias' in name.lower() or 
                         'norm' in name.lower())
        
        if is_bn_or_bias:
            direction.append(torch.zeros_like(p))
            continue

        # Génère direction aléatoire
        d = torch.randn_like(p)

        # Pour les paramètres multi-dimensionnels (Conv, Linear avec poids)
        if p.ndim >= 2:
            # Reshape : (out_channels, -1)
            d_flat = d.view(d.size(0), -1)
            p_flat = p.view(p.size(0), -1)

            # Normes par filtre
            d_norm = torch.norm(d_flat, dim=1, keepdim=True)
            p_norm = torch.norm(p_flat, dim=1, keepdim=True)

            # Filter-wise normalization
            d_flat = d_flat / (d_norm + 1e-10) * p_norm
            direction.append(d_flat.view_as(p))
        else:
            # Pour les paramètres 1D (si on ne les ignore pas)
            # Normaliser simplement par la norme totale
            d_norm = torch.norm(d)
            p_norm = torch.norm(p)
            direction.append(d / (d_norm + 1e-10) * p_norm)

    return direction


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    config_name = sys.argv[1]
    print(f"Paramètre reçu : {config_name}")

    ROOT_DIR = os.path.dirname(__file__)
    config_path = os.path.join(ROOT_DIR, "configs", f"{config_name}.yaml")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]

    # -------- Device --------
    device = config["training"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    print(f"[INFO] Device: {device}")

    # -------- Model --------
    model = build_model(
        nn_architecture=model_cfg["nn_architecture"],
        num_config=model_cfg["num_config"],
        activation=model_cfg["activation"],
        dropout=model_cfg.get("dropout", 0.0)
    ).to(device)

    # -------- Checkpoint --------
    checkpoint_path = f"models/{config_name}.pt"
    model, _ = load_checkpoint(model, checkpoint_path)
    model.eval()

    # -------- Directions --------
    os.makedirs("plot_resources", exist_ok=True)
    directions_path = f"plot_resources/directions_{config_name}.pt"

    if os.path.exists(directions_path):
        print("[INFO] Loading existing directions...")
        directions = torch.load(directions_path, map_location=device)
        delta = directions["delta"]
        eta = directions["eta"]
    else:
        print(f"[INFO] Generating filter-wise random directions")
        torch.manual_seed(0)  # Reproductibilité
        delta = random_directions_filterwise(model)
        eta = random_directions_filterwise(model)

        torch.save(
            {"delta": delta, "eta": eta},
            directions_path
        )
        print(f"[INFO] Directions saved to {directions_path}")

