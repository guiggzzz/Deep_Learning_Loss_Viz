import os
import sys
import yaml
import torch
from architecture import build_model
from utils.checkpoint import load_checkpoint

# ---------------------------
# Filter-wise random directions (CORRIGÉ)
# ---------------------------
def random_directions_filterwise(model, ignore_bn_bias=True):
    """
    Génère des directions aléatoires normalisées filter-wise.
    
    Args:
        model: Le modèle PyTorch
        ignore_bn_bias: Si True, met des zéros pour BatchNorm et biais
                       Si False, normalise aussi ces paramètres
    """
    direction = []

    for name, p in model.named_parameters():
        # Vérifier si c'est un paramètre BatchNorm ou biais
        is_bn_or_bias = ('bn' in name.lower() or 
                         'bias' in name.lower() or 
                         'norm' in name.lower())
        
        # Si on ignore BN/bias ET que c'est un paramètre BN/bias
        if ignore_bn_bias and is_bn_or_bias:
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


def verify_directions(model, delta, eta):
    """
    Vérifie que les directions sont correctement normalisées.
    """
    print("\n=== Vérification des directions ===")
    
    total_params = 0
    zero_params = 0
    
    for (name, p), d_delta, d_eta in zip(model.named_parameters(), delta, eta):
        p_norm = p.norm().item()
        d_delta_norm = d_delta.norm().item()
        d_eta_norm = d_eta.norm().item()
        
        if d_delta.abs().max().item() == 0:
            zero_params += 1
            status = "❌ ZÉRO"
        else:
            status = "✓"
        
        total_params += 1
        
        ratio_delta = d_delta_norm / (p_norm + 1e-10)
        ratio_eta = d_eta_norm / (p_norm + 1e-10)
        
        print(f"{status} {name:40s} | param_norm={p_norm:8.4f} | "
              f"delta_norm={d_delta_norm:8.4f} (ratio={ratio_delta:.4f}) | "
              f"eta_norm={d_eta_norm:8.4f} (ratio={ratio_eta:.4f})")
    
    print(f"\nRésumé: {zero_params}/{total_params} paramètres ont des directions nulles")
    
    if zero_params == total_params:
        print("⚠️  ATTENTION: TOUTES les directions sont nulles!")
    elif zero_params > total_params * 0.5:
        print("⚠️  ATTENTION: Plus de 50% des directions sont nulles!")
    else:
        print("✓ Les directions semblent correctes")


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
        resnet=model_cfg["resnet"],
        num_config=model_cfg["num_config"],
        use_skip=model_cfg["use_skip"],
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

    # OPTION: Choisir si on ignore BN/bias ou non
    # ignore_bn_bias=True  -> Directions nulles pour BN/bias (comme le papier original)
    # ignore_bn_bias=False -> Directions normalisées pour tous les paramètres
    IGNORE_BN_BIAS = True  # Changez à False pour tester

    if os.path.exists(directions_path):
        print("[INFO] Loading existing directions...")
        directions = torch.load(directions_path, map_location=device)
        delta = directions["delta"]
        eta = directions["eta"]
    else:
        print(f"[INFO] Generating filter-wise random directions (ignore_bn_bias={IGNORE_BN_BIAS})...")
        torch.manual_seed(0)  # Reproductibilité
        delta = random_directions_filterwise(model, ignore_bn_bias=IGNORE_BN_BIAS)
        eta = random_directions_filterwise(model, ignore_bn_bias=IGNORE_BN_BIAS)

        torch.save(
            {"delta": delta, "eta": eta},
            directions_path
        )
        print(f"[INFO] Directions saved to {directions_path}")

    # -------- Vérification --------
    verify_directions(model, delta, eta)
    
    print("\n[INFO] Directions ready!")
