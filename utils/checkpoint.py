import torch

def save_checkpoint(model, optimizer=None, path="models/checkpoint.pt"):
    """Sauvegarde le modèle et éventuellement l’optimizer."""
    state = {"model_state": model.state_dict()}
    if optimizer is not None:
        state["optimizer_state"] = optimizer.state_dict()
    torch.save(state, path)
    print(f"[INFO] Checkpoint saved at {path}")

def load_checkpoint(model, path="models/checkpoint.pt", optimizer=None, map_location="cpu"):
    """Charge le modèle et optionnellement l’optimizer depuis un checkpoint."""
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    print(f"[INFO] Checkpoint loaded from {path}")
    return model, optimizer
