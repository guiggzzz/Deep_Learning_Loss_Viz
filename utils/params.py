import torch

def get_params(model):
    """
    Retourne tous les paramètres du modèle concaténés en 1D tensor.
    Utile pour générer le landscape.
    """
    return torch.cat([p.data.flatten() for p in model.parameters()])

def set_params(model, flat_params):
    """
    Remet les paramètres du modèle depuis un 1D tensor.
    """
    pointer = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(flat_params[pointer:pointer+numel].view_as(p))
        pointer += numel
