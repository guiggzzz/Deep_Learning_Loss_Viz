import os
import sys
import yaml
import numpy as np
import matplotlib

# Backend non-interactif (serveur / cluster)
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

    # -------- Paths --------
    input_path = f"plot_resources/Z_{config_name}.npy"
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)

    output_path = os.path.join(
        plots_dir, f"loss_landscape_3D_{config_name}.png"
    )

    # -------- Landscape params (doivent matcher landscape.py) --------
    alpha_range = (-1.0, 1.0)
    beta_range  = (-1.0, 1.0)
    n_points = 25

    # -------- Load Z --------
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Landscape file not found: {input_path}"
        )

    Z = np.load(input_path)

    alphas = np.linspace(alpha_range[0], alpha_range[1], n_points)
    betas  = np.linspace(beta_range[0], beta_range[1], n_points)
    A, B = np.meshgrid(alphas, betas)

    # -------- Plot 3D --------
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        A, B, Z.T,
        cmap="viridis",
        edgecolor="k",
        linewidth=0.2,
        antialiased=True
    )

    ax.set_xlabel("Alpha")
    ax.set_ylabel("Beta")
    ax.set_zlabel("Loss")
    ax.set_title(f"Loss Landscape – {config_name}")

    fig.colorbar(surf, shrink=0.5, aspect=10)

    # -------- Save --------
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[INFO] Plot saved to {output_path}")
