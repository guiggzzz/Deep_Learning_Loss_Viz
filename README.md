# Deep Learning Loss Landscape Visualization

Visualize and analyze the loss landscape geometry of deep neural networks. Implementation based on [Visualizing the Loss Landscape of Neural Nets](https://arxiv.org/abs/1712.09913).

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run complete pipeline (train + landscape + plot)
python main.py ResNet-56

# Or process all configs
python main.py all
```

## Usage

### Basic Pipeline

```bash
python train.py <config_name>       # Train model
python directions.py <config_name>  # Generate directions
python landscape.py <config_name>   # Compute landscape
python plot.py <config_name>        # Create visualization
```

### Advanced Analysis

```bash
# Compute metrics (sharpness, convexity, F1 score)
python analyze_metrics.py <config_name>

# Enhanced plots with metrics overlay
python plot_with_metrics.py <config_name>

# Compare all configurations
python analyze_metrics.py all
python plot_with_metrics.py all
```

## Configuration

Create YAML files in `configs/` to define model architectures:

```yaml
model:
  resnet: true              # true=ResNet, false=DenseNet
  num_config: "18"          # 18, 34, 56, 110, 121, 169
  use_skip: true            # Enable skip connections
  activation: "relu"        # relu, gelu, silu
  dropout: 0.0

training:
  epochs: 100
  lr: 0.1
  momentum: 0.9
  device: "cuda"

dataset:
  batch_size: 128
  num_workers: 4
```

## Project Structure

```
├── configs/              # Model configurations (YAML)
├── architecture.py       # ResNet & DenseNet definitions
├── train.py              # Training script
├── directions.py         # Generate random directions
├── landscape.py          # Loss landscape computation
├── analyze_metrics.py    # Compute landscape metrics
├── plot.py               # Basic visualization
├── main.py               # Pipeline orchestrator
└── utils/                # Data loading & checkpoints
```

## Output

```
models/          - Model checkpoints (.pt)
plot_resources/  - Directions & landscape data (.npy)
plots/           - Visualizations (.png)
```

