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

## Configuration

Create YAML files in `configs/` to define model architectures:

```yaml
dataset:
  name: CIFAR10
  path: data/cifar10
  batch_size: 128
  num_workers: 4
  n_fixed_batches: 5

model:
  nn_architecture: "ResNet"       
  num_config: "18"        
  activation: relu  
  dropout: 0.0    

training:
  epochs: 5
  lr: 0.1
  momentum: 0.9
  nesterov: true
  weight_decay: 0.0005
  device: cuda   # if available, else cpu is automatic
```

## Project Structure

```
├── configs/              # Model configurations (YAML)
├── architecture.py       # ResNet & DenseNet definitions
├── train.py              # Training script
├── directions.py         # Generate random directions
├── landscape.py          # Loss landscape computation
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

