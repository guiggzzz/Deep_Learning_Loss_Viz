# Deep Learning Loss Landscape Visualization
Visualize and analyze the loss landscape geometry of deep neural networks. Implementation based on Visualizing the Loss Landscape of Neural Nets.
Examples
ResNet-56ResNet-18 + DropoutDenseNet-121Smooth, convexMultiple minimaComplex geometry
Quick Start
bash# Install dependencies
pip install -r requirements.txt

# Run complete pipeline (train + landscape + plot)
python main.py ResNet-56

# Or process all configs
python main.py all
Usage
Basic Pipeline
bashpython train.py <config_name>      # Train model
python directions.py <config_name>  # Generate directions
python landscape.py <config_name>   # Compute landscape
python plot.py <config_name>        # Create visualization
Advanced Analysis
bash# Compute metrics (sharpness, convexity, F1 score)
python analyze_metrics.py <config_name>

# Enhanced plots with metrics
python plot_with_metrics.py <config_name>

Project Structure
├── configs/             # Model configurations (YAML)
├── architecture.py      # ResNet & DenseNet definitions
├── train.py             # Training script
├── directions.py        # Define the plot directions
├── landscape.py         # Loss landscape computation
├── plot.py              # Visualization
├── main.py              # Pipeline orchestrator
└── utils/               # Data loading & checkpoints
Output

models/ - Model checkpoints (.pt)
plot_resources/ - Directions & landscape data (.npy)
plots/ - Visualizations (.png)
metrics/ - Quantitative analysis (.json)