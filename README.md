# Finding Dynamic Lottery Tickets in One Training Cycle

**Author:** Romana Qureshi  
**Affiliation:** Department of Computer Science, King Saud University, Riyadh, Saudi Arabia

## Abstract

This repository implements a progressive dynamic pruning method that identifies and trains sparse neural networks during a single training cycle by incrementally increasing sparsity throughout the training process. Unlike traditional lottery ticket methods that require expensive iterative pruning and retraining cycles, our approach achieves **95.19% accuracy on CIFAR-10** and **99.37% accuracy on MNIST** with **79.6%** and **74.4%** sparsity respectively, while eliminating computational overhead.

## Key Features

- **Single-Pass Training**: No iterative retraining cycles required
- **Progressive Sparsification**: Gradual sparsity increase mimicking biological neural development
- **Dynamic Adaptation**: Continuously adapts pruning masks based on evolving weight magnitudes
- **Superior Performance**: Achieves competitive accuracy with substantial parameter reduction
- **Implementation Simplicity**: No complex regrowth mechanisms or specialized tracking systems

## Method Overview

Our approach implements **monotonic progressive pruning** where:

1. **Progressive Sparsity**: Linear increase from 0% to target sparsity throughout training
2. **Magnitude-Based Criterion**: Prune weights with smallest magnitudes at each epoch
3. **No Regrowth Policy**: Once pruned, connections remain permanently inactive
4. **Multi-Stage Masking**: Enforce sparsity during forward pass, gradient computation, and optimizer updates

### Biological Inspiration

The method draws inspiration from synaptic pruning in human brain development, where neural connections are gradually eliminated during maturation to create more efficient circuits.

## Results

### CIFAR-10 Performance
- **Test Accuracy**: 95.19% ± 0.11%
- **Sparsity**: 79.6%
- **Parameter Reduction**: 79.6% (from 11.2M to 2.3M parameters)
- **Dense Baseline**: 95.27% (only 0.08pp degradation)

### MNIST Performance
- **Test Accuracy**: 99.37% ± 0.02%
- **Sparsity**: 74.4%
- **Parameter Reduction**: 74.4% (from 11.2M to 2.9M parameters)
- **Dense Baseline**: 99.51% (only 0.14pp degradation)

## Repository Structure

```
dynamic-lottery-tickets/
├── src/
│   ├── models/
│   │   ├── resnet.py              # ResNet-18 architecture
│   │   └── dynamic_pruning.py     # Progressive pruning implementation
│   ├── datasets/
│   │   ├── cifar10_loader.py      # CIFAR-10 data loading
│   │   └── mnist_loader.py        # MNIST data loading
│   ├── training/
│   │   ├── train_cifar10.py       # CIFAR-10 training script
│   │   ├── train_mnist.py         # MNIST training script
│   │   └── train_dense.py         # Dense baseline training
│   └── utils/
│       ├── masking.py             # Multi-stage weight masking
│       ├── metrics.py             # Evaluation metrics
│       └── visualization.py       # Training dynamics plots
├── experiments/
│   ├── cifar10_experiments.py     # CIFAR-10 experimental setup
│   ├── mnist_experiments.py       # MNIST experimental setup
│   └── baseline_comparison.py     # Dense vs sparse comparison
├── configs/
│   ├── cifar10_config.yaml        # CIFAR-10 hyperparameters
│   └── mnist_config.yaml          # MNIST hyperparameters
└── results/
    ├── cifar10_results.json       # CIFAR-10 experimental results
    ├── mnist_results.json         # MNIST experimental results
    └── plots/                     # Generated visualization plots
```

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/dynamic-lottery-tickets.git
cd dynamic-lottery-tickets

# Create conda environment
conda create -n dynamic-lottery python=3.8
conda activate dynamic-lottery

# Install dependencies
pip install torch torchvision numpy matplotlib seaborn pyyaml
```

## Quick Start

### CIFAR-10 Experiments

```bash
# Run progressive dynamic pruning on CIFAR-10
python src/training/train_cifar10.py --config configs/cifar10_config.yaml

# Run dense baseline for comparison
python src/training/train_dense.py --dataset cifar10 --config configs/cifar10_config.yaml

# Run full experimental suite with multiple seeds
python experiments/cifar10_experiments.py
```

### MNIST Experiments

```bash
# Run progressive dynamic pruning on MNIST
python src/training/train_mnist.py --config configs/mnist_config.yaml

# Run dense baseline for comparison
python src/training/train_dense.py --dataset mnist --config configs/mnist_config.yaml

# Run full experimental suite
python experiments/mnist_experiments.py
```

### Dense Baseline Comparison

```bash
# Compare sparse vs dense training
python experiments/baseline_comparison.py --dataset cifar10
python experiments/baseline_comparison.py --dataset mnist
```

## Configuration

### CIFAR-10 Settings
- **Architecture**: ResNet-18
- **Training**: 200 epochs, SGD with momentum 0.9
- **Learning Rate**: 0.1 with cosine annealing
- **Target Sparsity**: 79.6%
- **Batch Size**: 128

### MNIST Settings  
- **Architecture**: ResNet-18 (adapted for 1-channel input)
- **Training**: 30 epochs, SGD with momentum 0.9
- **Learning Rate**: 0.1 with cosine annealing
- **Target Sparsity**: 74.4%
- **Batch Size**: 32

## Algorithm Details

The progressive dynamic pruning algorithm works as follows:

1. **Initialize**: All weights active (mask = 1)
2. **For each epoch**:
   - Compute target sparsity: `Se = min(Sfinal, ΔS × e)`
   - Identify threshold for Se percentile of active weights
   - Update mask monotonically (no regrowth)
   - Enforce sparsity by zeroing pruned weights
3. **For each batch**:
   - Apply pre-forward masking
   - Compute forward and backward passes
   - Mask gradients to prevent updates to pruned weights
   - Apply post-optimizer enforcement to handle momentum

## Key Advantages

- **Computational Efficiency**: Single training cycle vs multiple iterations
- **Performance**: Competitive accuracy with minimal degradation
- **Simplicity**: No complex regrowth or tracking mechanisms  
- **Adaptability**: Dynamic mask updates throughout training
- **Regularization**: Progressive sparsification prevents overfitting

## Experimental Validation

All experiments include:
- **Multiple Random Seeds**: Statistical validation across 5 seeds (CIFAR-10) and 3 seeds (MNIST)
- **Proper Methodology**: Train-validation-test splits with rigorous evaluation
- **Comprehensive Baselines**: Comparison with dense networks and literature methods
- **Ablation Studies**: Analysis of sparsity schedules, masking strategies, and design choices

## Comparison with Literature

| Method | Dataset | Sparsity | Accuracy | Training Cycles |
|--------|---------|----------|----------|----------------|
| Traditional LTH | CIFAR-10 | 80% | 91.2% | Multiple (3-10x) |
| SNIP | CIFAR-10 | 95% | 92.8% | Single |
| RigL | CIFAR-10 | 80% | 92.1% | Single |
| **Ours** | **CIFAR-10** | **79.6%** | **95.19%** | **Single** |

## Reproducing Results

### Statistical Validation
- CIFAR-10: 5 random seeds [42, 123, 456, 789, 999]
- MNIST: 3 random seeds [42, 123, 456]
- All results include mean ± standard deviation

### Expected Outputs
- **CIFAR-10**: 95.19% ± 0.11% test accuracy at 79.6% sparsity
- **MNIST**: 99.37% ± 0.02% test accuracy at 74.4% sparsity

### Custom Sparsity Schedule
```python
# Linear sparsity schedule (default)
sparsity_schedule = "linear"

# Alternative schedules
# sparsity_schedule = "exponential"  # 94.87% ± 0.15%
# sparsity_schedule = "polynomial"   # 94.95% ± 0.13%
# sparsity_schedule = "cosine"       # 94.91% ± 0.12%
```

## Hardware Requirements

- **GPU**: NVIDIA GPU with CUDA support recommended
- **Memory**: 8GB+ GPU memory for CIFAR-10, 4GB+ for MNIST
- **Training Time**: 
  - CIFAR-10: ~2-3 hours on RTX 3080
  - MNIST: ~10-15 minutes on RTX 3080

## Contact

For questions or collaboration:
- **Email**: rq.romana@gmail.com
- **Institution**: King Saud University, Riyadh, Saudi Arabia
