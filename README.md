# Finding Dynamic Lottery Tickets in One Training Cycle

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
AI_Research_Dynamic_NN/
├── 00_Implementation_On_Cifar10.ipynb # CIFAR-10 with progressive pruning
├── 01_Implementation_On_MNIST.ipynb # MNIST experiment
├── Dense_Baseline_Comparison_Code.ipynb # Dense baseline for comparison
├── algorithm_diagram.png 
├── diagram2.pdf 
└── README.md # This file                          


## Getting Started

The implementation is provided through three Jupyter notebooks that contain all the code and experiments:

1. **`00_Implementation_On_Cifar10.ipynb`** — Main CIFAR-10 implementation
2. **`01_Implementation_On_MNIST.ipynb`** — MNIST experiments
3. **`Dense_Baseline_Comparison_Code.ipynb`** — Dense baseline comparisons

Simply open any notebook in Jupyter or Google Colab and run all cells to reproduce the results.

## Quick Start

### CIFAR-10 Experiments

```bash
jupyter notebook 00_Implementation_On_Cifar10.ipynb
```

Includes:

* Progressive dynamic pruning implementation
* Training with 79.6% target sparsity
* Statistical validation across multiple seeds
* Performance visualization and analysis

### MNIST Experiments

```bash
jupyter notebook 01_Implementation_On_MNIST.ipynb
```

Includes:

* MNIST-specific ResNet-18 adaptation (1-channel input)
* Progressive pruning to 74.4% sparsity
* Rapid convergence analysis
* Cross-validation results

### Dense Baseline Comparison

```bash
jupyter notebook Dense_Baseline_Comparison_Code.ipynb
```

Includes:

* Side-by-side comparison of dense vs sparse training
* Performance metrics and parameter reduction analysis
* Efficiency-performance trade-off evaluation
* Statistical significance testing

## Configuration

### CIFAR-10 Settings

* **Architecture**: ResNet-18
* **Training**: 200 epochs, SGD with momentum 0.9
* **Learning Rate**: 0.1 with cosine annealing
* **Target Sparsity**: 79.6%
* **Batch Size**: 128

### MNIST Settings

* **Architecture**: ResNet-18 (adapted for 1-channel input)
* **Training**: 30 epochs, SGD with momentum 0.9
* **Learning Rate**: 0.1 with cosine annealing
* **Target Sparsity**: 74.4%
* **Batch Size**: 32

## Algorithm Details

The progressive dynamic pruning algorithm works as follows:

1. **Initialize**: All weights active (mask = 1)
2. **For each epoch**:

   * Compute target sparsity: `Se = min(Sfinal, ΔS × e)`
   * Identify threshold for Se percentile of active weights
   * Update mask monotonically (no regrowth)
   * Enforce sparsity by zeroing pruned weights
3. **For each batch**:

   * Apply pre-forward masking
   * Compute forward and backward passes
   * Mask gradients to prevent updates to pruned weights
   * Apply post-optimizer enforcement to handle momentum

### Algorithm Diagram

![Progressive Dynamic Pruning Algorithm](algorithm_diagram.png)

*Figure: Overview of the progressive dynamic pruning process applied during training.*

## Key Advantages

* **Computational Efficiency**: Single training cycle vs multiple iterations
* **Performance**: Competitive accuracy with minimal degradation
* **Simplicity**: No complex regrowth or tracking mechanisms
* **Adaptability**: Dynamic mask updates throughout training
* **Regularization**: Progressive sparsification prevents overfitting

## Experimental Validation

All experiments include:

* **Multiple Random Seeds**: 5 seeds for CIFAR-10, 3 seeds for MNIST
* **Proper Methodology**: Train-validation-test splits with rigorous evaluation
* **Comprehensive Baselines**: Dense vs sparse networks
* **Ablation Studies**: Sparsity schedules, masking strategies, and design choices

## Comparison with Literature

| Method          | Dataset  | Sparsity | Accuracy   | Training Cycles  |
| --------------- | -------- | -------- | ---------- | ---------------- |
| Traditional LTH | CIFAR-10 | 80%      | 91.2%      | Multiple (3–10x) |
| SNIP            | CIFAR-10 | 95%      | 92.8%      | Single           |
| RigL            | CIFAR-10 | 80%      | 92.1%      | Single           |
| **Ours**        | CIFAR-10 | 79.6%    | **95.19%** | **Single**       |

## Reproducing Results

### Statistical Validation

* CIFAR-10: seeds \[42, 123, 456, 789, 999]
* MNIST: seeds \[42, 123, 456]
* Results reported as mean ± standard deviation

### Expected Outputs

* **CIFAR-10**: 95.19% ± 0.11% at 79.6% sparsity
* **MNIST**: 99.37% ± 0.02% at 74.4% sparsity

## Hardware Requirements

* **GPU**: NVIDIA GPU with CUDA support recommended
* **Memory**: 8GB+ for CIFAR-10, 4GB+ for MNIST

## Author

**Romana Qureshi**
Master’s student in Artificial Intelligence
Department of Computer Science
King Saud University, Riyadh, Saudi Arabia
📧 [rq.romana@gmail.com](mailto:rq.romana@gmail.com)
