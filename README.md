# Training to Reduce Catastrophic Forgetting

This project demonstrates continual learning on MNIST using a small CNN and three strategies to mitigate catastrophic forgetting:

- **Naive sequential training** (baseline)
- **Experience replay** (rehearsal buffer)
- **Elastic Weight Consolidation (EWC)** (regularization)

All experiments are contained in the notebook [adaptive_learning.ipynb](adaptive_learning.ipynb).

## Model overview

The model is a simple CNN with two convolutional blocks and a two-layer MLP head:

- Conv(1->16) + ReLU + MaxPool
- Conv(16->32) + ReLU + MaxPool
- Linear(32*7*7->128) + ReLU
- Linear(128->10)

It is trained on two tasks:

1. **Task 1:** Standard MNIST.
2. **Task 2:** Permuted MNIST (fixed random pixel permutation).

## Methods in the notebook

1. **Naive training**
	- Train on Task 1, then train on Task 2 without any protection.
2. **Experience replay**
	- Mix a small buffer of Task 1 samples into Task 2 training.
3. **EWC**
	- Estimate Fisher information after Task 1 and add a quadratic penalty during Task 2.

## Metrics and visualization

The notebook computes:

- Accuracy on Task 1 after Task 2 (forgetting).
- Accuracy on Task 2 after Task 2 (plasticity).
- Average accuracy across tasks.
- Forgetting measure: $\Delta = \text{peak}_\text{T1} - \text{acc}_\text{T1 after T2}$.

It also produces a plot comparing methods and a summary table.

## Requirements

- Python 3.9+
- PyTorch
- torchvision
- matplotlib
- tqdm
- numpy

## How to run

1. Open [adaptive_learning.ipynb](adaptive_learning.ipynb) in VS Code or Jupyter.
2. Run cells top-to-bottom.

## Notes

- The permutation for Task 2 is fixed with a single random permutation tensor.
- Training uses a small number of epochs per task for quick iteration.
