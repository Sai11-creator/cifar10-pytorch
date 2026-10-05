# CIFAR-10 Image Classification with PyTorch

A PyTorch project for learning and experimenting with deep learning through
image classification on the CIFAR-10 dataset.

## Objectives

- Build CNN architectures from scratch
- Experiment with optimization and regularization
- Implement residual connections
- Compare custom models with pretrained architectures

## Dataset

CIFAR-10 contains 60,000 32×32 RGB images across 10 classes.

## Baseline Model

The current baseline is a small ResNet-style CNN trained on CIFAR-10.

The model uses residual blocks with Batch Normalization and ReLU activations, followed by global average pooling, dropout, and a final linear classifier.

### Baseline training setup

- Optimizer: Adam
- Learning rate: `1e-3`
- Weight decay: `1e-4`
- Learning-rate scheduler: StepLR
- Epochs: 15
- Batch size: 64
- Model selection: best validation accuracy

The training pipeline saves the checkpoint with the best validation performance rather than simply keeping the final epoch.

### Baseline result

- Test accuracy: **87.39%**

