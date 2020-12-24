# SENet-CIFAR10
An implementation of the paper [_Squeeze-and-Excitation Networks_](https://arxiv.org/abs/1709.01507) on CIFAR10 dataset.

## how to run
Code: `python3 Cifar10.py`

For experiments with hyper-parameters, check `Cifar10.py`

## Experiment

## Implementations
- pytorch 1.0.1
- torchvision 0.2.2

### Conditions
- Data augmentation: pad=4, crop=32; horizontal flip
- optim: `default = SGD(lr=0.1,m=0.9,wd=1e-4, bs=128)`

### Experiments with different network archs and regularizations. 
| Base Network | Optim | Acc (Mine + SE + cutout=16) | Acc (Mine + SE) | Acc (Mine) | Acc (ResNet paper) |
|:------------:|:------:|:------:|:------:|:------:|:------:|
| res20 | default | 93.49 (+2.24) | 92.15 (+0.90) | 92.08 (+0.83) | 91.25 |
| res32 | default | 94.20 (+1.71) | 92.96 (+0.47) | 92.55 (+0.06) | 92.49 |
| res44 | default | 94.55 (+1.79) | 93.53 (+0.70) | 92.76 (-0.07) | 92.83 |
| res56 | default | 95.15 (+2.12) | 94.02 (+0.99) | 93.62 (+0.59) | 93.03 |
| res110 | default | 95.63 (+2.24) |94.70 (+1.31) | 93.70 (+0.31) | 93.39 |
| res20 | bs=64 | 93.46 (+2.21) | 92.79 (+1.54) | 92.50 (+1.25) | 91.25 |
| res110 | bs=64 | 95.85 (+2.22) | 94.81 (+1.18) | 94.61 (+0.98) | (93.63) |

### Experiments with Dropouts and Cutout

| experiment | network | Size(_Cutout_) | P(_dropout_) | Acc |
|:------:|:------:|:------:|:------:|:------:|
| baseline | res20 | - | - | 92.15 |
| -- | res20 | - | 0.1 | 92.35 (+ 0.20) |
| -- | res20 | - | 0.2 | 92.35 (+ 0.20) |
| -- | res20 | - | 0.4 | 92.03 (-0.12) |
| -- | res20 | - | 0.5 | 92.16 (+0.01) |
| -- | res20 | - | 0.6 | 92.15 (+ 0.00) |
| -- | res20 | - | 0.8 | 91.67 (-0.48) |
| -- | res20 | - | 0.9 | 89.69 (-2.46) |
| baseline | res20 | - | - | 92.15 |
| -- | res20 | 2 | - | 92.14 (-0.01) |
| -- | res20 | 4 | - | 92.76 (+0.61) |
| -- | res20 | 6 | - | 92.37 (+0.22) |
| -- | res20 | 8 | - | 93.12 (+0.97) |
| -- | res20 | 12 | - | 93.12 (+0.97) |
| -- | res20 | 16 | - | 93.27 (+1.12) |
| -- | res20 | 20 | - | 93.05 (+0.90) |

