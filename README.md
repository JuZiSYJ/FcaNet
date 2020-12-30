# FcaNet-CIFAR
An implementation of the paper [FcaNet: Frequency Channel Attention Networks](https://arxiv.org/abs/2012.11879) on CIFAR10/CIFAR100 dataset.

## how to run
Code: `python Cifar.py --network fca_resnet20 `

## Notes
- This project is only for my own study purpose. Please don't star this project because I'm not one of the authors. If you want to try FcaNet, welcome to use the codes and follow the author of the paper--[cfzd](https://github.com/cfzd).
- The basic code architecture is based on [SENet-cifar10](https://github.com/Jyouhou/SENet-cifar10). Very few tricks are utilized, so the performance may not be satisfying.
- The CIFAR datasets are pretty small compared with ImageNet, so the experiments are not stable and representative for verifying the algorithm. More experiments on ImageNet will coming soon.
## Experiment

## Denpendencies
- python 3.7
- pytorch 1.4.0
- torchvision 0.5.0

### Conditions
- Data augmentation: pad=4, crop=32; horizontal flip
- optim: `default = SGD(lr=0.1,m=0.9,wd=1e-4, bs=128)`

### Experiments with different network archs and attentions. 
2020-12-28 update: Fix the DCT weights indexes calculation. Also referring to [author's comment in zhihu](https://zhuanlan.zhihu.com/p/338904015)
| Base Network  | Dataset | Acc (ResNet + SE) | Acc (ResNet + FCA)  |
|:------------:|:------:|:------:|:------:|
| resnet20 | CIFAR10 | 92.30 | 92.49 (+0.19)|
| resnet20 | CIFAR100 | 68.81 | 69.00 (+0.19)|
| res44 | CIFAR10  |  - | -  |
| res44 | CIFAR100  |  71.75 | 71.9(+0.15)  |
| res56 | CIFAR10  |  - | -  |
| res56 | CIFAR100  |  72.33 | 72.28  |

### Ablation Study about dct_weights
refer to [the comments in zhihu](https://zhuanlan.zhihu.com/p/338904015).
| Dataset | network | DCT_Weight | Acc |
|:------:|:------:|:------:|:------:|
| CIFAR100 | resnet20 |DCT+Buffer (default)| 69.00 |
| CIFAR100 | resnet20 |DCT+Param | 68.76 |
| CIFAR100 | resnet20 |Rand+Param| 64.67 |


