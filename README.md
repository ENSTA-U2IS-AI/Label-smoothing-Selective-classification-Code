# Towards Understanding Why Label Smoothing Degrades Selective Classification and How to Fix It

[![arXiv](https://img.shields.io/badge/arXiv-2403.14715-b31b1b.svg)](https://arxiv.org/abs/2403.14715)
[![Models on Hugging Face](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm-dark.svg)](https://huggingface.co/ENSTA-U2IS/Label-smoothing-Selective-classification)

These experiments build on top of [TorchUncertainty](https://torch-uncertainty.github.io/).

## Requirements

Make sure that TorchUncertainty >= 0.4.0 is installed.

First, install your desired PyTorch version for instance with 

```bash
pip3 install torch torchvision
```

Then install the local torch-uncertainty-ls on your computer with:

```bash
pip install -e .
```

The latest release of [TorchUncertainty](https://github.com/ENSTA-U2IS/torch-uncertainty) will be installed automatically.

### LaTeX

You will need LaTeX installed on your computer to plot the figures that include LaTeX symbols.

## How to use this software

### Experiments

The `experiments` folder provides the configuration files and the commands to run and reproduce the results of the paper. The tabular data experiments are small scale and therefore included in separate notebooks.

Please find the commands in the `readme.md` files contained in each of the subfolders.

If you do not wish to perform the experiments on your machine, you may download the models directly on [HuggingFace](https://huggingface.co/ENSTA-U2IS/Label-smoothing-Selective-classification).

### Notebooks

The `notebooks` folder provides jupyter notebooks to reproduce the plots made in the paper. Just update the paths to the models that you trained to create your own figures.

We include:
- CIFAR-100 with DenseNet
- ImageNet-1k with VIT-S-16 and with ResNet-50 (the latter with logit normalization and hyperparameter optimization)
- Cityscapes with DeepLabV3+ (ResNet-101 backbone)
- IMDB with an LSTM-based model

### Citation

If you find this work helpful for your research, consider citing:
```
@inproceedings{xia2024understanding,
    title={Towards Understanding Why Label Smoothing Degrades Selective Classification and How to Fix It},
    author={Xia, Guoxuan and Laurent, Olivier and Franchi, Gianni and Bouganis, Christos-Savvas},
    booktitle={ICLR},
    year={2025}       
}
```

You will find 3 other notebooks directly in the `experiments` folder:
- Bank Marteking with multilayer perceptron in `experiments/tabular`
- Online Shoppers with multilayer perceptron in `experiments/tabular`
- CIFAR-10 with ResNet-20 in `experiments/cifar10`
