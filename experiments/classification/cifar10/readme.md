# Training a ResNet-20 on CIFAR-10

This example shows how to train a ResNet-20 on CIFAR-10 with and without label-smoothing.

## Requirements


## Training a ResNet-20 on CIFAR-10 **without** label-smoothing

To train a ResNet-20 on CIFAR-10 without label-smoothing, run the following command:

```bash
python resnet.py fit --config configs/resnet20.yaml --trainer.devices 0,
```

The training logs will be saved in the `classification/cifar10/logs/resnet20` directory.

## Training a ResNet-20 on CIFAR-10 **with** label-smoothing

To train a ResNet-20 on CIFAR-10 with label-smoothing, run the following command:

```bash
python resnet.py fit --config configs/resnet20_ls.yaml --trainer.devices 0,
```

The default label-smoothing value will be 0.2. Change the label-smoothing value either in the config file or by running the following command, 
adapting `your_value` to the desired value:

```bash
python resnet.py fit --config configs/resnet_20_ls.yaml --trainer.devices 0, --model.loss.label_smoothing your_value
```

The training logs will be saved in the `classification/cifar10/logs/resnet20_ls` directory.

## Training curves

The training curves can be visualized using Tensorboard. To do so, run the following command:

```bash
tensorboard --logdir logs
```

Then, open a browser and go to `http://localhost:6006/` to visualize the training curves and the evolution of the metrics including the AURC, the Coverage @ 5% Risk and the Risk @ 80% Coverage.
