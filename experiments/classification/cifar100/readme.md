# Training a DenseNet on CIFAR-100

This example shows how to train a DenseNet on CIFAR-100 with and without label-smoothing.

## Requirements

Make sure that the TorchUncertaintyLS package is installed as stated in the main README.md.

## Training a DenseNet on CIFAR-100 **without** label-smoothing

To train a DenseNet on CIFAR-100 without label-smoothing, run the following command:

```bash
python densenet.py fit --config configs/densenet.yaml --trainer.devices 0,
```

The training logs will be saved in the `classification/cifar100/logs/densenet` directory.

## Training a DenseNet on CIFAR-100 **with** label-smoothing

To train a DenseNet on CIFAR-100 with label-smoothing, run the following command:

```bash
python densenet.py fit --config configs/densenet_ls.yaml --trainer.devices 0,
```

The default label-smoothing value will be 0.2. Change the label-smoothing value either in the config file or by running the following command, 
adapting `your_value` to the desired value:

```bash
python densenet.py fit --config configs/densenet_ls.yaml --trainer.devices 0, --model.loss.label_smoothing your_value
```

The training logs will be saved in the `classification/cifar100/logs/densenet_ls` directory.

## Training curves

The training curves can be visualized using Tensorboard. To do so, run the following command:

```bash
tensorboard --logdir logs
```

Then, open a browser and go to `http://localhost:6006/` to visualize the training curves and the evolution of the metrics including the AURC, the Coverage @ 5% Risk and the Risk @ 80% Coverage.
