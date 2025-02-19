# Training a DeepLabv3+ on Cityscapes

This example shows how to train a DeepLabv3+ on Cityscapes with and without label-smoothing.

## Requirements

Make sure that the TorchUncertaintyLS package is installed as stated in the main README.md.

Download the Cityscapes dataset from the official website: https://www.cityscapes-dataset.com/.
Put the zips `gtFine_trainvaltest.zip` and `leftImg8bit_trainvaltest.zip` in `./data/Cityscapes`.
The extraction will be done automatically.


## Training a DeepLabv3+ on Cityscapes **without** label-smoothing

To train a DeepLabv3+ on Cityscapes without label-smoothing, run the following command:

```bash
python deeplab.py fit --config configs/deeplab.yaml  --trainer.devices [0,1]
```

The training logs will be saved in the `classification/cityscapes/logs/deeplab` directory.

## Training a DeepLabv3+ on Cityscapes **with** label-smoothing

To train a DeepLabv3+ on Cityscapes with label-smoothing, run the following command:

```bash
python deeplab.py fit --config configs/deeplab_ls.yaml --trainer.devices [0,1]
```

The default label-smoothing value will be 0.2. Change the label-smoothing value either in the config file or by running the following command, 
adapting `your_value` to the desired value:

```bash
python deeplab.py fit --config configs/deeplab_ls.yaml --trainer.devices [0,1] --model.loss.label_smoothing your_value
```

The training logs will be saved in the `classification/cityscapes/logs/deeplab_ls` directory.

## Training curves

The training curves can be visualized using Tensorboard. To do so, run the following command:

```bash
tensorboard --logdir logs
```

Then, open a browser and go to `http://localhost:6006/` to visualize the training curves and the evolution of the metrics including the AURC, the Coverage @ 5% Risk and the Risk @ 80% Coverage.
