# Training an LSTM-based model on IMDB movie reviews

This example shows how to train an LSTM-based model on IMDB movie reviews with and without label-smoothing.

## Requirements

Make sure that the TorchUncertaintyLS package is installed as stated in the main README.md.

The dataset will be automatically downloaded through the TorchUncertainty datamodule.

## Training an LSTM-based model on IMDB movie reviews **without** label-smoothing

To train an LSTM-based model on IMDB movie reviews without label-smoothing, run the following command:

```bash
python lstm.py fit --config configs/lstm.yaml  --trainer.devices 0,
```

The training logs will be saved in the `classification/logs/lstm/` directory.

## Training an LSTM-based model on IMDB movie reviews **with** label-smoothing

To train an LSTM-based model on IMDB movie reviews with label-smoothing, run the following command:

```bash
python lstm.py fit --config configs/lstm_ls.yaml --trainer.devices 0,
```

The default label-smoothing value will be 0.2. Change the label-smoothing value either in the config file or by running the following command, 
adapting `your_value` to the desired value:

```bash
python lstm.py fit --config configs/lstm_ls.yaml --trainer.devices 0 --model.loss.label_smoothing your_value
```

The training logs will be saved in the `classification/logs/lstm_ls/` directory.

## Training curves

The training curves can be visualized using Tensorboard. To do so, run the following command:

```bash
tensorboard --logdir logs
```

Then, open a browser and go to `http://localhost:6006/` to visualize the training curves and the evolution of the metrics including the AURC, the Coverage @ 5% Risk and the Risk @ 80% Coverage.
