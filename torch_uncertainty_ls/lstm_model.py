import torch
from torch import nn
from torch.nn import functional as F


class LSTMNet(nn.Module):
    def __init__(self) -> None:
        """Simple LSTM-based model followed by a 3-layer-perceptron."""
        super().__init__()
        self.lstm = nn.LSTM(300, 256, batch_first=True)
        self.dr1 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(256, 256)
        self.dr2 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the prediction given an input.

        Args:
            x (Tensor): the input tensor.

        Returns:
            Tensor: the prediction.

        """
        return self.fc3(F.relu(self.fc2(self.dr2(F.relu(self.fc1(self.dr1(self.lstm(x)[0][:, -1, :])))))))
