from pathlib import Path
from typing import Any, Literal

import nltk
import torch
from nltk.tokenize import word_tokenize
from torch_uncertainty.datamodules.abstract import TUDataModule
from torch_uncertainty.utils import create_train_val_split

from torch_uncertainty_ls.dataset import IMDBReviews
from torch_uncertainty_ls.torchtext import GloVe, Truncate

GLOVE_SIZES = ["6B", "twitter.27B", "42B", "840B"]


def get_glove_params(size: int) -> dict[str, Any]:
    glove_params = {}
    if size == 6:
        glove_params["name"] = "6B"
        glove_params["dim"] = 50
    elif size == 27:
        glove_params["name"] = "twitter.27B"
        glove_params["dim"] = 200
    elif size == 42:
        glove_params["name"] = "42B"
        glove_params["dim"] = 300
    elif size == 840:
        glove_params["name"] = "840B"
        glove_params["dim"] = 300
    return glove_params


class IMDBDataModule(TUDataModule):
    """The IMDBDataModule for the IMDB datasets.

    Args:
        root (string): Root directory of the datasets.
        batch_size (int): The batch size for training and testing.
        val_split (float, optional): Share of validation samples. Defaults
            to ``0``.
        num_workers (int, optional): How many subprocesses to use for data
            loading. Defaults to ``1``.
        pin_memory (bool, optional): Whether to pin memory in the GPU. Defaults
            to ``True``.
        persistent_workers (bool, optional): Whether to use persistent workers.
            Defaults to ``True``.

    """

    num_classes = 2
    training_task = "classification"

    def __init__(
        self,
        root: str | Path,
        batch_size: int,
        glove_size: Literal[6, 27, 42, 840] = 840,
        max_seq_len: int = 256,
        val_split: float | None = None,
        num_workers: int = 1,
        pin_memory: bool = True,
        persistent_workers: bool = True,
    ) -> None:
        super().__init__(
            root=root,
            batch_size=batch_size,
            val_split=val_split,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        root = Path(root)
        nltk.download("punkt_tab")

        self.max_seq_len = max_seq_len
        self.dataset = IMDBReviews
        self.vocab = GloVe(cache=root / "GloVe", **get_glove_params(glove_size))
        self.truncate = Truncate(self.max_seq_len)

    def transform(self, x):
        tokens = word_tokenize(x)
        if len(tokens) < self.max_seq_len:
            tokens.extend(["<pad>"] * (self.max_seq_len - len(tokens)))
        text = self.truncate(tokens)
        vects = self.vocab.get_vecs_by_tokens(text)
        return torch.as_tensor(vects)

    def prepare_data(self) -> None:
        """Download the dataset."""
        self.dataset(root=self.root, train=True, download=True)
        self.dataset(root=self.root, train=False, download=True)

    def setup(self, stage: str | None = None) -> None:
        """Split the datasets into train, val, and test."""
        if stage == "fit" or stage is None:
            full = self.dataset(
                self.root,
                train=True,
                download=False,
                transform=self.transform,
            )
            if self.val_split:
                self.train, self.val = create_train_val_split(
                    full,
                    self.val_split,
                )
            else:
                self.train = full
                self.val = self.dataset(
                    self.root,
                    train=False,
                    download=False,
                    transform=self.transform,
                )

        elif stage == "test":
            self.test = self.dataset(
                root=self.root,
                train=False,
                download=False,
                transform=self.transform,
            )
