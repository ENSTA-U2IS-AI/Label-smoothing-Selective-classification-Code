import logging
from collections.abc import Callable
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.datasets.folder import make_dataset
from torchvision.datasets.utils import (
    check_integrity,
    download_and_extract_archive,
)


class IMDBReviews(Dataset):
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    filename = "aclImdb_v1.tar.gz"
    tgz_md5 = "7c2ac02c03563afcf9b574c7e56c153a"

    def __init__(
        self,
        root: Path | str,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        train: bool = True,
        download: bool = False,
    ):
        super().__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        if isinstance(root, str):
            self.root = Path(root)

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to " "download it.")

        data = make_dataset(
            directory=self.root / "aclImdb" / ("train" if train else "test"),
            class_to_idx={"neg": 0, "pos": 1},
            extensions=".txt",
        )
        self.data = [self._txt_loader(s[0]) for s in data]
        self.targets = torch.as_tensor([s[1] for s in data], dtype=torch.long)

    def _txt_loader(self, path: Path | str) -> str:
        with Path(path).open("r") as f:
            return f.read()

    def _check_integrity(self) -> bool:
        return check_integrity(
            self.root / self.filename,
            self.tgz_md5,
        )

    def download(self) -> None:
        if self._check_integrity():
            logging.info("Files already downloaded and verified")
            return

        download_and_extract_archive(
            url=self.url,
            download_root=self.root,
            extract_root=self.root,
            filename=self.filename,
            md5=self.tgz_md5,
        )

    def __getitem__(self, index: int) -> tuple[str, Tensor]:
        """Get sample and target for a given index."""
        data = self.data[index]
        if self.transform is not None:
            data = self.transform(data)
        target = self.targets[index]
        if self.target_transform is not None:
            target = self.target_transform(target)
        return data, target

    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        return len(self.data)
