import torch
from lightning.pytorch.cli import LightningArgumentParser
from torch_uncertainty.datamodules import ImageNetDataModule
from torch_uncertainty.optim_recipes import CosineAnnealingWarmup
from torch_uncertainty.utils import TULightningCLI

from torch_uncertainty_ls import SAM, SAMClassificationRoutine


class ViTCLI(TULightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_optimizer_args(SAM)
        parser.add_lr_scheduler_args(CosineAnnealingWarmup)


def cli_main() -> ViTCLI:
    return ViTCLI(model_class=SAMClassificationRoutine, datamodule_class=ImageNetDataModule)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    cli = cli_main()
    if (not cli.trainer.fast_dev_run) and cli.subcommand == "fit" and cli._get(cli.config, "eval_after_fit"):
        cli.trainer.test(datamodule=cli.datamodule, ckpt_path="best")
