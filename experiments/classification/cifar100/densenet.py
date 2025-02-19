import torch
from lightning.pytorch.cli import LightningArgumentParser
from torch_uncertainty.datamodules import CIFAR100DataModule
from torch_uncertainty.routines import ClassificationRoutine
from torch_uncertainty.utils import TULightningCLI


class DenseNetCLI(TULightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_optimizer_args(torch.optim.SGD)
        parser.add_lr_scheduler_args(torch.optim.lr_scheduler.MultiStepLR)


def cli_main() -> DenseNetCLI:
    return DenseNetCLI(model_class=ClassificationRoutine, datamodule_class=CIFAR100DataModule)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    cli = cli_main()
    if (not cli.trainer.fast_dev_run) and cli.subcommand == "fit" and cli._get(cli.config, "eval_after_fit"):
        cli.trainer.test(datamodule=cli.datamodule, ckpt_path="best")
