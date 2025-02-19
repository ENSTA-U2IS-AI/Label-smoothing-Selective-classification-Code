import torch
from lightning.pytorch.cli import LightningArgumentParser
from torch_uncertainty.routines import ClassificationRoutine
from torch_uncertainty.utils import TULightningCLI

from torch_uncertainty_ls.datamodule import IMDBDataModule


class LSTMCLI(TULightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_optimizer_args(torch.optim.Adam)


def cli_main() -> LSTMCLI:
    return LSTMCLI(model_class=ClassificationRoutine, datamodule_class=IMDBDataModule)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    cli = cli_main()
    if (not cli.trainer.fast_dev_run) and cli.subcommand == "fit" and cli._get(cli.config, "eval_after_fit"):
        cli.trainer.test(datamodule=cli.datamodule, ckpt_path="best")
