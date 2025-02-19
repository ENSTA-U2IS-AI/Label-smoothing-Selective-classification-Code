from typing import Literal

from lightning.pytorch.strategies import DDPStrategy
from torch import Tensor, nn
from torch.optim import Optimizer
from torch_uncertainty.post_processing import PostProcessing
from torch_uncertainty.routines import ClassificationRoutine

from torch_uncertainty_ls import SAM


class SAMClassificationRoutine(ClassificationRoutine):
    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        loss: nn.Module,
        is_ensemble: bool = False,
        format_batch_fn: nn.Module | None = None,
        optim_recipe: dict | Optimizer | None = None,
        mixup_params: dict | None = None,
        eval_ood: bool = False,
        eval_shift: bool = False,
        eval_grouping_loss: bool = False,
        ood_criterion: Literal["msp", "logit", "energy", "entropy", "mi", "vr"] = "msp",
        post_processing: PostProcessing | None = None,
        calibration_set: Literal["val", "test"] = "val",
        num_calibration_bins: int = 15,
        sam_gradient_clip_val: float = 0.0,
        sam_gradient_clip_algorithm: str = "value",
        log_plots: bool = False,
        save_in_csv: bool = False,
    ) -> None:
        """Classification routine using sharpness-aware minimization.

        Args:
            model (torch.nn.Module): Model to train.
            num_classes (int): Number of classes.
            loss (torch.nn.Module): Loss function to optimize the :attr:`model`.
            is_ensemble (bool, optional): Indicates whether the model is an
                ensemble at test time or not. Defaults to ``False``.
            format_batch_fn (torch.nn.Module, optional): Function to format the batch.
                Defaults to :class:`torch.nn.Identity()`.
            optim_recipe (dict or torch.optim.Optimizer, optional): The optimizer and
                optionally the scheduler to use. Defaults to ``None``.
            mixup_params (dict, optional): Mixup parameters. Can include mixup type,
                mixup mode, distance similarity, kernel tau max, kernel tau std,
                mixup alpha, and cutmix alpha. If None, no mixup augmentations.
                Defaults to ``None``.
            eval_ood (bool, optional): Indicates whether to evaluate the OOD
                detection performance. Defaults to ``False``.
            eval_shift (bool, optional): Indicates whether to evaluate the Distribution
                shift performance. Defaults to ``False``.
            eval_grouping_loss (bool, optional): Indicates whether to evaluate the
                grouping loss or not. Defaults to ``False``.
            ood_criterion (str, optional): OOD criterion. Available options are
                - ``"msp"`` (default): Maximum softmax probability.
                - ``"logit"``: Maximum logit.
                - ``"energy"``: Logsumexp of the mean logits.
                - ``"entropy"``: Entropy of the mean prediction.
                - ``"mi"``: Mutual information of the ensemble.
                - ``"vr"``: Variation ratio of the ensemble.
            post_processing (PostProcessing, optional): Post-processing method
                to train on the calibration set. No post-processing if None.
                Defaults to ``None``.
            calibration_set (str, optional): The post-hoc calibration dataset to
                use for the post-processing method. Defaults to ``val``.
            num_calibration_bins (int, optional): Number of bins to compute calibration
                metrics. Defaults to ``15``.
            sam_gradient_clip_val (float): The amount of gradient clipping done according to :attr:`sam_gradient_clip_algorithm`.
                Defaults to ``0``.
            sam_gradient_clip_algorithm (str): How to clip the gradient. Defaults to ``value``.
            log_plots (bool, optional): Indicates whether to log plots from
                metrics. Defaults to ``False``.
            save_in_csv(bool, optional): Save the results in csv. Defaults to
                ``False``.

        Warning:
            You must define :attr:`optim_recipe` if you do not use the Lightning CLI.

        Note:
            :attr:`optim_recipe` can be anything that can be returned by
            :meth:`LightningModule.configure_optimizers()`. Find more details
            `here <https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers>`_.

        """
        super().__init__(
            model=model,
            num_classes=num_classes,
            loss=loss,
            is_ensemble=is_ensemble,
            format_batch_fn=format_batch_fn,
            optim_recipe=optim_recipe,
            mixup_params=mixup_params,
            eval_ood=eval_ood,
            eval_shift=eval_shift,
            eval_grouping_loss=eval_grouping_loss,
            ood_criterion=ood_criterion,
            post_processing=post_processing,
            calibration_set=calibration_set,
            num_calibration_bins=num_calibration_bins,
            log_plots=log_plots,
            save_in_csv=save_in_csv,
        )
        self.automatic_optimization = False
        self.gradient_clip_val = sam_gradient_clip_val
        self.gradient_clip_algorithm = sam_gradient_clip_algorithm

    def configure_optimizers(self) -> Optimizer | dict:
        if isinstance(self.optim_recipe, dict) and not isinstance(self.optim_recipe["optimizer"], SAM):
            raise TypeError(f"Optimizer must be SAM. Got {type(self.optim_recipe['optimizer'])}")
        if isinstance(self.optim_recipe, Optimizer) and not isinstance(self.optim_recipe, SAM):
            raise TypeError(f"Optimizer must be SAM. Got {type(self.optim_recipe)}")
        return self.optim_recipe

    def training_step(self, batch: tuple[Tensor, Tensor]):
        loss = super().training_step(batch)
        loss_1 = loss.detach()
        optimizer: SAM = self.optimizers()
        # first forward-backward pass
        if isinstance(self.trainer.strategy, DDPStrategy):
            # gradients not synced for first step
            with self.trainer.model.no_sync():
                self.manual_backward(loss)
        else:
            self.manual_backward(loss)

        self.clip_gradients(
            optimizer,
            gradient_clip_val=self.gradient_clip_val,
            gradient_clip_algorithm=self.gradient_clip_algorithm,
        )

        optimizer.prestep(zero_grad=True)

        # second forward-backward pass
        loss = super().training_step(batch)
        loss_2 = loss.detach()
        self.manual_backward(loss)

        self.clip_gradients(
            optimizer,
            gradient_clip_val=self.gradient_clip_val,
            gradient_clip_algorithm=self.gradient_clip_algorithm,
        )
        optimizer.step()
        optimizer.zero_grad()
        return loss_1 + loss_2

    def on_train_epoch_end(self):
        sch = self.lr_schedulers()
        sch.step()
