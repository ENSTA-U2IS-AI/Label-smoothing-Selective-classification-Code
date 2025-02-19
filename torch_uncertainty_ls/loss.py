from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.nn.functional import log_softmax


class CrossEntropyLossGLS(CrossEntropyLoss):
    def __init__(
        self,
        weight: Tensor | None = None,
        reduction: str = "mean",
        label_smoothing: float = 0,
    ) -> None:
        super().__init__(weight=weight, reduction=reduction, label_smoothing=label_smoothing)

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        if self.label_smoothing >= 0:
            return super().forward(preds, targets)
        return self._nls_forward(preds, targets)

    def _nls_forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        logprobs = log_softmax(preds, dim=-1)
        smooth_loss = -logprobs.mean(dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
        loss = (1 - self.label_smoothing) * nll_loss + self.label_smoothing * smooth_loss
        if self.weight is not None:
            loss = loss * self.weight
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class OriginalCrossEntropyLossGLS(CrossEntropyLoss):
    def __init__(
        self,
        weight: Tensor | None = None,
        reduction: str = "mean",
        label_smoothing: float = 0,
    ) -> None:
        super().__init__(weight=weight, reduction=reduction, label_smoothing=label_smoothing)

    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        confidence = 1.0 - self.label_smoothing
        logprobs = log_softmax(logits, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=labels.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.label_smoothing * smooth_loss
        loss_numpy = loss.data.cpu().numpy()
        num_batch = len(loss_numpy)
        return loss.sum() / num_batch
