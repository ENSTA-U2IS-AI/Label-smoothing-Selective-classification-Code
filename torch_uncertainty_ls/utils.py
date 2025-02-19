import matplotlib as mpl
import torch
from matplotlib import pyplot as plt
from torch_uncertainty import routines


def risk_coverage_curve(
    y_true: torch.Tensor, y_score: torch.Tensor, sample_weight: torch.Tensor | None | int = None, dev: str = "cpu"
):
    if sample_weight is None:
        sample_weight = 1
    sorted_idx = y_score.argsort(descending=True)
    # risk for each coverage value rather than recall
    # add one to cover situation with zero coverage, assume risk is zero
    # when nothing is selected
    coverage = torch.linspace(0, 1, len(y_score) + 1).to(dev)
    # invert labels to get invalid predictions
    sample_costs = ~(y_true.to(bool)) * sample_weight
    sorted_cost = sample_costs[sorted_idx]
    summed_cost = torch.cumsum(sorted_cost, 0)
    n_selected = torch.arange(1, len(y_score) + 1).to(dev)
    # zero risk when none selected
    risk = torch.cat([torch.zeros(1).to(dev), summed_cost / n_selected])
    thresholds = y_score[sorted_idx]  # select >= threshold
    return risk, coverage, thresholds


def norm_logits(logits: torch.Tensor, p: int = 2, add_unit: float | str = 0):
    """Perform logit normalization.

    Args:
        logits (Tensor): the logits to be normalized.
        p (int): the dimension of the norm to normalize with.
        add_unit (int): how to change the magnitude of the logits (leaves the prediction unchanged).

    Returns:
        Tensor: the normalized logits.

    Note:
        We noticed that converting the logits to float64 was important as some normlized values will be very similar.

    """
    logits = logits.to(torch.double)
    if isinstance(add_unit, float):
        logits += add_unit
    elif isinstance(add_unit, str):
        if add_unit == "mean":
            logits -= logits.mean(0)
        elif add_unit == "min":
            logits -= logits.min(0).values
        else:
            raise ValueError
    else:
        raise TypeError
    return logits / logits.norm(p=p, dim=0)


def plot_segmentation_figure(
    val_dst: torch.utils.data.Dataset,
    routines: list[routines.ClassificationRoutine],
    device: str,
    ls_value: float,
    cbar,
    heights: dict,
    widths: dict,
    stats: dict,
    img_id: int,
    save: bool = False,
    full_plot: bool = False,
    plot: bool = True,
):
    """Plot the segmentation figures from the paper.

    Args:
        val_dst (torch.utils.data.Dataset): the validation dataset.
        routines (list[routines.ClassificationRoutine]): the routines for the computation of the predictions (~models).
        device (str): the device on which to make the predictions.
        ls_value (float): the label-smoothing value used during training (to be written on the plot).
        cbar: the colorbar for the plot.
        heights (dict): the min and max heights of the final plot.
        widths (dict): the min and max widths of the final plot.
        stats (dict): the mean and standard deviation of the inputs for standardization.
        img_id (int): the id of the image to plot in the dataset.
        save (bool): Whether to save the image on disk. Defaults to False.
        full_plot (bool): Whether to show the rankings of both incorrect and correct predictions. Only incorrect rankins are
            shown if False. Defaults to False.
        plot (bool): Whether to show the image. Defaults to True.

    """
    global ce_pred, ls_pred, unlbld_mask, ce_most_conf_errors, ls_most_conf_errors, ce_errors, ls_errors

    height, min_height, max_height = heights["full"], heights["min"], heights["max"]
    width, min_width, max_width = widths["full"], widths["min"], widths["max"]
    routine, routine_ls = routines
    mean, std = stats["mean"], stats["std"]

    with torch.no_grad():
        img, tgt = val_dst.__getitem__(img_id)
        img = img.to(device).unsqueeze(0)
        tgt = (
            torch.nn.functional.interpolate(tgt.double().unsqueeze(0), (height, width), mode="nearest")
            .to(device, dtype=torch.long)
            .squeeze(0)
        )
        unlbld_mask = tgt.cpu() != 255
        unlbld_mask = (
            torch.nn.functional.interpolate(unlbld_mask.double().unsqueeze(0), (height, width), mode="nearest")
            .to(device, dtype=torch.long)
            .squeeze(0)[0]
            .cpu()
        )
        ce_pred = routine(img)[0].double()
        ce_pred = torch.nn.functional.interpolate(ce_pred.unsqueeze(0), (height, width), mode="bilinear").to(device)[0].cpu()

        ls_pred = routine_ls(img)[0].double()
        ls_pred = torch.nn.functional.interpolate(ls_pred.unsqueeze(0), (height, width), mode="bilinear").to(device)[0].cpu()

    ce_cls = ce_pred.softmax(0).max(0).indices.cpu()

    # Get the top label "probabilities"
    ce_prob_top_label = ce_pred.softmax(0).max(0).values.clone().cpu()
    # Remove the values corresponding to the unlabelled pixels
    ce_prob_top_label[unlbld_mask == 0] = -1
    # Sort the values in ascending order
    sorted_indices = torch.argsort(ce_prob_top_label.flatten())
    # Compute the ranks of the top label probabilities for each of the valid pixels
    sorted_ce_prob_top_label = torch.zeros_like(ce_prob_top_label.flatten()).double()
    sorted_ce_prob_top_label[sorted_indices.cpu()] = torch.arange(len(ce_prob_top_label.flatten()), dtype=torch.double)

    # Normalize the values
    ce_cov = ((sorted_ce_prob_top_label - (unlbld_mask == 0).sum()) / (unlbld_mask != 0).sum()).view(height, width)
    ce_cov[ce_cov < 0] = 0

    # Print only errors
    ce_errors = (ce_cls != tgt.cpu()) & (tgt.cpu() != 255)

    # Ensure that the cmap is normalized
    ce_errors[0, 0, 0] = True
    ce_errors[0, 0, 1] = True
    ce_cov[0, 0] = 1
    ce_cov[0, 1] = 0

    # Do the same for the label-smoothing model
    ls_cls = ls_pred.softmax(0).max(0).indices.cpu()
    ls_prob_top_label = ls_pred.softmax(0).max(0).values.clone().cpu()
    ls_prob_top_label[unlbld_mask == 0] = -1
    sorted_indices = torch.argsort(ls_prob_top_label.flatten())
    sorted_ls_prob_top_label = torch.zeros_like(ls_prob_top_label.flatten()).double()
    sorted_ls_prob_top_label[sorted_indices] = torch.arange(len(ls_prob_top_label.flatten())).double()
    ls_cov = ((sorted_ls_prob_top_label - (unlbld_mask == 0).sum()) / (unlbld_mask != 0).sum()).view(height, width)
    ls_cov[ls_cov < 0] = 0

    ls_errors = (ls_cls != tgt.cpu()) & (tgt.cpu() != 255)
    ls_errors[0, 0, 0] = True
    ls_errors[0, 0, 1] = True
    ls_cov[0, 0] = 1
    ls_cov[0, 1] = 0

    maximum = max(torch.max(ce_cov.flatten()), torch.max(ls_cov.flatten()))

    ce_cov[0, 0] = maximum
    ls_cov[0, 0] = maximum

    # Compute the colors
    ce_most_conf_errors = torch.tensor(cbar(ce_cov.cpu())).squeeze(0).permute(2, 0, 1)[:3, ...] * 255
    ls_most_conf_errors = torch.tensor(cbar(ls_cov.cpu())).squeeze(0).permute(2, 0, 1)[:3, ...] * 255
    if full_plot:
        ce_most_conf_errors[(ce_errors == 0).expand(ce_most_conf_errors.shape)] *= 0.35
        ls_most_conf_errors[(ls_errors == 0).expand(ls_most_conf_errors.shape)] *= 0.35
    else:
        ce_most_conf_errors[(ce_errors == 0).expand(ce_most_conf_errors.shape)] = 60
        ls_most_conf_errors[(ls_errors == 0).expand(ls_most_conf_errors.shape)] = 60

    ce_most_conf_errors = ce_most_conf_errors.long()
    ls_most_conf_errors = ls_most_conf_errors.long()

    img = img.cpu().squeeze(0) * std.unsqueeze(-1).unsqueeze(-1) + mean.unsqueeze(-1).unsqueeze(-1)
    final_img = (
        torch.nn.functional.interpolate(img.clone().detach().unsqueeze(0) * 255, (height, width), mode="bilinear")
        .long()
        .squeeze(0)
    )
    final_tgt = (
        torch.nn.functional.interpolate(
            val_dst.decode_target(tgt.squeeze(0).cpu()).permute(2, 0, 1).unsqueeze(0).double(), (height, width), mode="nearest"
        )
        .squeeze(0)
        .long()
    )

    fig = plt.figure(figsize=(12, 6.5))
    fig.set_size_inches(12, 6.5)

    gs = mpl.gridspec.GridSpec(
        3, 3, wspace=0.05, hspace=0.32, width_ratios=[0.5, 0.5, 0.015], height_ratios=[0.5, 0.5, 0.5]
    )  # 2x2 grid
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[2, 0])
    ax5 = fig.add_subplot(gs[2, 1])

    ax_cbar = fig.add_subplot(gs[:, 2])  # full second row

    mpl.colorbar.ColorbarBase(ax_cbar, cmap=mpl.cm.viridis_r, orientation="vertical")

    ax0.imshow(final_img.permute(1, 2, 0)[min_height:max_height, min_width:max_width])
    ax0.set_title("Input Image", pad=8)
    ax0.axis("off")

    ax1.imshow(final_tgt.permute(1, 2, 0)[min_height:max_height, min_width:max_width])
    ax1.set_title("Ground Truth", pad=8)
    ax1.axis("off")

    ax2.imshow(val_dst.decode_target(ce_cls.squeeze(0).cpu()).squeeze(0)[min_height:max_height, min_width:max_width])
    ax2.set_title("Prediction - Cross-Entropy", pad=8)
    ax2.axis("off")

    ax3.imshow(val_dst.decode_target(ls_cls.squeeze(0).cpu()).squeeze(0)[min_height:max_height, min_width:max_width])
    ax3.set_title(rf"Prediction - Label-Smoothing ($\alpha={ls_value}$)", pad=8)
    ax3.axis("off")

    ax3.text(
        -1.1,
        0.5,
        "DeepLab-V3+ (ResNet-101) - Cityscapes",
        ha="center",
        va="center",
        transform=ax3.transAxes,
        rotation="vertical",
        size=12,
    )

    ax4.imshow(ce_most_conf_errors.permute(1, 2, 0)[min_height:max_height, min_width:max_width])
    ax4.set_title("Uncertainty Ranking (MSP) - Cross-Entropy", pad=8)
    ax4.axis("off")

    ax5.imshow(ls_most_conf_errors.permute(1, 2, 0)[min_height:max_height, min_width:max_width])
    ax5.set_title(rf"Uncertainty Ranking (MSP) - Label-Smoothing ($\alpha={ls_value}$)", pad=8)
    ax5.axis("off")

    # Add figure title
    fig.suptitle(
        "Lowest uncertainty-ranked (i.e. most confident) errors ✗ are yellow - Correct pixels ✓ are greyed out", x=0.52, y=0.08
    )
    if save:
        fig.savefig(f"figure_{img_id}.pdf", dpi=300, bbox_inches="tight")
    if plot:
        plt.show()
    else:
        plt.close()


def plot_segmentation_normalization(
    norm_dimension: int,
    ls_value: float,
    cbar,
    heights: dict,
    widths: dict,
    img_id: int,
    add_unit: float | str = "mean",
    norm_dimension_ce: int | None = None,
    save: bool = False,
    full_plot: bool = False,
    plot: bool = True,
):
    """Plot the segmentation figures with logit normalization.

    The arguments are similar to the previous function.
    """
    if norm_dimension_ce is None:
        norm_dimension_ce = norm_dimension
    height, min_height, max_height = heights["full"], heights["min"], heights["max"]
    width, min_width, max_width = widths["full"], widths["min"], widths["max"]

    norm_ce_score_top_label = norm_logits(ce_pred, p=norm_dimension_ce, add_unit=add_unit).max(0).values.clone().cpu()
    norm_ce_score_top_label[unlbld_mask == 0] = -1
    sorted_indices = torch.argsort(norm_ce_score_top_label.flatten())
    sorted_norm_ce_score_top_label = torch.zeros_like(norm_ce_score_top_label.flatten()).double()
    sorted_norm_ce_score_top_label[sorted_indices.cpu()] = torch.arange(
        len(norm_ce_score_top_label.flatten()), dtype=torch.double
    )
    norm_ce_cov = ((sorted_norm_ce_score_top_label - (unlbld_mask == 0).sum()) / (unlbld_mask != 0).sum()).view(height, width)
    norm_ce_cov[norm_ce_cov < 0] = 0
    norm_ce_cov[0, 0] = 1
    norm_ce_cov[0, 1] = 0

    norm_ls_score_top_label = norm_logits(ls_pred, p=norm_dimension, add_unit=add_unit).max(0).values.clone().cpu()
    norm_ls_score_top_label[unlbld_mask == 0] = -1
    sorted_indices = torch.argsort(norm_ls_score_top_label.flatten())
    sorted_norm_ls_score_top_label = torch.zeros_like(norm_ls_score_top_label.flatten()).double()
    sorted_norm_ls_score_top_label[sorted_indices] = torch.arange(len(norm_ls_score_top_label.flatten())).double()
    norm_ls_cov = ((sorted_norm_ls_score_top_label - (unlbld_mask == 0).sum()) / (unlbld_mask != 0).sum()).view(height, width)
    norm_ls_cov[norm_ls_cov < 0] = 0

    norm_ls_cov[0, 0] = 1
    norm_ls_cov[0, 1] = 0

    maximum = max(torch.max(norm_ce_cov.flatten()), torch.max(norm_ls_cov.flatten()))

    norm_ce_cov[0, 0] = maximum
    norm_ls_cov[0, 0] = maximum

    # Compute the colors
    norm_ce_most_conf_errors = torch.tensor(cbar(norm_ce_cov.cpu())).squeeze(0).permute(2, 0, 1)[:3, ...] * 255
    norm_ls_most_conf_errors = torch.tensor(cbar(norm_ls_cov.cpu())).squeeze(0).permute(2, 0, 1)[:3, ...] * 255
    if full_plot:
        norm_ce_most_conf_errors[(ce_errors == 0).expand(norm_ce_most_conf_errors.shape)] *= 0.35
        norm_ls_most_conf_errors[(ls_errors == 0).expand(norm_ls_most_conf_errors.shape)] *= 0.35
    else:
        norm_ce_most_conf_errors[(ce_errors == 0).expand(norm_ce_most_conf_errors.shape)] = 60
        norm_ls_most_conf_errors[(ls_errors == 0).expand(norm_ls_most_conf_errors.shape)] = 60
    norm_ce_most_conf_errors = norm_ce_most_conf_errors.long()
    norm_ls_most_conf_errors = norm_ls_most_conf_errors.long()

    fig = plt.figure(figsize=(12, 4.2))
    fig.set_size_inches(12, 4.2)

    gs = mpl.gridspec.GridSpec(
        2, 3, wspace=0.05, hspace=0.34, width_ratios=[0.5, 0.5, 0.015], height_ratios=[0.5, 0.5]
    )  # 2x2 grid
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    ax_cbar = fig.add_subplot(gs[:, 2])  # full second row

    mpl.colorbar.ColorbarBase(ax_cbar, cmap=mpl.cm.viridis_r, orientation="vertical")

    ax0.imshow(ce_most_conf_errors.permute(1, 2, 0)[min_height:max_height, min_width:max_width])
    ax0.set_title("Cross-Entropy - MSP", pad=8)
    ax0.axis("off")

    ax1.imshow(ls_most_conf_errors.permute(1, 2, 0)[min_height:max_height, min_width:max_width])
    ax1.set_title(rf"Label-Smoothing ($\alpha={ls_value}$) - MSP", pad=8)
    ax1.axis("off")

    ax2.imshow(norm_ce_most_conf_errors.permute(1, 2, 0)[min_height:max_height, min_width:max_width])
    ax2.set_title(rf"Cross-Entropy - Normalized logits $(p={norm_dimension_ce})$", pad=8)
    ax2.axis("off")
    ax2.text(
        -1.1,
        1.15,
        "DeepLab-V3+ (ResNet-101) - Cityscapes",
        ha="center",
        va="center",
        transform=ax3.transAxes,
        rotation="vertical",
        size=12,
    )

    ax3.imshow(norm_ls_most_conf_errors.permute(1, 2, 0)[min_height:max_height, min_width:max_width])
    ax3.set_title(rf"Label-Smoothing ($\alpha={ls_value}$) - Normalized logits $(p={norm_dimension})$", pad=8)
    ax3.axis("off")

    # Add figure title
    fig.suptitle(
        "Lowest uncertainty-ranked (i.e. most confident) errors ✗ are yellow - Correct pixels ✓ are greyed out", x=0.52, y=0.06
    )
    if save:
        fig.savefig(f"figure_comparison_{img_id}.pdf", dpi=300, bbox_inches="tight")
    if plot:
        plt.show()
    else:
        plt.close()
