import torch
from torch_geometric.data import Batch

from fgsim.config import conf
from fgsim.io.sel_loader import scaler
from fgsim.plot.eval_plots import eval_plots
from fgsim.plot.fig_logger import FigLogger
from fgsim.plot.modelgrads import fig_grads


def postprocess(batch: Batch) -> Batch:
    batch.x_scaled = batch.x.clone()
    batch.x = scaler.inverse_transform(batch.x, "x")
    if "y" in batch.keys:
        batch.y_scaled = batch.y.clone()
        batch.y = scaler.inverse_transform(batch.y, "y")

    if len({"kpd", "fgd"} & set(conf.training.val.metrics)):
        from fgsim.utils.jetnetutils import to_efp

        batch = to_efp(batch)

    if conf.dataset_name == "calochallange":
        from fgsim.loaders.calochallange.convcoord import batch_to_Exyz

        batch = batch_to_Exyz(batch)
    return batch


def eval_res_d(results_d, holder):
    step = holder.state.grad_step
    plot = step % conf.training.val.plot_interval == 0 or conf.command == "test"
    step = step if step != 0 else 1
    epoch = holder.state.epoch

    # evaluate the validation metrics
    with torch.no_grad():
        holder.val_metrics(**results_d)
    up_metrics_d, score = holder.val_metrics.get_metrics()

    holder.train_log.log_metrics(up_metrics_d, prefix="val", step=step, epoch=epoch)

    if conf.debug:
        return score

    if not plot:
        return score
    best_last_val = "test" if conf.command == "test" else "val"
    fig_logger = FigLogger(
        holder.train_log, plot_path=None, best_last_val=[best_last_val], step=step
    )

    eval_plots(fig_logger=fig_logger, res=results_d)

    if best_last_val != "val":
        return score

    # Plot the gradients and the weights
    fig_logger.best_last_val = ["val"]
    for lpart in holder.losses:
        if len(lpart.grad_aggr.steps):
            grad_fig = fig_grads(lpart.grad_aggr, lpart.name)
            fig_logger(grad_fig, f"grads/{lpart.name}")

    return score
