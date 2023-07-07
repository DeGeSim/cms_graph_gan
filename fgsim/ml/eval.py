import torch
from torch_geometric.data import Batch

from fgsim.config import conf, device
from fgsim.io.sel_loader import scaler
from fgsim.ml.holder import Holder
from fgsim.plot.eval_plots import eval_plots
from fgsim.plot.fig_logger import FigLogger
from fgsim.plot.modelgrads import fig_grads


def gen_res_from_sim_batches(batches: list[Batch], holder: Holder):
    res_d_l = {
        "sim_batch": [],
        "gen_batch": [],
        "sim_crit": [],
        "gen_crit": [],
    }
    for batch in batches:
        batch = batch.to(device)
        for k, val in holder.pass_batch_through_model(batch, eval=True).items():
            if k in ["sim_batch", "gen_batch"]:
                for e in val.to_data_list():
                    res_d_l[k].append(e.detach().cpu())
            elif k in ["sim_crit", "gen_crit"]:
                res_d_l[k].append(val.detach().cpu())
    sim_crit = torch.vstack(res_d_l["sim_crit"])
    gen_crit = torch.vstack(res_d_l["gen_crit"])

    sim_batch = Batch.from_data_list(res_d_l["sim_batch"])
    gen_batch = Batch.from_data_list(res_d_l["gen_batch"])

    assert sim_batch.x.shape == gen_batch.x.shape

    results_d = {
        "sim_batch": sim_batch,
        "gen_batch": gen_batch,
        "gen_crit": gen_crit,
        "sim_crit": sim_crit,
    }
    gen_batch.y = sim_batch.y.clone()
    for k in ["sim_batch", "gen_batch"]:
        results_d[k] = postprocess(results_d[k])
    return results_d


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
        from fgsim.loaders.calochallange import postprocess

        batch = postprocess(batch)
    return batch


def eval_res_d(
    results_d: dict, holder: Holder, step: int, epoch: int, plot_path=None
):
    plot = step % conf.training.val.plot_interval == 0 or conf.command == "test"
    step = step if step != 0 else 1

    # evaluate the validation metrics
    with torch.no_grad():
        holder.eval_metrics(**results_d)
    up_metrics_d, score = holder.eval_metrics.get_metrics()

    holder.train_log.log_metrics(up_metrics_d, prefix="val", step=step, epoch=epoch)

    if conf.debug:
        return score

    if not plot:
        return score
    best_last_val = "test" if conf.command == "test" else "val"

    fig_logger = FigLogger(
        holder.train_log,
        plot_path=plot_path,
        best_last_val=[best_last_val],
        step=step,
        epoch=epoch,
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
