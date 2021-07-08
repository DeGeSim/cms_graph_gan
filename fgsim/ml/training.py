import sys
import time
from copy import deepcopy

import torch
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..cometml import Experiment, comet_conf, get_experiment
from ..config import conf, device
from ..io.queued_dataset import QueuedDataLoader
from ..utils.check_for_nans import check_chain_for_nans
from ..utils.logger import logger
from .holder import model_holder


def setup_experiment_and_writer():
    # Create an experiment with your api key
    if hasattr(model_holder.state, "comet_experiment_key"):
        experiment = get_experiment(model_holder.state.comet_experiment_key)
    else:
        experiment = Experiment(**comet_conf)
        model_holder.state.comet_experiment_key = experiment.get_key()

    writer = SummaryWriter(conf.path.tensorboard)
    experiment.set_model_graph(str(model_holder.model))
    return (writer, experiment)


def writelogs():
    times = [
        model_holder.state.batch_start_time,
        model_holder.state.time_io_done,
        model_holder.state.time_training_done,
        model_holder.state.time_validation_done,
        model_holder.state.time_early_stopping_done,
    ]
    (iotime, traintime, valtime, esttime) = [
        times[itime] - times[itime - 1] for itime in range(1, len(times))
    ]
    timesD = {
        "iotime": iotime,
        "traintime": traintime,
        "valtime": valtime,
        "esttime": esttime,
    }

    model_holder.writer.add_scalars(
        "times",
        timesD,
        model_holder.state["grad_step"],
    )
    model_holder.writer.add_scalar(
        "loss", model_holder.loss, model_holder.state["grad_step"]
    )
    model_holder.writer.flush()

    model_holder.experiment.log_metrics(
        dic=timesD,
        prefix="times",
        step=model_holder.state.grad_step,
        epoch=model_holder.state.epoch,
    )
    model_holder.experiment.log_metric(
        "loss", model_holder.loss, model_holder.state["grad_step"]
    )


def training_step(batch):
    model_holder.optim.zero_grad()
    from torch.profiler import ProfilerActivity, profile, record_function

    logger.warning("Starting profiling.")

    if conf.profile:
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            profile_memory=True,
            record_shapes=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                conf.path.tensorboard
            ),
        ) as prof:
            output = model_holder.model(batch)
        logger.warning("Profiling done.")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
        print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))

        model_holder.loader.qfseq._stop()
        exit(0)
    else:
        output = model_holder.model(batch)

    prediction = torch.squeeze(output.T)
    model_holder.loss = model_holder.lossf(prediction, batch.y.float())
    model_holder.loss.backward()
    model_holder.optim.step()
    check_chain_for_nans((batch, prediction, model_holder.loss, model_holder.model))


def validate():
    if model_holder.state["grad_step"] % conf.training.validation_interval == 0:
        losses = []
        for batch in model_holder.loader.validation_batches:
            batch = batch.to(device)
            prediction = torch.squeeze(model_holder.model(batch).T)
            losses.append(model_holder.lossf(prediction, batch.y.float()))

        mean_loss = torch.mean(torch.tensor(losses))
        model_holder.state.val_losses.append(float(mean_loss))

        model_holder.writer.add_scalar(
            "val_loss", mean_loss, model_holder.state["grad_step"]
        )
        model_holder.experiment.log_metric(
            "val_loss", mean_loss, model_holder.state["grad_step"]
        )

        mean_loss = float(mean_loss)
        if (
            not hasattr(model_holder.state, "min_val_loss")
            or model_holder.state.min_val_loss > mean_loss
        ):
            model_holder.state.min_val_loss = mean_loss
            model_holder.best_grad_step = model_holder.state["grad_step"]
            model_holder.best_model_state = deepcopy(model_holder.model.state_dict())

    if (
        model_holder.state["grad_step"] != 0
        and model_holder.state["grad_step"] % conf.training.checkpoint_interval == 0
    ):
        model_holder.save_models()


def early_stopping():
    if (
        model_holder.state["grad_step"] != 0
        and model_holder.state["grad_step"] % conf.training.validation_interval == 0
    ):
        # the the most recent losses
        # dont stop for the first epochs
        if len(model_holder.state.val_losses) < conf.training.early_stopping:
            return
        recent_losses = model_holder.state.val_losses[-conf.training.early_stopping :]
        relative_improvement = 1 - (min(recent_losses) / recent_losses[0])

        if relative_improvement < conf.training.early_stopping_improvement:
            model_holder.save_models()
            model_holder.writer.flush()
            model_holder.writer.close()
            logger.warn("Early Stopping criteria fullfilled")
            if hasattr(model_holder, "loader"):
                model_holder.loader.qfseq.drain_seq()
            sys.exit()


# from comet_ml.experiment import BaseExperiment
# from .holder import model_holder
# @dataclass
# class TrainingState:
#     model_holder: ModelHolder
#     experiment: BaseExperiment


def training_procedure():
    logger.warn(
        "Starting training with state\n" + OmegaConf.to_yaml(model_holder.state)
    )
    # Check if the training already has finished:
    early_stopping()

    # Initialize the training
    # switch model in training mode
    model_holder.model.train()
    model_holder.loader = QueuedDataLoader()
    model_holder.writer, model_holder.experiment = setup_experiment_and_writer()
    try:
        # Iterate over the Epochs
        for model_holder.state.epoch in range(
            model_holder.state.epoch, conf.model["n_epochs"]
        ):
            # Iterate over the batches
            model_holder.state.batch_start_time = time.time()
            model_holder.loader.queue_epoch(
                n_skip_events=model_holder.state.processed_events
            ),
            for model_holder.state.ibatch, batch in enumerate(
                tqdm(
                    model_holder.loader,
                    initial=model_holder.state.ibatch,
                ),
                start=model_holder.state.ibatch,
            ):
                batch = batch.to(device)
                model_holder.state.time_io_done = time.time()

                training_step(batch)

                model_holder.state.time_training_done = time.time()

                # save the generated torch tensor models to disk
                validate()

                model_holder.state.time_validation_done = time.time()

                early_stopping()

                model_holder.state.time_early_stopping_done = time.time()

                writelogs()

                # preparatoin for next step
                model_holder.state.processed_events += conf.loader.batch_size
                model_holder.state["grad_step"] += 1
                model_holder.state.batch_start_time = time.time()

            model_holder.state.ibatch = 0
            model_holder.save_checkpoint()
            model_holder.save_best_model()
    except Exception as error:
        logger.error("Error detected, stopping qfseq.")
        model_holder.loader.qfseq._stop()
        raise error
