import torch
from omegaconf import OmegaConf
from torch.profiler import ProfilerActivity, profile

from ..config import conf, device
from ..io.queued_dataset import QueuedDataLoader
from ..monitor import setup_writer
from ..utils.logger import logger
from .holder import model_holder


def profile_procedure() -> None:
    logger.warn(
        "Starting profiling with state\n" + OmegaConf.to_yaml(model_holder.state)
    )
    model_holder.writer = setup_writer()
    # Check if the training already has finished:

    # Initialize the training
    # switch model in training mode
    model_holder.model.train()
    model_holder.loader = QueuedDataLoader()
    model_holder.optim.zero_grad()

    batch = model_holder.loader.validation_batches[0].to(device)

    model_holder.optim.zero_grad()
    logger.warning("Starting profiling.")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,
        record_shapes=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(conf.path.tensorboard),
    ) as prof:
        _ = model_holder.model(batch)
    logger.warning("Profiling done.")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
    print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))
