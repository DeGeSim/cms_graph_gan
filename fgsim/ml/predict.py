import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ..config import conf, device
from ..io.queued_dataset import QueuedDataLoader
from ..monitor import setup_experiment, setup_writer
from ..utils.logger import logger
from .holder import model_holder
from .train_state import TrainState


def prediction_procedure():
    train_state = TrainState(
        model_holder,
        model_holder.state,
        QueuedDataLoader(),
        setup_writer(),
        setup_experiment(model_holder),
    )

    train_state.holder.select_best_model()
    train_state.holder.model.eval()
    # Initialize the training
    ys = []
    yhats = []

    train_state.loader.load_test_batches()
    logger.info("Start iterating batches.")
    for ibatch, batch in enumerate(tqdm(train_state.loader.testing_batches)):
        batch = batch.to(device)
        prediction = torch.squeeze(train_state.holder.model(batch).T)

        yhat = prediction.detach().to("cpu").numpy()
        yhats.append(yhat)

        y = batch.y.detach().to("cpu").numpy()
        ys.append(y)
        if ibatch > 3:
            break

    logger.info("Done with batches.")
    ys = np.hstack(ys)
    yhats = np.hstack(yhats)
    logger.info("Conversion done.")
    vars_dict = {
        "Energy": ys,
        "Prediction": yhats,
        "Relativ Error": np.abs(1 - yhats / ys),
    }
    df = pd.DataFrame(vars_dict)
    logger.info("Dataframe done.")
    df.to_csv(conf.path.predict_csv)
    logger.info(f"CVS written to {conf.path.predict_csv}.")
    # train_state.writer.add_scalars(
    #     "test", vars_dict, global_step=None, walltime=None
    # )
    # train_state.experiment.log_dataframe_profile(df,name='Test DF', log_raw_dataframe=True)

    # train_state.experiment.log_curve(
    #     "True vs Predicted Energy", list(ys), yhats, overwrite=True, step=None
    # )
    # train_state.experiment.log_curve(
    #     "Relative Error", ys, df["Relativ Error"], overwrite=True, step=None
    # )
