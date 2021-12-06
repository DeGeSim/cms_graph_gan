import numpy as np
import pandas as pd
import torch
from queueflow.batch_utils import move_batch_to_device
from tqdm import tqdm

from fgsim.config import conf, device
from fgsim.io.queued_dataset import QueuedDataLoader
from fgsim.ml.holder import Holder
from fgsim.monitoring.logger import logger
from fgsim.monitoring.monitor import setup_experiment


def prediction_procedure() -> None:
    holder: Holder = Holder()
    loader: QueuedDataLoader = QueuedDataLoader()
    experiment = setup_experiment(holder)

    if experiment.ended:
        logger.error("Training has not completed, stopping.")
        loader.qfseq.stop()
        exit(0)

    holder.select_best_model()
    holder.models.eval()
    # Initialize the training
    ys = []
    ypreds = []

    # Make sure the batches are loaded
    _ = loader.testing_batches
    loader.qfseq.stop()

    logger.info("Start iterating batches.")
    for _, batch in enumerate(tqdm(loader.testing_batches)):
        batch = move_batch_to_device(batch, device)
        with torch.no_grad():
            prediction = torch.squeeze(holder.models(batch).T)
            ypred = prediction.to("cpu").numpy()
            ytrue = batch.ytrue.to("cpu").numpy()

        ypreds.append(ypred)
        ys.append(ytrue)

    logger.info("Done with batches.")
    ys = np.hstack(ys)
    ypreds = np.hstack(ypreds)
    logger.info("Conversion done.")
    vars_dict = {
        "Energy": ys,
        "Prediction": ypreds,
        "Relativ Error": np.abs(1 - ypreds / ys),
    }
    df = pd.DataFrame(vars_dict)
    logger.info("Dataframe done.")
    df.to_csv(conf.path.predict_csv)
    logger.info(f"CVS written to {conf.path.predict_csv}.")
    # train_state.writer.add_scalars(
    #     "test", vars_dict, global_step=None, walltime=None
    # )
    # experiment.log_dataframe_profile(df,name='Test DF', log_raw_dataframe=True)

    # experiment.log_curve(
    #     "True vs Predicted Energy", list(ys), ypreds, overwrite=True, step=None
    # )
    # experiment.log_curve(
    #     "Relative Error", ys, df["Relativ Error"], overwrite=True, step=None
    # )
    exit(0)
