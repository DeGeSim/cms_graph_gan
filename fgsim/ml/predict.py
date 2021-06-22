import pandas as pd
import torch

from ..config import conf, device
from ..io.queued_dataset import get_loader
from .holder import model_holder as holder


def prediction_procedure():
    holder.validation_batches, holder.qfseq = get_loader()
    holder.qfseq.stop()

    # Initialize the training
    ys = []
    yhats = []

    holder.model.load_state_dict(holder.best_model_state)
    holder.model.eval()

    for batch in holder.validation_batches:
        batch = batch.to(device)
        prediction = torch.squeeze(holder.model(batch).T)
        ys.append(batch.y.float())
        yhats.append(prediction)
    ys = torch.hstack(ys).detach().to("cpu").numpy()
    yhats = torch.hstack(yhats).detach().to("cpu").numpy()
    df = pd.DataFrame({"Energy ": ys, "Prediction": yhats})
    df.to_csv(conf.path.predict_csv)
