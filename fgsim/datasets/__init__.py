from ..config import conf

if conf.dataset_name == "jetnet":
    from .jetnet import Dataset, postprocess, scaler
