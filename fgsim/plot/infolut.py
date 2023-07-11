from typing import Dict, Optional, Union

import numpy as np

from fgsim.config import conf

labels_dict: Dict[str, str] = {
    "etarel": "$\\eta_\\mathrm{rel}$",
    "phirel": "$\\phi_\\mathrm{rel}$",
    "ptrel": "$p^\\mathrm{T}_\\mathrm{rel}$",
    "mass": "$m_{rel}$",
    "phi": "$Σ ϕ_{rel}$",
    "pt": "$Σp_{T}^{rel}$",
    "eta": "$Ση_{rel}$",
}


def var_to_label(v: Union[str, int]) -> str:
    if isinstance(v, int):
        vname = conf.loader.x_features[v]
    else:
        vname = v
    if vname in labels_dict:
        return labels_dict[vname]
    else:
        return vname


def var_to_bins(v: Union[str, int]) -> Optional[np.ndarray]:
    if isinstance(v, int):
        vname = conf.loader.x_features[v]
    else:
        vname = v

    if (
        conf.dataset_name == "calochallange"
        and "calochallange2" in conf.loader.dataset_path
    ):
        return {
            "E": np.linspace(0, 6000, 100),  # E
            "z": np.linspace(0, 45, 45),  # z
            "alpha": np.linspace(0, 16, 16),  # alpha
            "r": np.linspace(0, 9, 9),  # r
        }[vname]
    return None
