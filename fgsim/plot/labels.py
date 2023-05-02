from typing import Dict, Union

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


def var_to_label(vname: Union[str, int]) -> str:
    if isinstance(vname, int):
        vname = conf.loader.x_features[vname]
    if vname in labels_dict:
        return labels_dict[vname]
    else:
        return vname
