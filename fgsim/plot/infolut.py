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
    "response": "Response ($E/\\sum_i {E}_i$)",
    "showershape_peak_layer": "Peak Layer",
    "showershape_psr": (
        "$\\frac{|\\mathrm{Layer}^\\mathrm{Peak}-\\mathrm{Layer}^\\mathrm{Turnoff}|+1}"
        "{\\mathrm{Layer}^\\mathrm{Peak}-\\mathrm{Layer}^\\mathrm{Turnon}|+1}$"
    ),
    "showershape_turnon_layer": "Turnon Layer",
    "sphere_ratio": "$\\sum_i {E}_i$ Sphere(Δ0.3) / $\\sum_i {E}_i$ Sphere(Δ0.8)",
    "sphere_small": "$\\sum_i {E}_i$ Sphere(Δ0.3)",
    "sphere_large": "$\\sum_i {E}_i$ Sphere(Δ0.8)",
    "cyratio_ratio": (
        "$\\sum_i {E}_i$ Cylinder(Δ0.3)/ $\\sum_i {E}_i$ Cylinder(Δ0.8)"
    ),
    "cyratio_small": "$\\sum_i {E}_i$ Cylinder(Δ0.3)",
    "cyratio_large": "$\\sum_i {E}_i$ Cylinder(Δ0.8)",
    "fpc_x": "First PCA vector x",
    "fpc_y": "First PCA vector y",
    "fpc_z": "First PCA vector z",
    "fpc_eval": "First PCA Eigenvalue",
    "nhits_n": "Number of Hits",
    "nhits_n_by_E": "Number of Hits / Shower Energy",
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
        from caloutils import calorimeter

        return {
            "E": np.linspace(0, 6000, 100 + 1) - 0.5,
            "z": np.linspace(0, calorimeter.num_z, calorimeter.num_z + 1) - 0.5,
            "alpha": (
                np.linspace(0, calorimeter.num_alpha, calorimeter.num_alpha + 1)
                - 0.5
            ),
            "r": np.linspace(0, calorimeter.num_r, calorimeter.num_r + 1) - 0.5,
        }[vname]
    return None
