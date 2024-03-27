from typing import Dict, Optional, Union

import numpy as np

from fgsim.config import conf

labels_dict: Dict[str, str] = {
    "w1p": "\\wop ($\\times 10^{3}$)",
    "w1m": "\\wom ($\\times 10^{3}$)",
    "w1efp": "\\woefp ($\\times 10^{5}$)",
    "fpnd": "FPND ($\\times 10^{5}$)",
    "fpd": "FPD ($\\times 10^{4}$)",
    "num_particles": "Number of Constituents",
    "speed": "Model Time/Event [ms]",
    "speed_gen": "Generator Time/Event [ms]",
    "speed_crit": "Critic Time/Event [ms]",
    "mean": "\\mavg",
    "etarel": "\\etarel",
    "phirel": "\\phirel",
    "ptrel": "\\ptrel",
    "mass": "\\mjet",
    "phi": "\\phijet",
    "pt": "\\ptjet",
    "eta": "\\etajet",
    "response": "Response ($\\textstyle\\sum\\nolimits_i \\mathrm{E_i}/E$)",
    "showershape_peak_layer": "Peak Layer",
    "showershape_psr": (
        "$\\textstyle \\frac{|\\mathrm{Layer}^\\mathrm{Peak}-"
        "\\mathrm{Layer}^\\mathrm{Turnoff}|+1}{|\\mathrm{Layer}^\\mathrm{Peak}"
        "-\\mathrm{Layer}^\\mathrm{Turnon}|+1}$"
    ),
    "showershape_turnon_layer": "Turnon Layer",
    "sphereratio_ratio": (
        "$\\sum\\nolimits_i^{\\mathrm{Sphere}(0.3 σ)}\\mathrm{E_i} /"
        " \\sum\\nolimits_i^{\\mathrm{Sphere}(0.8 σ)}\\mathrm{E_i}$"
    ),
    "sphereratio_small": (
        "$\\sum\\nolimits_i^{\\mathrm{Sphere}(0.3 σ)} \\mathrm{E_i}$"
    ),
    "sphereratio_large": (
        "$\\sum\\nolimits_i^{\\mathrm{Sphere}(0.8 σ)} \\mathrm{E_i}$"
    ),
    "cyratio_ratio": (
        "$\\frac{\\sum\\nolimits_i^{\\mathrm{Cylinder}(0.2 σ)}"
        " \\mathrm{E_i}}{\\sum\\nolimits_i^{\\mathrm{Cylinder}(0.6 σ)}"
        "\\mathrm{E_i}}$"
    ),
    "cyratio_small": (
        "$\\sum\\nolimits_i^{\\mathrm{Cylinder}(0.2 σ)} \\mathrm{E_i}$"
    ),
    "cyratio_large": (
        "$\\sum\\nolimits_i^{\\mathrm{Cylinder}(0.6 σ)} \\mathrm{E_i}$"
    ),
    "fpc_x": "First PCA vector x",
    "fpc_y": "First PCA vector y",
    "fpc_z": "First PCA vector z",
    "fpc_eval": "First PCA Eigenvalue",
    "nhits_n": "Number of Hits",
    "nhits_n_by_E": "Number of Hits / Shower Energy",
    "E": "Hit Energy [GeV]",
    "alpha": "$\\alpha$",
    "alpha_Ew": "$\\alpha$ weighted",
    "r": "$r$",
    "r_Ew": "$r$ weighted",
    "z": "$z$",
    "z_Ew": "$z$ weighted",
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
        v = conf.loader.x_features[v]

    if (
        conf.dataset_name == "calochallange"
        and "calochallange2" in conf.loader.dataset_path
    ):
        from caloutils import calorimeter

        bin_d = {
            "E": np.linspace(0, 6000, 100 + 1) - 0.5,
            "z": np.linspace(0, calorimeter.num_z, calorimeter.num_z + 1) - 0.5,
            "alpha": (
                np.linspace(0, calorimeter.num_alpha, calorimeter.num_alpha + 1)
                - 0.5
            ),
            "r": np.linspace(0, calorimeter.num_r, calorimeter.num_r + 1) - 0.5,
        }
        if v in bin_d:
            return bin_d[v]
    elif conf.dataset_name == "jetnet":
        nbins = 50
        bin_d = {
            "num_particles": np.linspace(
                0, conf.loader.n_points, conf.loader.n_points, endpoint=True
            ),
            "ptrel": np.linspace(0, 1, nbins),
            "etarel": np.linspace(-0.4, 0.4, nbins),
            "phirel": np.linspace(-0.4, 0.4, nbins),
            "ptjet": np.linspace(0.5, 1 + 1e-5, nbins),
            "etajet": np.linspace(-0.045, 0.045, nbins),
            "phijet": np.linspace(-0.045, 0.045, nbins),
            "mjet": np.linspace(0, 0.25, nbins),
        }
        if conf.loader.n_points == 150:
            bin_d["ptjet"] = np.linspace(0.9, 1 + 1e-10, nbins, endpoint=True)
            # bin_d["ptjet"] = np.linspace(0.9, 1, nbins, endpoint=True)
        if v in bin_d:
            return bin_d[v]
    return None
