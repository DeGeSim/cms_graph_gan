from typing import Dict, Optional, Union

import numpy as np

from fgsim.config import conf

labels_dict: Dict[str, str] = {
    "w1p": "\\wop ($\\times 10^{3}$)",
    "w1m": "\\wom ($\\times 10^{3}$)",
    "w1efp": "\\woefp ($\\times 10^{5}$)",
    "fpnd": "FPND ($\\times 10^{5}$)",
    "fpd": "FPD ($\\times 10^{4}$)",
    "num_particles": "Cardinality",
    "speed": "Model Time/PC [ms]",
    "speed_gen": "Generator Time/PC [ms]",
    "speed_crit": "Critic Time/PC [ms]",
    "mean": "\\mavg",
    "etarel": "\\etarel",
    "phirel": "\\phirel",
    "ptrel": "\\ptrel",
    "mass": "\\mjet",
    "pt": "\\ptjet",
    "phi": "\\phijet",
    "eta": "\\etajet",
    "mass_raw": "\\mjetr",
    "pt_raw": "\\ptjetr",
    "phi_raw": "\\phijetr",
    "eta_raw": "\\etajetr",
    "ptsum": "$\\sum_i \\ptreli$",
    "response": "Response ($\\textstyle\\sum\\nolimits_i \\mathrm{E_i}/E$)",
    "showershape_peak_layer": "Peak Layer",
    "showershape_psr": (
        "$\\textstyle \\frac{|\\mathrm{Layer}^\\mathrm{Peak}-"
        "\\mathrm{Layer}^\\mathrm{Turnoff}|+1}{|\\mathrm{Layer}^\\mathrm{Peak}"
        "-\\mathrm{Layer}^\\mathrm{Turnon}|+1}$"
    ),
    "showershape_turnon_layer": "Turn-on Layer",
    "showershape_turnoff_layer": "Turn-off Layer",
    "sphereratio_ratio": (
        "$\\sum\\nolimits_i^{\\mathrm{Sphere}(0.3"
        " σ)}\\frac{\\mathrm{E_i}}{E_\\mathrm{Tot}} /"
        " \\sum\\nolimits_i^{\\mathrm{Sphere}(0.8"
        " σ)}\\frac{\\mathrm{E_i}}{E_\\mathrm{Tot}}$"
    ),
    "sphereratio_small": (
        "$\\sum\\nolimits_i^{\\mathrm{Sphere}(0.3 σ)}"
        " \\frac{\\mathrm{E_i}}{E_\\mathrm{Tot}}$"
    ),
    "sphereratio_large": (
        "$\\sum\\nolimits_i^{\\mathrm{Sphere}(0.8 σ)}"
        " \\frac{\\mathrm{E_i}}{E_\\mathrm{Tot}}$"
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
    "E": "Hit Energy [MeV]",
    "Eshower": "Shower Energy [MeV]",
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
        from caloutils import calorimeter as calo

        match v:
            case "E":
                # return 10 ** np.linspace(0, np.log10(5020), 100 + 1) - 0.5
                return np.linspace(0, 5020, 100 + 1) - 0.5
            case "z":
                return np.linspace(0, calo.num_z, calo.num_z + 1) - 0.5
            case "alpha":
                return np.linspace(0, calo.num_alpha, calo.num_alpha + 1) - 0.5
            case "r":
                return np.linspace(0, calo.num_r, calo.num_r + 1) - 0.5
            case _:
                pass
    elif conf.dataset_name == "jetnet":
        nbins = 50
        bin_d = {
            "num_particles": np.linspace(
                0, conf.loader.n_points, conf.loader.n_points, endpoint=True
            ),
            "ptrel": np.linspace(0, 1, nbins),
            "etarel": np.linspace(-0.4, 0.4, nbins),
            "phirel": np.linspace(-0.4, 0.4, nbins),
            "mass": np.linspace(0, 0.25, nbins),
        }
        if conf.loader.n_points == 30:
            bin_d |= {
                "eta": np.linspace(-0.045, 0.045, nbins),
                "phi": np.linspace(-0.045, 0.045, nbins),
                "pt": np.linspace(0.5, 1 + 1e-5, nbins),
            }
        if conf.loader.n_points == 150:
            bin_d |= {
                "pt": np.linspace(0.975, 1 + 1e-5, nbins, endpoint=True),
            }
            # bin_d["ptjet"] = np.linspace(0.9, 1, nbins, endpoint=True)
        if v in bin_d:
            return bin_d[v]
    return None
