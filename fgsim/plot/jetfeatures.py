from typing import Dict

import jetnet
from matplotlib.figure import Figure

from fgsim.plot.infolut import var_to_label
from fgsim.utils.jetnetutils import to_stacked_mask

from .ratioplot import ratioplot


def jet_features(sim, gen, step=None) -> Dict[str, Figure]:
    sim_features_agr = jetnet.utils.jet_features(
        to_stacked_mask(sim).cpu().numpy()[..., :3]
    )
    gen_features_agr = jetnet.utils.jet_features(
        to_stacked_mask(gen).cpu().numpy()[..., :3]
    )

    plots_d = {}

    for ftn in ["pt", "eta", "mass"]:
        plots_d[f"jetfeatures_{ftn}.pdf"] = ratioplot(
            sim=sim_features_agr[ftn],
            gen=gen_features_agr[ftn],
            title=var_to_label(ftn),
        )

    return plots_d
