from typing import Dict, Optional

from matplotlib.figure import Figure

from fgsim.config import conf
from fgsim.plot.labels import var_to_label

from .ratioplot import ratioplot


def ftx_marginals(
    sim,
    gen,
    step: Optional[int] = None,
) -> Dict[str, Figure]:
    sim_features = {
        varname: arr
        for varname, arr in zip(
            conf.loader.x_features,
            sim.x.reshape(-1, conf.loader.n_features).T.cpu().numpy(),
        )
    }
    gen_features = {
        varname: arr
        for varname, arr in zip(
            conf.loader.x_features,
            gen.x.reshape(-1, conf.loader.n_features).T.cpu().numpy(),
        )
    }
    plots_d: Dict[str, Figure] = {}
    for ftn in conf.loader.x_features:
        plots_d[f"marginal_{ftn}.pdf"] = ratioplot(
            sim=sim_features[ftn],
            gen=gen_features[ftn],
            title=var_to_label(ftn),
            step=step,
        )

    return plots_d
