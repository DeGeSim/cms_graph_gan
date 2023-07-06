from typing import Dict

from matplotlib.figure import Figure

from fgsim.config import conf
from fgsim.plot.labels import var_to_label

from .ratioplot import ratioplot


def ftx_marginals(
    sim,
    gen,
    ftxname: str,
) -> Dict[str, Figure]:
    sim_features = {
        varname: arr
        for varname, arr in zip(
            conf.loader.x_features,
            sim[ftxname].reshape(-1, conf.loader.n_features).T.cpu().numpy(),
        )
    }
    gen_features = {
        varname: arr
        for varname, arr in zip(
            conf.loader.x_features,
            gen[ftxname].reshape(-1, conf.loader.n_features).T.cpu().numpy(),
        )
    }
    plots_d: Dict[str, Figure] = {}
    for ftn in conf.loader.x_features:
        plots_d[f"marginal_{ftn}.pdf"] = ratioplot(
            sim=sim_features[ftn], gen=gen_features[ftn], title=var_to_label(ftn)
        )

    return plots_d
