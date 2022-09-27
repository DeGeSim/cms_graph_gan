from typing import Dict, Optional

from matplotlib.figure import Figure

from fgsim.config import conf

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
        plots_d[f"ftxmarginal_{ftn}.pdf"] = ratioplot(
            sim_arr=sim_features[ftn],
            gen_arr=gen_features[ftn],
            title=ftn,
            step=step,
        )

    return plots_d
