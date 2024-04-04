from typing import Dict

from matplotlib.figure import Figure

from fgsim.config import conf

from .ratioplot import ratioplot


def hist1d(
    sim, gen, ftxname: str, bins=None, energy_weighted=False
) -> Dict[str, Figure]:
    epos = conf.loader.x_ftx_energy_pos
    ename = conf.loader.x_features[epos]

    sim_features = _exftxt(sim[ftxname])
    gen_features = _exftxt(gen[ftxname])
    fext = "_Ew" if energy_weighted else ""

    plots_d: Dict[str, Figure] = {}
    for iftx, ftn in enumerate(conf.loader.x_features):
        if energy_weighted and iftx == epos:
            continue
        b = bins[iftx] if bins is not None else None
        simw = sim_features[ename] if energy_weighted else None
        genw = gen_features[ename] if energy_weighted else None
        plots_d[f"marginal_{ftn}{fext}.pdf"] = ratioplot(
            [sim_features[ftn], gen_features[ftn]],
            ftn=ftn,
            bins=b,
            weights=[simw, genw],
        )

    return plots_d


def _exftxt(arr):
    return {
        varname: arr
        for varname, arr in zip(
            conf.loader.x_features,
            arr.reshape(-1, conf.loader.n_features).T.cpu().numpy(),
        )
    }
