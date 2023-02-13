from pathlib import Path

import awkward as ak
import pandas as pd
import uproot

from fgsim.config import conf

pickle_lup_path = Path(conf.path.geo_lup).with_suffix(".pd")
if not pickle_lup_path.is_file():
    with uproot.open(conf.path.geo_lup) as rf:
        geo_lup = rf["analyzer/tree;1"].arrays(library="ak")
    geo_lup = ak.to_dataframe(geo_lup)
    geo_lup.set_index("globalid", inplace=True)
    # Fix the layers being the same for all subdetector
    # assign negative values for z<0
    factor = 1 - 2 * (geo_lup.z < 0)
    # get the layerid within the subdetector
    layerid = geo_lup.layerid
    # calculate the needed shift
    detectors = sorted(geo_lup.detectorid.value_counts().index.values)
    # check that there is only the 8 detectors
    assert detectors == [8, 9, 10]
    # shift all layers from other subdetectors by the layers in the EE
    shift = layerid[geo_lup.detectorid == 8].max() * (geo_lup.detectorid != 8)
    geo_lup["layer"] = factor * (layerid + shift)
    geo_lup.to_pickle(pickle_lup_path)


geo_lup = pd.read_pickle(pickle_lup_path)
