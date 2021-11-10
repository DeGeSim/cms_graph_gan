from pathlib import Path

import awkward as ak
import pandas as pd
import uproot

from fgsim.config import conf

pickle_lup_path = Path(conf.path.geo_lup).with_suffix(".pd")
if not pickle_lup_path.is_file():
    with uproot.open(conf.path.geo_lup) as rf:
        geo_lup = rf["analyzer/tree;1"].arrays(library="ak")
    geo_lup = ak.to_pandas(geo_lup)
    geo_lup.set_index("globalid", inplace=True)
    geo_lup.to_pickle(pickle_lup_path)


geo_lup = pd.read_pickle(pickle_lup_path)
