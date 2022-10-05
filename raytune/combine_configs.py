# %%
from collections import Counter

import numpy as np
import pandas as pd
from ray.tune import ExperimentAnalysis

from fgsim.utils.oc_utils import dict_to_kv

analysis = ExperimentAnalysis(
    "~/fgsim/wd/ray/jetnet-deeptree4", default_metric="fpnd", default_mode="min"
)
df = analysis.results_df
# analysis.get_best_config()
df = df[~df["w1m"].isnull()]

configs_d = analysis.get_all_configs()
configs_d = {k.split("/")[-1]: dict_to_kv(v) for k, v in configs_d.items()}

trialid_to_hash = {
    v["trial_id"][0]: k.split("/")[-1]
    for k, v in analysis.fetch_trial_dataframes().items()
}


def replace_options(*parset):
    if isinstance(parset[0], dict):
        return {k: replace_options(*[e[k] for e in parset]) for k in parset[0]}
    elif isinstance(parset[0], list):
        return replace_options(*["/".join(sorted(e)) for e in parset])
    elif isinstance(parset[0], float):
        return np.mean(parset)
    else:
        ct = Counter(parset)
        mf, freq = ct.most_common(1)[0]
        return f"{mf}[{freq}/{len(parset)}]"


# %%
# def replace_options(*parset):
#     lead_conv_val = parset[0]
#     if isinstance(lead_conv_val, dict):
#         return {k: replace_options(*[e[k] for e in parset]) for k in lead_conv_val}
#     elif isinstance(lead_conv_val, tuple):
#         replace_options(
#             *list(filter(lambda tup: tup[0] == lead_conv_val, pe) for pe in parset)
#         )
#     elif all([isinstance(e,tuple) for e in lead_conv_val]):

#         return replace_options(*["/".join(sorted(e)) for e in parset])
#     elif isinstance(lead_conv_val, float):
#         return np.mean(parset)
#     else:
#         ct = Counter(parset)
#         mf, freq = ct.most_common(1)[0]
#         return f"{mf}[{freq}/{len(parset)}]"


def is_number(n):
    try:
        float(n)
    except ValueError:
        return False
    return True


best_config = {}
best_config["alltime"] = {
    k: v for k, v in dict_to_kv(analysis.get_best_config(scope="all"))
}
for nsamples in [1, 5, 10]:
    best_hashes = []
    for metric in ["fpnd", "w1m", "w1p"]:
        best_hashes += list(df[metric][df[metric].argsort()].index[:nsamples])
    configs_l = [configs_d[trialid_to_hash[h]] for h in best_hashes]

    # best_config = replace_options(*configs_l)
    best_config[nsamples] = {}
    for kidx, (k, _) in enumerate(configs_l[0]):
        val_l = []
        for cfg in configs_l:
            assert cfg[kidx][0] == k
            val_l.append(cfg[kidx][1])
        if all([is_number(e) for e in val_l]):
            best_config[nsamples][k] = np.mean([float(e) for e in val_l])
        else:
            ct = Counter(val_l)
            mf, freq = ct.most_common(1)[0]
            best_config[nsamples][k] = f"{mf}[{freq}/{len(val_l)}]"


pd.DataFrame(best_config).sort_index()

# %%
