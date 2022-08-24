from collections import Counter
from pprint import pprint

import numpy as np
from ray.tune import ExperimentAnalysis

analysis = ExperimentAnalysis(
    "~/fgsim/wd/ray/jetnet-deeptree", default_metric="fpnd", default_mode="min"
)
df = analysis.results_df
df = df[~df["w1m"].isnull()]

configs_d = analysis.get_all_configs()
configs_d = {k.split("/")[-1]: v for k, v in configs_d.items()}

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

for nsamples in [1, 5, 10]:
    best_hashes = []
    for metric in ["fpnd", "w1m", "w1p"]:
        best_hashes += list(df[metric][df[metric].argsort()].index[:nsamples])
    configs_l = [configs_d[trialid_to_hash[h]] for h in best_hashes]

    best_config = replace_options(*configs_l)
    pprint(best_config)
