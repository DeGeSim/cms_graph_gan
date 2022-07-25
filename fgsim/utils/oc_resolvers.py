from omegaconf import DictConfig, OmegaConf


# Add a custum resolver to OmegaConf allowing for divisions
# Give int back if you can:
def divide(numerator, denominator):
    if numerator // denominator == numerator / denominator:
        return numerator // denominator
    else:
        return numerator / denominator


def optionlist(options, ol):
    return DictConfig({item: options[item] for item in ol})


def merge(*configs):
    return OmegaConf.merge(*configs)


def register_resolvers():
    OmegaConf.register_new_resolver("div", divide, replace=True)
    OmegaConf.register_new_resolver("optionlist", optionlist, replace=True)
    OmegaConf.register_new_resolver("merge", merge, replace=True)
    OmegaConf.register_new_resolver("len", len, replace=True)
