import torch

from .logger import logger

def contains_nans(inp, string=""):
    if isinstance(inp, torch.Tensor):
        res = torch.any(torch.isnan(inp))
        return (res, string)
    elif hasattr(inp, "state_dict"):
        return contains_nans(inp.state_dict())
    elif hasattr(inp, "to_dict"):
        return contains_nans(inp.to_dict())
    elif hasattr(inp, "items"):
        for k, elem in inp.items():
            res,string = contains_nans(elem, str(k) + " " + string)
            if res:
                return (res, string)
        return (res, string)
    elif hasattr(inp, "__iter__"):
        for k, elem in enumerate(inp):
            res,string = contains_nans(elem, str(k) + " " + string)
            if res:
                return (res, string)
        return (res, string)
    elif isinstance(inp, (int, float)):
        return((False,string))
    elif inp is None:
        return((False,string))
    else:
        raise Exception


def check_chain_for_nans(chain):
    nan_detected = False
    # go backwards to the chain, if the model is fine, there
    # is no need to check anything else
    oldstr = ""
    for i, e in list(
        zip(
            range(len(chain)),
            chain,
        )
    )[::-1]:
        problem, element_name = contains_nans(e)
        if problem:
            nan_detected = True
            oldstr = element_name
            if i == 0:
                logger.error(
                    f"Nan in elem number {0} in the chain of type {type(chain[0])}"
                    f" {'concerning' if oldstr else ''} {oldstr}."
                )
        if not problem and nan_detected:
            logger.error(
                f"Nan in elem number {i+1} in the chain of type {type(chain[i+1])}"
                f" {'concerning' if oldstr else ''} {oldstr}."
            )
            raise ValueError
        else:
            break
