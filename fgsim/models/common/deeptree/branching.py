from .branching_bppool import BranchingBPPool
from .branching_equivar import BranchingEquivar
from .branching_mat import BranchingMat
from .branching_noise import BranchingBase, BranchingNoise


def get_br_layer(mode, **kwargs) -> BranchingBase:
    match mode:
        case "equivar":
            return BranchingEquivar(**kwargs)
        case "mat":
            return BranchingMat(**kwargs)
        case "bppool":
            return BranchingBPPool(**kwargs)
        case "noise":
            return BranchingNoise(**kwargs)
        case _:
            raise Exception()
