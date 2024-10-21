import torch
from caloutils import calorimeter


def Ezalphar_to_Exyz(Ezalphar: torch.Tensor) -> torch.Tensor:
    E, z, alpha, r = Ezalphar.T.double()
    # shift idx by one to go from (min,max)
    # (0, 1-1/num) -> (1/num,1)
    # z = (z + 1) / num_z
    # r = (r + 1) / num_r
    alpha = (alpha + 1) / calorimeter.num_alpha * torch.pi * 2
    y = r * torch.cos(alpha)
    x = r * torch.sin(alpha)
    #       r
    #       ^
    #     / |
    #    /  |
    #   /   |
    # / θ)  |
    # ------> z

    # theta = torch.arctan(r / z)
    # if (theta.abs() > torch.pi).any():
    #     raise RuntimeError("θ not in forward direction")
    # if (theta < 0).any():
    #     raise RuntimeError("θ not in forward direction")
    # batch.eta = -torch.log(torch.tan(theta / 2.0)).reshape(-1, 1)
    # batch.phi = alpha.reshape(-1, 1)
    return torch.stack([E, x, y, z]).T.float()


def Exyz_to_Ezalphar(Exyz: torch.Tensor) -> torch.Tensor:
    E, x, y, z = Exyz.T.double()

    r = torch.sqrt(x**2 + y**2)
    # alpha = torch.atan(x / y)
    alpha = torch.atan2(x, y)
    # alpha = torch.arccos(y / r)

    alpha = (alpha * calorimeter.num_alpha) / (torch.pi * 2) - 1

    return torch.stack([E, z, alpha, r]).T.float()
