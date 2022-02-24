from torch import nn


def dnn_gen(input_dim: int, output_dim: int, n_layers: int):
    if n_layers == 1:
        layers = [nn.Linear(input_dim, output_dim)]
    elif n_layers == 2:
        layers = [nn.Linear(input_dim, input_dim), nn.Linear(input_dim, output_dim)]
    else:
        layers = [
            nn.Linear(input_dim, input_dim),
            nn.Linear(input_dim, output_dim),
        ] + [nn.Linear(output_dim, output_dim) for _ in range(n_layers - 1)]
    seq = []
    for e in layers:
        seq.append(e)
        seq.append(nn.LeakyReLU(0.2))
    return nn.Sequential(*seq)
