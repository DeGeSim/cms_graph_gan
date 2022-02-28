from torch import nn


def dnn_gen(
    input_dim: int,
    output_dim: int,
    n_layers: int,
    activation_last_layer: bool = True,
):
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
    for ilayer, e in enumerate(layers):
        seq.append(e)
        if ilayer != n_layers - 1 or activation_last_layer:
            seq.append(nn.LeakyReLU(0.2))
    return nn.Sequential(*seq)
