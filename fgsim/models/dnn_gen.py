from torch import nn


def dnn_gen(
    input_dim: int,
    output_dim: int,
    n_layers: int = 8,
    activation_last_layer=nn.Identity(),
):
    assert n_layers >= 4
    if (input_dim + output_dim) * n_layers > 300:
        inter_dim = max(
            int(300 / n_layers / (input_dim + output_dim)), input_dim, output_dim
        )
    else:
        inter_dim = max(30, input_dim + output_dim)
    # print(f"inter_dim {inter_dim} nodes {inter_dim*n_layers}")
    assert inter_dim * n_layers > 100, "Use at least 100 nodes"
    layers = (
        [nn.Linear(input_dim, inter_dim)]
        + [nn.Linear(inter_dim, inter_dim) for _ in range(n_layers - 2)]
        + [nn.Linear(inter_dim, output_dim)]
    )
    seq = []
    for ilayer, e in enumerate(layers):
        seq.append(e)
        if ilayer != n_layers - 1:
            seq.append(nn.LeakyReLU(0.2))
        else:
            seq.append(activation_last_layer)
    return nn.Sequential(*seq)
