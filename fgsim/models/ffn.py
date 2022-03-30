from torch import nn

from fgsim.config import conf


class FFN(nn.Sequential):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_layers: int = 8,
        activation_last_layer=nn.Identity(),
    ) -> None:
        assert n_layers >= 4
        if (input_dim + output_dim) * n_layers > 300:
            inter_dim = max(
                int(300 / n_layers / (input_dim + output_dim)),
                input_dim,
                output_dim,
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
                seq.append(
                    getattr(nn, conf.ffn.activation)(**conf.ffn.activation_params)
                )
            else:
                seq.append(activation_last_layer)
        super(FFN, self).__init__(*seq)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.activation_last_layer = activation_last_layer

        self.reset_parameters()

    def reset_parameters(self):
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nonlinearity = {
                "SELU": "selu",
                "Sigmoid": "sigmoid",
                "ReLU": "relu",
                "LeakyReLU": "leaky_relu",
            }[conf.ffn.activation]

            # nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="linear")
            nn.init.xavier_uniform_(
                m.weight, gain=nn.init.calculate_gain(nonlinearity)
            )
            m.bias.data.fill_(conf.ffn.init_weights_bias_const)


nn.ReLU
