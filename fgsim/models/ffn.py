from math import ceil

from torch import nn

from fgsim.config import conf


class FFN(nn.Sequential):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_layers: int = 4,
        activation_last_layer=nn.Identity(),
    ) -> None:
        assert n_layers >= 4
        # +2 for input and output
        features = [
            ceil(
                (1 - ilayer / (n_layers)) * input_dim
                + (ilayer / (n_layers) * output_dim)
            )
            for ilayer in range(n_layers + 1)
        ]
        layers = [
            nn.Linear(features[ilayer], features[ilayer + 1])
            for ilayer in range(n_layers)
        ]
        seq = []
        for ilayer, e in enumerate(layers):
            seq.append(e)
            if ilayer != n_layers - 1:
                seq.append(
                    getattr(nn, conf.ffn.activation)(**conf.ffn.activation_params)
                )
                seq.append(nn.BatchNorm1d(features[ilayer + 1]))
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
