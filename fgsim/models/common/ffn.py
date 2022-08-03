from torch import nn

from fgsim.config import conf


class FFN(nn.Sequential):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation_last_layer=nn.Identity(),
        n_layers: int = conf.ffn.n_layers,
        n_nodes_per_layer: int = conf.ffn.hidden_layer_size,
        normalize=True,
    ) -> None:

        # +2 for input and output
        features = [
            input_dim,
            n_nodes_per_layer,
            n_nodes_per_layer,
            n_nodes_per_layer,
            output_dim,
        ]
        layers = [
            nn.Linear(features[ilayer], features[ilayer + 1])
            for ilayer in range(n_layers + 1)
        ]
        seq = []
        for ilayer, e in enumerate(layers):
            seq.append(e)
            if ilayer != n_layers:
                seq.append(
                    getattr(nn, conf.ffn.activation)(**conf.ffn.activation_params)
                )
                if normalize:
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