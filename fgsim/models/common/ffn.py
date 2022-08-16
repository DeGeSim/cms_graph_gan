from typing import Optional

from torch import nn

from fgsim.config import conf


class FFN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        normalize: Optional[bool] = None,
        activation_last_layer: Optional[nn.Module] = None,
        n_layers: Optional[int] = None,
        n_nodes_per_layer: Optional[int] = None,
    ) -> None:
        if normalize is None:
            normalize = conf.ffn.normalize
        if activation_last_layer is None:
            activation_last_layer = nn.Identity()
        if n_layers is None:
            n_layers = conf.ffn.n_layers
        if n_nodes_per_layer is None:
            n_nodes_per_layer = max(
                conf.ffn.hidden_layer_size, input_dim, output_dim
            )
        super().__init__()
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
        activation = getattr(nn, conf.ffn.activation)(**conf.ffn.activation_params)
        for ilayer, e in enumerate(layers):
            seq.append(e)
            if ilayer != n_layers:
                seq.append(activation)
                if normalize:
                    seq.append(
                        nn.BatchNorm1d(
                            features[ilayer + 1],
                            affine=False,
                            track_running_stats=False,
                        )
                    )
            else:
                seq.append(activation_last_layer)
        self.seq = nn.Sequential(*seq)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.activation_last_layer = activation_last_layer
        self.n_nodes_per_layer = n_nodes_per_layer
        self.activation = activation
        self.reset_parameters()

    def forward(self, *args, **kwargs):
        return self.seq(*args, **kwargs)

    def __repr__(self):
        return (
            f"FFN({self.input_dim}->{self.output_dim},n_layers={self.n_layers},hidden_nodes={self.n_nodes_per_layer},activation={self.activation},"
            f" {self.activation_last_layer})"
        )

    def reset_parameters(self):
        self.seq.apply(self.init_weights)

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
