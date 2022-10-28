from typing import List, Optional

from torch import nn

from fgsim.config import conf


class FFN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        norm: Optional[str] = None,
        dropout: Optional[bool] = None,
        n_layers: Optional[int] = None,
        final_linear: Optional[bool] = False,
        n_nodes_per_layer: Optional[int] = None,
    ) -> None:
        if norm is None:
            norm = conf.ffn.norm
        if dropout is None:
            dropout = conf.ffn.dropout
        if n_layers is None:
            n_layers = conf.ffn.n_layers
        if n_nodes_per_layer is None:
            n_nodes_per_layer = max(
                conf.ffn.hidden_layer_size, input_dim, output_dim
            )
        super().__init__()
        # +2 for input and output
        features: List[int] = (
            [input_dim]
            + [n_nodes_per_layer] * (n_layers - 1)
            + [
                output_dim,
            ]
        )
        self.seq = nn.Sequential()
        activation = getattr(nn, conf.ffn.activation)(
            **conf.ffn.activation_params[conf.ffn.activation]
        )
        for ilayer in range(n_layers):
            self.seq.append(
                nn.Linear(features[ilayer], features[ilayer + 1], bias=False)
            )
            if ilayer != n_layers - 1:
                self.seq.append(activation)
                if dropout:
                    self.seq.append(nn.Dropout(0.2))
                if norm == "batchnorm":
                    self.seq.append(
                        nn.BatchNorm1d(
                            features[ilayer + 1],
                            affine=False,
                            track_running_stats=False,
                        )
                    )
                elif norm == "layernorm":
                    self.seq.append(nn.LayerNorm(features[ilayer + 1]))
                elif norm == "none":
                    pass
                else:
                    raise Exception
            else:
                if not final_linear:
                    self.seq.append(activation)
                    if norm == "batchnorm":
                        self.seq.append(
                            nn.BatchNorm1d(
                                features[ilayer + 1],
                                # affine=False,
                                # track_running_stats=False,
                            )
                        )
                    elif norm == "layernorm":
                        self.seq.append(nn.LayerNorm(features[ilayer + 1]))
                    elif norm == "none":
                        pass
                    else:
                        raise Exception

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.n_nodes_per_layer = n_nodes_per_layer
        self.activation = activation
        # if conf.ffn.init_weights != "kaiming_uniform_":
        #     self.reset_parameters()

    def forward(self, x):
        return self.seq(x)

    def __repr__(self):
        return f"FFN({self.input_dim}->{self.output_dim},n_layers={self.n_layers},hidden_nodes={self.n_nodes_per_layer},activation={self.activation})"

    def reset_parameters(self):
        self.seq.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nonlinearity = {
                "SELU": "selu",
                "Sigmoid": "sigmoid",
                "ReLU": "relu",
                "LeakyReLU": "leaky_relu",
                "Tanh": "tanh",
            }[conf.ffn.activation]

            # nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="linear")
            getattr(nn.init, conf.ffn.init_weights)(
                m.weight, gain=nn.init.calculate_gain(nonlinearity)
            )
            # m.bias.data.fill_(conf.ffn.init_weights_bias_const)
