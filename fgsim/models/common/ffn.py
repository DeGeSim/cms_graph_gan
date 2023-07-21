import math
from typing import List, Optional

import torch
from torch import nn
from torch_geometric import nn as gnn

from fgsim.config import conf
from fgsim.models.common.benno import WeightNormalizedLinear


class FFN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        weight_init_method: Optional[str] = None,
        activation: Optional[str] = None,
        norm: Optional[str] = None,
        dropout: Optional[float] = None,
        n_layers: Optional[int] = None,
        final_linear: bool = False,
        bias: bool = False,
        hidden_layer_size: Optional[int] = None,
        equallr: Optional[bool] = None,
    ) -> None:
        if norm is None:
            self.norm = conf.ffn.norm
        else:
            self.norm = norm
        allnorms = {"snbn", "batchnorm", "spectral", "weight", "bwn", "graph"}
        assert self.norm in allnorms
        if dropout is None:
            dropout = conf.ffn.dropout
        if dropout == 0:
            dropout = None
        if n_layers is None:
            n_layers = conf.ffn.n_layers

        if hidden_layer_size is None:
            hidden_layer_size = max(
                conf.ffn.hidden_layer_size, input_dim, output_dim
            )
        if activation is None:
            activation = conf.ffn.activation
        if weight_init_method is None:
            weight_init_method = conf.ffn.weight_init_method

        if equallr is None:
            equallr = conf.ffn.equallr

        if equallr and self.norm in ["spectral", "weight", "bwn", "graph"]:
            raise RuntimeError()

        def activation_function():
            return getattr(nn, activation)(
                **conf.ffn.activation_params[conf.ffn.activation]
            )

        super().__init__()
        # +2 for input and output
        features: List[int] = (
            [input_dim]
            + [hidden_layer_size] * (n_layers - 1)
            + [
                output_dim,
            ]
        )
        # to keep the std of 1, the last layer should not see a reduction
        # in dimensionality, because otherwise it

        seqtmp = []
        for ilayer in range(n_layers):
            if self.norm == "bwn":
                m = WeightNormalizedLinear(
                    features[ilayer],
                    features[ilayer + 1],
                    bias=bias,
                    scale=True,
                    init_scale=0.5,
                    init_factor=0.5,
                )
            else:
                if equallr:
                    m = EqualLinear(
                        features[ilayer], features[ilayer + 1], bias=bias
                    )
                else:
                    m = nn.Linear(features[ilayer], features[ilayer + 1], bias=bias)
                if self.norm in ["spectral", "snbn"]:
                    m = nn.utils.parametrizations.spectral_norm(m)
                elif self.norm == "weight":
                    m = nn.utils.weight_norm(m)

            seqtmp.append((m, "x->x"))
            if ilayer == n_layers - 1 and final_linear:
                continue
            else:
                if dropout:
                    seqtmp.append((nn.Dropout(dropout), "x->x"))
                if self.norm in ["batchnorm", "snbn"]:
                    seqtmp.append(
                        (
                            nn.BatchNorm1d(
                                features[ilayer + 1],
                                affine=False,
                                track_running_stats=False,
                            ),
                            "x->x",
                        )
                    )
                elif self.norm == "graph":
                    seqtmp.append(
                        (GraphOrBatch(features[ilayer + 1]), "x, batch -> x")
                    )
                elif self.norm == "layernorm":
                    seqtmp.append((nn.LayerNorm(features[ilayer + 1]), "x->x"))
                elif self.norm in ("none", "spectral", "weight", "bwn"):
                    pass
                else:
                    raise Exception
                seqtmp.append((activation_function(), "x->x"))

        self.seq = gnn.Sequential("x, batch", seqtmp)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight_init_method = weight_init_method
        self.n_layers = n_layers
        self.hidden_layer_size = hidden_layer_size
        self.activation = activation
        self.bias = bias
        if not equallr:
            self.reset_parameters()

    def forward(self, x, batchidx=None):
        oldshape = x.shape
        dim = len(oldshape)
        if self.norm == "graph":
            if dim == 2:
                if batchidx is None:
                    raise Exception("batchidx is None despite 2dim tensor.")
                elif (batchidx.unique(return_counts=True)[1] < 2).any():
                    raise Exception()
            return self.seq(x.clone(), batchidx)
        if dim == 3:
            x = x.reshape(-1, oldshape[2])
            x = self.seq(x.clone(), None)
            x = x.reshape(oldshape[0], oldshape[1], self.output_dim)
            return x
        else:
            x = self.seq(x.clone(), None)
            return x

    def __repr__(self):
        return (
            f"FFN({self.input_dim}->{self.output_dim},n_layers={self.n_layers},"
            f"hidden_nodes={self.hidden_layer_size},activation={self.activation})"
        )

    def reset_parameters(self):
        self.seq.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            if self.weight_init_method == "default":
                m.reset_parameters()
            elif self.weight_init_method == "kaiming_uniform_":
                nn.init.kaiming_uniform_(
                    m.weight,
                    a=conf.ffn.activation_params["LeakyReLU"]["negative_slope"],
                    mode="fan_in",
                    nonlinearity="leaky_relu",
                )
                return
            elif self.weight_init_method == "xavier_uniform_":
                nonlinearity = {
                    "SELU": "selu",
                    "Sigmoid": "sigmoid",
                    "ReLU": "relu",
                    "LeakyReLU": "leaky_relu",
                    "Tanh": "tanh",
                    "GELU": "relu",
                }[self.activation]
                if nonlinearity == "leaky_relu":
                    getattr(nn.init, self.weight_init_method)(
                        m.weight,
                        gain=nn.init.calculate_gain(
                            nonlinearity,
                            conf.ffn.activation_params["LeakyReLU"][
                                "negative_slope"
                            ],
                        ),
                    )
                else:
                    getattr(nn.init, self.weight_init_method)(
                        m.weight, gain=nn.init.calculate_gain(nonlinearity)
                    )
            else:
                pass


class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + "_orig")
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * math.sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)  # 1
        del module._parameters[name]
        module.register_parameter(name + "_orig", nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)  # 2

        return fn

    def __call__(self, module, input):  # 3
        weight = self.compute_weight(module)  # 4
        setattr(module, self.name, weight)  # 5


def equal_lr(module, name="weight"):
    EqualLR.apply(module, name)

    return module


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim, bias=bias)
        linear.weight.data.normal_()
        if bias:
            linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, input):
        return self.linear(input)


class GraphOrBatch(nn.Module):
    def __init__(self, nftx):
        super().__init__()

        self.bn = nn.BatchNorm1d(nftx)
        self.inorm = nn.InstanceNorm1d(nftx)
        self.lnorm = nn.LayerNorm(nftx)
        self.gn = gnn.GraphNorm(nftx)

    def assert_mode(self, mode: bool):
        if not hasattr(self, "norm_mode"):
            self.norm_mode = mode
        if self.norm_mode != mode:
            raise Exception("switching mode not allowed")

    def forward(self, x, batchidx):
        xs = x.shape
        assert x.dim() in [2, 3]
        if x.dim() == 2:
            self.assert_mode(True)
            x = self.gn(x, batchidx)
            assert not (x == 0).all()
            return
        elif xs[1] < 3:
            self.assert_mode(False)
            x = x.reshape(-1, xs[-1])
            x = self.bn(x).reshape(*xs)
            assert not (x == 0).all()
            return x
        else:
            self.assert_mode(True)
            batchidx = torch.arange(xs[1], device=x.device).repeat_interleave(xs[1])
            x = x.reshape(-1, xs[-1])
            x = self.gn(x).reshape(*xs)
            assert not (x == 0).all()
            return x
