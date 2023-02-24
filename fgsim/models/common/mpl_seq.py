from typing import Optional

import torch
from torch_geometric.nn import EdgeConv, GINConv

from fgsim.models.common import FFN, GINCConv
from fgsim.models.common.deeptree import DeepConv


class MPLSeq(torch.nn.Module):
    """
    1. The global features are concatenated with the node
       features and passed to the message generation layer.
    2. The messages are aggregated and passed to the update layer.
    3. The update layer returns the updated node features."""

    def __init__(
        self,
        conv_name: str,
        in_features: int,
        out_features: int,
        n_cond: int,
        n_global: int,
        n_mpl: int,
        n_hidden_nodes: int,
        skip_connecton: bool,
        layer_param: dict,
    ):
        super().__init__()
        self.n_cond = n_cond
        self.n_global = n_global
        self.skip_connecton = skip_connecton
        self.in_features = in_features
        self.out_features = out_features
        self.n_hidden_nodes = n_hidden_nodes

        if n_mpl == 0:
            # assert in_features == out_features
            self.mpls = torch.nn.ModuleList([])
        else:
            self.features = (
                [in_features]
                + [n_hidden_nodes for _ in range(n_mpl - 1)]
                + [out_features]
            )
            self.mpls = torch.nn.ModuleList(
                [
                    self.wrap_layer_init(
                        conv_name,
                        in_features=self.features[n_ftx],
                        out_features=self.features[n_ftx + 1],
                        layer_param=layer_param,
                    )
                    for n_ftx in range(len(self.features) - 1)
                ]
            )

    def wrap_layer_init(self, conv_name, in_features, out_features, layer_param):
        if conv_name == "GINCConv":
            return GINCConv(
                FFN(
                    in_features + self.n_cond + self.n_global,
                    out_features,
                    **layer_param,
                    hidden_layer_size=self.n_hidden_nodes,
                )
            )
        if conv_name == "EdgeConv":
            return EdgeConv(
                FFN(
                    in_features * 2 + self.n_cond + self.n_global,
                    out_features,
                    **layer_param,
                    hidden_layer_size=self.n_hidden_nodes,
                )
            )
        elif conv_name == "GINConv":
            return GINConv(
                FFN(
                    in_features + self.n_cond + self.n_global,
                    out_features,
                    **layer_param,
                    hidden_layer_size=self.n_hidden_nodes,
                )
            )
        elif conv_name == "DeepConv":
            return DeepConv(
                in_features=in_features,
                out_features=out_features,
                n_cond=self.n_cond,
                n_global=self.n_global,
                **layer_param,
            )
        else:
            raise Exception

    def wrap_mpl(
        self, *, layer, x, cond, edge_index, edge_attr, batch, global_features
    ):
        if edge_attr is None:
            edge_attr = torch.ones_like(edge_index[0]).reshape(-1, 1)
        if isinstance(layer, GINCConv):
            return layer(
                x,
                torch.hstack(
                    (
                        cond[batch],
                        global_features[batch],
                    )
                ),
                edge_index,
            )
        elif isinstance(layer, GINConv):
            return layer(
                torch.hstack(
                    (
                        x,
                        cond[batch],
                        global_features[batch],
                    )
                ),
                edge_index,
            )
        elif isinstance(layer, EdgeConv):
            return layer(
                x,
                edge_index,
            )
        elif isinstance(layer, DeepConv):
            return layer(
                x=x,
                cond=cond,
                edge_index=edge_index,
                edge_attr=edge_attr,
                batch=batch,
                global_features=global_features,
            )
        else:
            raise Exception

    def forward(
        self,
        *,
        x: torch.Tensor,
        cond: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        global_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert x.shape[-1] == self.in_features
        if len(self.mpls) == 0:
            return x[..., : self.out_features]
        if self.skip_connecton:
            x_clone = x.clone()
        for conv in self.mpls:
            x = self.wrap_mpl(
                layer=conv,
                x=x,
                cond=cond,
                edge_index=edge_index,
                edge_attr=edge_attr,
                batch=batch,
                global_features=global_features,
            )

        assert x.shape[-1] == self.out_features
        if self.skip_connecton:
            x[..., : self.in_features] += x_clone[..., : self.out_features]
        return x
