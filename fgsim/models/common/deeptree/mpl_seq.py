from typing import Optional

import torch

from . import DeepConv


class MPLSeq(torch.nn.Module):
    """
    1. The global features are concatenated with the node
       features and passed to the message generation layer.
    2. The messages are aggregated and passed to the update layer.
    3. The update layer returns the updated node features."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_mpl: int,
        n_hidden_nodes: int,
        **ancestor_conv_args,
    ):
        super().__init__()
        features = (
            [in_features] + [n_hidden_nodes for _ in range(n_mpl)] + [out_features]
        )
        self.mpls = torch.nn.ModuleList(
            [
                DeepConv(
                    in_features=features[n_ftx],
                    out_features=features[n_ftx + 1],
                    n_global=0,
                    **ancestor_conv_args,
                )
                for n_ftx in range(len(features) - 1)
            ]
        )

    def forward(
        self,
        *,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        global_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for conv in self.mpls:
            x = conv(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                batch=batch,
            )
        return x
