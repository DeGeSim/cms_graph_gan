import torch
from torch import Tensor, nn

from fgsim.models.common.gapt.common import ISAB, PMA, SAB, LinearNet, _attn_mask
from fgsim.utils.jetnetutils import to_stacked_mask

# from https://github.com/rkansal47/MPGAN/tree/main/gapt


class ModelClass(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.model = GAPT_D(**kwargs)

    def forward(self, batch, condition):
        x = to_stacked_mask(batch)
        x[..., -1] = x[..., -1] - 0.5
        x = self.model(x, labels=condition)
        return x


class GAPT_D(nn.Module):
    def __init__(
        self,
        num_particles: int,
        input_feat_size: int,
        sab_layers: int = 2,
        num_heads: int = 4,
        embed_dim: int = 32,
        sab_fc_layers: list = [],
        layer_norm: bool = False,
        dropout_p: float = 0.0,
        final_fc_layers: list = [],
        use_mask: bool = True,
        use_isab: bool = False,
        num_isab_nodes: int = 10,
        linear_args: dict = {},
    ):
        super(GAPT_D, self).__init__()
        self.num_particles = num_particles
        self.input_feat_size = input_feat_size
        self.use_mask = use_mask

        self.sabs = nn.ModuleList()

        sab_args = {
            "embed_dim": embed_dim,
            "ff_layers": sab_fc_layers,
            "final_linear": False,
            "num_heads": num_heads,
            "layer_norm": layer_norm,
            "dropout_p": dropout_p,
            "linear_args": linear_args,
        }

        self.input_embedding = LinearNet(
            [], input_size=input_feat_size, output_size=embed_dim, **linear_args
        )

        # intermediate layers
        for _ in range(sab_layers):
            self.sabs.append(
                SAB(**sab_args)
                if not use_isab
                else ISAB(num_isab_nodes, **sab_args)
            )

        self.pma = PMA(
            num_seeds=1,
            **sab_args,
        )

        self.final_fc = LinearNet(
            final_fc_layers,
            input_size=embed_dim,
            output_size=1,
            final_linear=True,
            **linear_args,
        )

    def forward(self, x: Tensor, labels: Tensor = None):
        if self.use_mask:
            mask = x[..., -1:] + 0.5
            x = x[..., :-1]
        else:
            mask = None

        x = self.input_embedding(x)

        for sab in self.sabs:
            x = sab(x, _attn_mask(mask))

        return torch.sigmoid(self.final_fc(self.pma(x, _attn_mask(mask)).squeeze()))
