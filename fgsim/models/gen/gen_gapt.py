import torch
from torch import Tensor, nn
from torch_geometric.data import Batch, Data

from fgsim.config import conf
from fgsim.models.common.gapt.common import ISAB, SAB, LinearNet, _attn_mask


class ModelClass(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.num_particles = kwargs["num_particles"]
        self.output_feat_size = kwargs["output_feat_size"]
        self.embed_dim = kwargs["embed_dim"]

        self.model = GAPT_G(**kwargs)
        self.z_shape = (
            conf.loader.batch_size,
            conf.loader.n_points,
            self.embed_dim,
        )

    def forward(self, x: Tensor, cond: Tensor) -> Batch:
        x = self.model(x, cond)
        x, mask = x[..., : self.output_feat_size], x[..., -1]

        batch = Batch.from_data_list(
            [Data(x=xe[me > 0]) for xe, me in zip(x, mask)]
        )
        return batch


class GAPT_G(nn.Module):
    def __init__(
        self,
        num_particles: int,
        output_feat_size: int,
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
        super(GAPT_G, self).__init__()
        self.num_particles = num_particles
        self.output_feat_size = output_feat_size
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

        # intermediate layers
        for _ in range(sab_layers):
            self.sabs.append(
                SAB(**sab_args)
                if not use_isab
                else ISAB(num_isab_nodes, **sab_args)
            )

        self.final_fc = LinearNet(
            final_fc_layers,
            input_size=embed_dim,
            output_size=output_feat_size,
            final_linear=True,
            **linear_args,
        )

    def forward(self, x: Tensor, labels: Tensor = None):
        if self.use_mask:
            # unnormalize the last jet label - the normalized # of particles per jet
            # (between 1/``num_particles`` and 1) - to between 0 and ``num_particles`` - 1
            # *** MODIFIED ***
            # num_jet_particles = (labels[:, -1] * self.num_particles).int() - 1
            num_jet_particles = labels[:, -1].int() - 1
            assert (num_jet_particles > 0).all()
            # *** MODIFIED END ***
            # sort the particles bythe first noise feature per particle, and the first
            # ``num_jet_particles`` particles receive a 1-mask, the rest 0.
            mask = (
                (x[:, :, 0].argsort(1).argsort(1) <= num_jet_particles.unsqueeze(1))
                .unsqueeze(2)
                .float()
            )
            # logging.debug(
            #     f"x \n {x[:2, :, 0]} \n num particles \n {num_jet_particles[:2]} \n"
            #     f" gen mask \n {mask[:2]}"
            # )
        else:
            mask = None

        for sab in self.sabs:
            x = sab(x, _attn_mask(mask))

        # *** MODIFIED ***
        # x = torch.tanh(self.final_fc(x))
        x = self.final_fc(x)
        # *** MODIFIED END ***

        return torch.cat((x, mask - 0.5), dim=2) if mask is not None else x
