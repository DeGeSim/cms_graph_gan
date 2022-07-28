import torch
from torch import Tensor, nn
from torch_geometric.data import Batch, Data

from fgsim.config import conf
from fgsim.models.common.mpgan import LinearNet, MPNet


class ModelClass(MPNet):
    """
    Message passing generator.
    Goes through an optional latent fully connected layer then ``mp_iters`` iterations of message
    passing to output a tensor of shape ``[batch_size, num_particles, output_node_size]``.

    A number of options for masking are implemented, as described in the appendix of
    Kansal et. al. *Particle Cloud Generation with Message Passing Generative Adversarial Networks*
    (https://arxiv.org/abs/2106.11535).
    Args for masking are described in the masking functions below.

    Input ``x`` tensor to the forward pass must be of shape ``[batch_size, lfc_latent_size]`` if
    using ``lfc`` else ``[batch_size, num_particles, input_node_size]``.

    Args:
        lfc (bool): use a fully connected network to go from a vector latent space to a graph
          structure of ``num_particles`` nodes with ``node_input_size`` features. Defaults to False.
        lfc_latent_size (int): if using ``lfc``, size of the vector latent space. Defaults to 128.
        **mpnet_args: args for ``MPNet`` base class.
    """

    def __init__(self, lfc: bool = False, lfc_latent_size: int = 128, **mpnet_args):
        super().__init__(**mpnet_args)

        # latent fully connected layer
        self.lfc = lfc
        if lfc:
            self.lfc_layer = nn.Linear(
                lfc_latent_size, self.num_particles * self.input_node_size
            )
        # Edit start
        self.z_shape = (
            conf.loader.batch_size,
            conf.loader.n_points,
            mpnet_args["input_node_size"],
        )
        # Edit end

    # Edit start
    def forward(self, x: Tensor) -> Batch:
        x = super().forward(x)
        x = x[..., : conf.loader.n_features]
        return Batch.from_data_list([Data(x=e) for e in x])

    # Edit end

    def _pre_mp(self, x, labels):
        """Pre-message-passing operations"""
        if self.lfc:
            x = self.lfc_layer(x).reshape(
                x.shape[0], self.num_particles, self.input_node_size
            )

        return x

    def _init_mask(
        self,
        mask_learn: bool = False,
        mask_learn_sep: bool = False,
        fmg: list = [64],
        **mask_args,
    ):
        """
        Intialize potential mask networks and variables.

        Args:
            mask_learn (bool): learning a mask per particle using each particle's initial noise.
              Defaults to False.
            mask_learn_sep (bool): predicting an overall number of particles per jet using separate
              jet noise. Defaults to False.
            fmg (list): list of mask network intermediate layer sizes. Defaults to [64].
            **mask_args: extra mask args not needed for this function.
        """

        if mask_learn or mask_learn_sep:
            self.fmg_layer = LinearNet(
                fmg,
                input_size=self.first_layer_node_size,
                output_size=1 if mask_learn else self.num_particles,
                final_linear=True,
                **self.linear_args,
            )

    def _get_mask(
        self,
        x: Tensor,
        labels: Tensor = None,
        mask_learn: bool = False,
        mask_learn_bin: bool = True,
        mask_learn_sep: bool = False,
        mask_c: bool = True,
        mask_fne_np: bool = False,
        **mask_args,
    ):
        """
        Develops mask for input tensor ``x`` depending on the chosen masking strategy.

        Args:
            x (Tensor): input tensor.
            labels (Tensor): input jet level features - last feature should be # of particles in jet
              if ``mask_c``.Defaults to None.
            mask_learn (bool): learning a mask per particle using each particle's initial noise.
              Defaults to False.
            mask_learn_bin (bool): learn a binary mask as opposed to continuous. Defaults to True.
            mask_learn_sep (bool): predicting an overall number of particles per jet using separate
              jet noise. Defaults to False.
            mask_c (bool): using input # of particles per jet to automatically choose masks for
              particles. Defaults to True.
            mask_fne_np (bool): feed # of particle per jet as an input to the node and edge
              networks. Defaults to False.
            **mask_args: extra mask args not needed for this function.

        Returns:
            x (Tensor): modified input tensor
            use_mask (bool): is masking being used in message passing layers
            mask (Tensor): if ``use_mask`` then tensor of masks of shape
              ``[batch size, # nodes, 1 (mask)]``, else None.
            num_jet_particles (Tensor): if ``use_mask`` then tensor of # of particles per jet of
              shape ``[batch size, 1 (num particles)]``, else None.

        """

        use_mask = mask_learn or mask_c or mask_learn_sep

        if not use_mask:
            return x, use_mask, None, None

        num_jet_particles = None

        if mask_learn:
            # predict a mask from the noise per particle using the fmg fully connected network
            mask = self.fmg_layer(x)
            # sign function if learning a binary mask else sigmoid
            mask = torch.sign(mask) if mask_learn_bin else torch.sigmoid(mask)

            if mask_fne_np:
                # num_jet_particles will be an extra feature inputted to the edge and node networks
                num_jet_particles = torch.mean(mask, dim=1)

        elif mask_c:
            # unnormalize the last jet label - the normalized # of particles per jet
            # (between 1/``num_particles`` and 1) - to between 0 and ``num_particles`` - 1
            num_jet_particles = (labels[:, -1] * self.num_particles).int() - 1
            # sort the particles bythe first noise feature per particle, and the first
            # ``num_jet_particles`` particles receive a 1-mask, the rest 0.
            mask = (
                (x[:, :, 0].argsort(1).argsort(1) <= num_jet_particles.unsqueeze(1))
                .unsqueeze(2)
                .float()
            )

        elif mask_learn_sep:
            # last 'particle' in tensor is input to the fmg ``num_jet_particles`` prediction network
            num_jet_particles_input = x[:, -1, :]
            x = x[:, :-1, :]

            num_jet_particles = self.fmg_layer(num_jet_particles_input)
            num_jet_particles = torch.argmax(num_jet_particles, dim=1)
            # sort the particles by the first noise feature per particle, and the first
            # ``num_jet_particles`` particles receive a 1-mask, the rest 0.
            mask = (
                (x[:, :, 0].argsort(1).argsort(1) <= num_jet_particles.unsqueeze(1))
                .unsqueeze(2)
                .float()
            )

        return x, use_mask, mask, num_jet_particles

    def _final_mask(
        self,
        x: Tensor,
        mask: Tensor,
        mask_feat_bin: bool = False,
        **mask_args,
    ):
        """
        Process the output to get the final mask.

        Args:
            x (Tensor): processed data tensor.
            mask (Tensor): mask tensor, if being used in this model.
            mask_feat_bin (bool): use the last output feature as a binary mask. Defaults to False.
            **mask_args: extra mask args not needed for this function.

        Returns:
            type: final ``x`` tensor possibly including the mask as the last feature.

        """

        if mask_feat_bin:
            # take last output feature and make it binary
            mask = x[:, :, -1]
            x = x[:, :, :-1]

            if mask_feat_bin:
                mask = torch.sign(mask)

        return torch.cat((x, mask - 0.5), dim=2) if mask is not None else x

    def __repr__(self):
        lfc_str = f"LFC = {self.lfc_layer},\n" if self.lfc else ""
        fmg_str = f"FMG = {self.fmg_layer},\n" if hasattr(self, "fmg_layer") else ""
        return (
            f"{self.__class__.__name__}({lfc_str}{fmg_str}MPLayers ="
            f" {self.mp_layers})"
        )
