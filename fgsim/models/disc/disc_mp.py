import torch
from torch import Tensor

from fgsim.models.common.mpgan import LinearNet, MPNet
from fgsim.utils.jetnetutils import to_stacked_mask


class ModelClass(MPNet):
    """
    Message passing discriminator.
    Goes through ``mp_iters`` iterations of message passing and then an optional final fully
    connected network to output a scalar prediction.

    A number of options for masking are implemented, as described in the appendix of
    Kansal et. al. *Particle Cloud Generation with Message Passing Generative Adversarial Networks*
    (https://arxiv.org/abs/2106.11535).
    Args for masking are described in the masking functions below.

    Input ``x`` tensor to the forward pass must be of shape
    ``[batch_size, num_particles, input_node_size]``.

    Args:
        dea (bool): 'discriminator early aggregation' i.e. aggregate the final graph and pass
          through a final fully connected network ``fnd``. Defaults to True.
        dea_sum (bool): if using ``dea``, use 'sum' as the aggregation operation as opposed to
          'mean'. Defaults to True.
        fnd (list): list of final FC network intermediate layer sizes. Defaults to [].
        mask_fnd_np (bool): pass number of particles as an extra feature into the final FC network.
          Defaults to False.
        **mpnet_args: args for ``MPNet`` base class.
    """

    def __init__(
        self,
        dea: bool = True,
        dea_sum: bool = True,
        fnd: list = [],
        mask_fnd_np: bool = False,
        **mpnet_args,
    ):
        super().__init__(output_node_size=1 if not dea else 0, **mpnet_args)

        self.dea = dea
        self.dea_sum = dea_sum

        self.mask_fnd_np = mask_fnd_np

        # final fully connected classification layer
        if dea:
            self.fnd_layer = LinearNet(
                fnd,
                input_size=self.hidden_node_size + int(mask_fnd_np),
                output_size=1,
                final_linear=True,
                **self.linear_args,
            )

    # Edit start
    def forward(self, batch, condition):
        assert condition.squeeze().dim() == 1
        x = to_stacked_mask(batch)
        x = super().forward(x, labels=condition.squeeze()).squeeze()
        return x

    # Edit end

    def _post_mp(self, x, labels, use_mask, mask, num_jet_particles):
        do_mean = not (
            self.dea and self.dea_sum
        )  # only summing if using ``dea`` and ``dea_sum`` is True
        if use_mask:
            # only sum contributions from 1-masked particles
            x = x * mask
            x = torch.sum(x, 1)
            if do_mean:
                # only divide by number of 1-masked particle per jet
                x = x / (torch.sum(mask, 1) + 1e-12)
        else:
            x = torch.mean(x, 1) if do_mean else torch.sum(x, 1)

        # feed into optional final FC network
        if self.dea:
            if self.mask_fnd_np:
                x = torch.cat((num_jet_particles, x), dim=1)

            x = self.fnd_layer(x)

        return x

    def _get_mask(
        self,
        x: Tensor,
        labels: Tensor,
        mask_manual: bool = False,
        mask_learn: bool = False,
        mask_learn_sep: bool = False,
        mask_c: bool = True,
        mask_fne_np: bool = False,
        mask_fnd_np: bool = False,
        **mask_args,
    ):
        """
        Develops mask for input tensor ``x`` depending on the chosen masking strategy.

        Args:
            x (Tensor): input tensor.
            mask_manual (bool): applying a manual mask after generation per particle based on a pT
              cutoff.
            mask_learn (bool): learning a mask per particle using each particle's initial noise.
              Defaults to False.
            mask_learn_sep (bool): predicting an overall number of particles per jet using separate
              jet noise. Defaults to False.
            mask_c (bool): using input # of particles per jet to automatically choose masks for
              particles. Defaults to True.
            mask_fne_np (bool): feed # of particle per jet as an input to the node and edge
              networks. Defaults to False.
            mask_fnd_np (bool): feed # of particle per jet as an input to final discriminator FC
              network. Defaults to False.
            **mask_args: extra mask args not needed for this function.

        Returns:
            x (Tensor): modified data tensor
            use_mask (bool): is masking being used
            mask (Tensor): if ``use_mask`` then tensor of masks of shape
              ``[batch size, # nodes, 1 (mask)]``, else None
            num_jet_particles (Tensor): if ``use_mask`` then tensor of # of particles per jet of
              shape ``[batch size, 1 (num particles)]``, else None.

        """

        mask = None
        num_jet_particles = None

        use_mask = mask_manual or mask_learn or mask_c or mask_learn_sep

        # separate mask from other features
        if use_mask or mask_fnd_np:
            mask = x[:, :, -1:] + 0.5

        if use_mask:
            x = x[:, :, :-1]

        if mask_fne_np:
            num_jet_particles = torch.mean(mask, dim=1)

        return x, use_mask, mask, num_jet_particles

    def __repr__(self):
        dea_str = f",\nFND = {self.fnd_layer}" if self.dea else ""
        return f"{self.__class__.__name__}(MPLayers = {self.mp_layers}{dea_str})"
