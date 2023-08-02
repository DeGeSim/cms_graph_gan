from torch.nn import LeakyReLU, Linear, Module
from torch.nn.utils import parametrizations


class GatedCondition(Module):
    def __init__(
        self, x_dim: int, res_dim: int, out_dim: int, spectral_norm: bool = True
    ) -> None:
        super().__init__()
        self.x_dim = x_dim
        self.res_dim = res_dim
        self.out_dim = out_dim
        self.x_tf = Linear(x_dim, 2 * out_dim)
        self.res_tf = Linear(res_dim, 2 * out_dim)
        self.joined_lin = Linear(2 * out_dim, 2 * out_dim)
        self.act = LeakyReLU(0.1)
        if spectral_norm:
            self.x_tf = parametrizations.spectral_norm(self.x_tf)
            self.res_tf = parametrizations.spectral_norm(self.res_tf)
            self.joined_lin = parametrizations.spectral_norm(self.joined_lin)

    def forward(self, x, res):
        assert x.shape[:-1] == res.shape[:-1]
        assert x.shape[-1] == self.x_dim
        assert res.shape[-1] == self.res_dim
        x = self.x_tf(x) + self.res_tf(res)
        x = self.joined_lin(x)
        a, b = x.split(self.out_dim, -1)
        return a * self.act(b)
