import torch.nn as nn
import torch.nn.functional as F
import torch

from pointwise_nets import FCNet, LinearNet


def get_hippo_matrix(n):
  row = ((2 * torch.arange(1, n+1, 1) + 1) ** 0.5)
  diagonal = -torch.diag(torch.arange(1, n+1, 1))
  hippo_matrix = -torch.tril(row.view(-1, 1) @ row.view(1, -1), diagonal=-1)

  return (hippo_matrix + diagonal).T


class SpectralShift(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        max_real_eigv_part = torch.linalg.eigvals(X).real.max().detach()
        return X - max_real_eigv_part * torch.eye(X.shape[0], device=X.device)


class StableLinear(nn.Module):
    def __init__(self, n, use_random_projection_init):
        super().__init__()
        self.X = nn.parameter.Parameter(self.get_normalized_matrix(n))
        self.K = nn.parameter.Parameter(self.get_normalized_matrix(n))

        if use_random_projection_init:
            self.init_xk_values(n=n)

    def init_xk_values(self, n):
        A = self.get_normalized_matrix(n) * 10
        self.K.data = 0.5 * (A - A.T)

        L, U = torch.linalg.eigh(0.5 * (- A - A.T))
        self.X.data = U @ torch.diag(F.relu(L) ** 0.5)

    @property
    def weight(self):
        K = self.K.triu(1)
        K = K - K.T
        return - self.X @ self.X.T + K

    @staticmethod
    def get_normalized_matrix(n):
        X = torch.rand((n, n))
        X /= torch.linalg.norm(X)
        return X

    def forward(self, x):
        return F.linear(x, self.weight)


class StableLinearV2(StableLinear):
    def __init__(self, n):
        super().__init__(n=n, use_random_projection_init=False)
        self.Y = nn.parameter.Parameter(self.get_normalized_matrix(n))
        self.X.data = torch.eye(n)

    @property
    def weight(self):
        K = self.K.triu(1)
        K = K - K.T
        return (self.X @ self.X.T) @ (K - self.Y @ self.Y.T)


class StableLinearV3(StableLinear):
    def __init__(self, n, use_random_projection_init, eps=1e-7):
        super().__init__(n=n, use_random_projection_init=use_random_projection_init)
        self.eps = eps
        self.Y = nn.parameter.Parameter(self.get_normalized_matrix(n))
        self.z = nn.parameter.Parameter(self.get_normalized_vector(n))
        if use_random_projection_init:
            self.init_yz_values(n)

    def init_yz_values(self, n):
        self.Y.data = torch.zeros_like(self.Y)
        self.z.data = torch.ones_like(self.z) - self.eps

    @staticmethod
    def get_normalized_vector(n):
        x = torch.rand(n)
        x /= torch.linalg.norm(x)
        return x

    @property
    def weight(self):
        K = self.K.triu(1)
        K = K - K.T

        Y = self.Y.tril(-1) + torch.diag(F.relu(self.z) + self.eps)
        Y = Y @ Y.T

        X = self.X @ self.X.T

        return (K - X) @ Y


class SimpleRHS(nn.Module):
    def __init__(self, system_dim, T=1, use_hippo_init=False):
        super().__init__()
        self.system_dim = system_dim
        # self.linear = nn.utils.parametrizations.spectral_norm(nn.Linear(in_features=system_dim, out_features=system_dim, bias=False))
        # self.linear = nn.Linear(in_features=system_dim, out_features=system_dim, bias=False)
        self.linear = StableLinearV3(n=system_dim, use_random_projection_init=True)
        # self.linear.weight.data *= 3
        # nn.utils.parametrize.register_parametrization(self.linear, 'weight', SpectralShift())

        # self.dropout1 = nn.Dropout(p=0.3)
        # self.dropout2 = nn.Dropout(p=0.3)

        # if use_hippo_init:
        #     self.linear.parametrizations.weight.original.data = get_hippo_matrix(system_dim)

        # self.linear.weight.data /= T
        # self.linear.parametrizations.weight.original.data *= 10

    def forward(self, t, x):
        # x = self.dropout1(x)
        x = self.linear(x)
        # x = self.dropout2(x)
        return x


class FCRHS(FCNet):
    def __init__(self, system_dim, n_layers, hidden_dim):
        super().__init__(input_dim=system_dim, output_dim=system_dim,
                         n_layers=n_layers, hidden_dim=hidden_dim)
        self.system_dim = system_dim

    def forward(self, t, x):
        super().forward(x)

        return x


class ControlledLinearRHS(nn.Module):
    def __init__(self, latent_dim, signal_dim, decoder, n_layers, hidden_dim):
        super().__init__()
        self.system_dim = latent_dim
        self.linear = StableLinear(n=latent_dim)
        self.controller = FCNet(input_dim=signal_dim, output_dim=latent_dim,
                                n_layers=n_layers, hidden_dim=hidden_dim)

        self.decoder = decoder

    def forward(self, t, x):
        x = self.linear(x) + self.controller(self.decoder(x))
        return x
