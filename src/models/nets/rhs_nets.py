import torch.nn as nn
import torch.nn.functional as F
import torch

from src.models.backbones.pointwise_nets import FCNet


class StableLinear(nn.Module):
    def __init__(self, n, use_random_projection_init, norm, skewsymmetricity_alpha=0):
        super().__init__()
        self.X = nn.parameter.Parameter(self.get_normalized_matrix(n))
        self.K = nn.parameter.Parameter(self.get_normalized_matrix(n))

        self.norm = norm
        self.skewsymmetricity_alpha = skewsymmetricity_alpha

        if use_random_projection_init:
            self.init_xk_values(n=n)

    def init_xk_values(self, n):
        A = self.get_normalized_matrix(n) * self.norm
        B = self.get_normalized_matrix(n) * self.norm
        A = A * (1 - self.skewsymmetricity_alpha) + self.skewsymmetricity_alpha * (B - B.T)

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


class StableDHLinear(StableLinear):
    def __init__(self, n, use_random_projection_init, eps, norm, skewsymmetricity_alpha=0):
        super().__init__(n=n, use_random_projection_init=use_random_projection_init, norm=norm,
                         skewsymmetricity_alpha=skewsymmetricity_alpha)
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
    def __init__(self, system_dim, linear, **kwargs):
        super().__init__()
        self.system_dim = system_dim
        self.dynamics = linear

    def forward(self, t, x):
        x = self.dynamics(x)
        return x


class FCRHS(nn.Module):
    def __init__(self, system_dim, n_layers, hidden_dim, activation, normalized, **kwargs):
        super().__init__()
        self.dynamics = FCNet(input_dim=system_dim, output_dim=system_dim,
                              n_layers=n_layers, hidden_dim=hidden_dim,
                              activation=activation, normalized=normalized)
        self.system_dim = system_dim

    def forward(self, t, x):
        x = self.dynamics(x)
        return x


class ControlledLinearRHS(nn.Module):
    def __init__(self, signal_dim, system_dim,
                 decoder, linear,
                 n_layers, hidden_dim, normalized_controller, controller_activation, **kwargs):
        super().__init__()
        self.system_dim = system_dim
        self.controller = FCNet(input_dim=signal_dim, output_dim=system_dim,
                                n_layers=n_layers, hidden_dim=hidden_dim,
                                activation=controller_activation, normalized=normalized_controller)

        self.dynamics = linear
        self.decoder = decoder

    def forward(self, t, x):
        init_shape = x.shape
        y = self.controller(self.decoder(x.view(1, -1, init_shape[-1]).permute(0, 2, 1)).permute(0, 2, 1))
        if len(init_shape) == 2:
            y = y[0]
        x = self.dynamics(x) + y
        return x
