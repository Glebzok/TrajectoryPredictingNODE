import torch
import torch.nn as nn


class TrivialEncoder(nn.Module):
    def __init__(self, n_shooting_vars, **kwargs):
        super(TrivialEncoder, self).__init__()
        self.n_shooting_vars = n_shooting_vars

    def forward(self, y):
        # y : (signal_dim, T)
        return y[:, :-1:y.shape[1] // self.n_shooting_vars].T  # (n_shooting_vars, signal_dim)


class DirectOptimizationEncoder(nn.Module):
    def __init__(self, n_shooting_vars, latent_dim, init_distribution, **kwargs):
        super(DirectOptimizationEncoder, self).__init__()
        if init_distribution == 'normal':
            self.shooting_vars = torch.randn(n_shooting_vars, latent_dim)  # N(0, 1)
        elif init_distribution == 'normal separate':
            self.shooting_vars = torch.randn(n_shooting_vars, latent_dim)  # N(0, 1)
            self.shooting_vars[:, latent_dim // 2:] *= 0.5  # N(0, 1/4)
        else:
            self.shooting_vars = (torch.rand(n_shooting_vars, latent_dim) - 0.5) * 2. # U(-1, 1)
        self.shooting_vars = nn.Parameter(self.shooting_vars)  # (n_shooting_vars, latent_dim)

    def forward(self, y):
        # y : (signal_dim, T)
        return self.shooting_vars  # (n_shooting_vars, latent_dim)


class NontrivialEncoder(nn.Module):
    def __init__(self, n_shooting_vars, encoder_net, **kwargs):
        super(NontrivialEncoder, self).__init__()
        self.n_shooting_vars = n_shooting_vars
        self.encoder_net = encoder_net

    def forward(self, y):
        # y : (signal_dim, T)
        return self.encoder_net(y[None, :, :])[0, :, :-1:y.shape[1] // self.n_shooting_vars].T  # (n_shooting_vars, latent_dim)
