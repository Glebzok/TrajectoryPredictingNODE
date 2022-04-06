import torch.nn as nn
import torch


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


class SimpleRHS(nn.Module):
    def __init__(self, system_dim, T=1, use_hippo_init=False):
        super().__init__()
        self.system_dim = system_dim
        # self.linear = nn.utils.parametrizations.spectral_norm(nn.Linear(in_features=system_dim, out_features=system_dim, bias=False))
        self.linear = nn.Linear(in_features=system_dim, out_features=system_dim, bias=False)
        # self.linear.weight.data *= 3
        # nn.utils.parametrize.register_parametrization(self.linear, 'weight', SpectralShift())

        # self.dropout1 = nn.Dropout(p=0.3)
        # self.dropout2 = nn.Dropout(p=0.3)

        if use_hippo_init:
            self.linear.parametrizations.weight.original.data = get_hippo_matrix(system_dim)

        # self.linear.weight.data /= T
        # self.linear.parametrizations.weight.original.data *= 10

    def forward(self, t, x):
        # x = self.dropout1(x)
        x = self.linear(x)
        # x = self.dropout2(x)
        return x


class FCRHS(nn.Module):
    def __init__(self, system_dim, n_layers, hidden_dim):
        super().__init__()
        self.system_dim = system_dim
        if n_layers == 1:
            self.layers = nn.ModuleList([nn.Linear(system_dim, system_dim)])
        elif n_layers == 2:
            self.layers = nn.ModuleList([nn.Linear(system_dim, hidden_dim),
                                         nn.Linear(hidden_dim, system_dim)])
        else:
            self.layers = nn.ModuleList([nn.Linear(system_dim, hidden_dim)] \
                                        + [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers - 2)] \
                                        + [nn.Linear(hidden_dim, system_dim)])

    def forward(self, t, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = nn.Tanh()(x)

        x = self.layers[-1](x)

        return x
