import torch.nn as nn


class SimpleRHS(nn.Module):
    def __init__(self, system_dim):
        super().__init__()
        self.system_dim = system_dim
        self.linear = nn.Linear(in_features=system_dim, out_features=system_dim, bias=False)

    def forward(self, t, x):
        return self.linear(x)


class FCRHS(nn.Module):
    def __init__(self, system_dim, n_layers, hidden_dim):
        super().__init__()
        self.system_dim = system_dim
        if n_layers == 1:
            self.layers = nn.ModuleList([nn.utils.spectral_norm(nn.Linear(system_dim, system_dim))])
        elif n_layers == 2:
            self.layers = nn.ModuleList([nn.utils.spectral_norm(nn.Linear(system_dim, hidden_dim)),
                                         nn.utils.spectral_norm(nn.Linear(hidden_dim, system_dim))])
        else:
            self.layers = nn.ModuleList([nn.utils.spectral_norm(nn.Linear(system_dim, hidden_dim))] \
                                        + [nn.utils.spectral_norm(nn.Linear(hidden_dim, hidden_dim)) for _ in range(n_layers - 2)] \
                                        + [nn.utils.spectral_norm(nn.Linear(hidden_dim, system_dim))])

    def forward(self, t, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = nn.Tanh()(x)

        x = self.layers[-1](x)

        return x
