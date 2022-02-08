import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleLatentSpaceDecoder(nn.Module):
    def __init__(self, latent_dim, signal_dim):
        super().__init__()
        self.linear = nn.Linear(in_features=latent_dim, out_features=signal_dim, bias=False)

    def forward(self, x):
        x = self.linear(x)
        return x


class FCLatentSpaceDecoder(nn.Module):
    def __init__(self, latent_dim, signal_dim, n_layers, hidden_dim):
        super().__init__()
        if n_layers == 1:
            self.layers = nn.ModuleList([nn.Linear(latent_dim, signal_dim)])
        elif n_layers == 2:
            self.layers = nn.ModuleList([nn.Linear(latent_dim, hidden_dim),
                                         nn.Linear(hidden_dim, signal_dim)])
        else:
            self.layers = nn.ModuleList([nn.Linear(latent_dim, hidden_dim)] \
                                        + [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers - 2)] \
                                        + [nn.Linear(hidden_dim, signal_dim)])

    def forward(self, x):
        # p = 0.3
        # x = self.layers[0](x)
        # x = nn.Tanh()(x)
        for layer in self.layers[:-1]:
            # x = F.dropout(x, p, self.training)
            x = layer(x)
            x = nn.Tanh()(x)
            # x = x + y
        x = self.layers[-1](x)
        return x


class InnerProduct(nn.Module):
    def __init__(self, in_features):
        super().__init__()

    def forward(self, x):
        # bs, in_features
        x = x[:, :, None] @ x[:, None, :]
        x = x[torch.triu(torch.ones(x.shape)) == 1].view(x.shape[0], -1)
        # bs, in_features * (in_features+1) / 2
        return x


class Power(nn.Module):
    def __init__(self, in_features, n_powers):
        super().__init__()
        self.W = nn.Parameter(torch.rand((1, in_features, n_powers)) - 0.5)

    def forward(self, x):
        # bs, in_features
        x = F.relu(x) + 1e-9
        x = (x[:, :, None] ** self.W).view(x.shape[0], -1)
        # bs, in_features * n_powers
        return x


class Algebraic(nn.Module):
    def __init__(self, in_features, n_powers):
        super().__init__()
        self.inner_product = InnerProduct(in_features=in_features)
        inner_product_out_features = in_features * (in_features + 1) // 2

        self.power = Power(in_features=in_features, n_powers=n_powers)
        power_out_features = in_features * n_powers

        self.linear = nn.Linear(in_features=in_features + inner_product_out_features + power_out_features,
                                out_features=in_features)

    def forward(self, x):
        x1 = self.inner_product(x)
        x2 = self.power(x)

        x = x + self.linear(torch.cat([x, x1, x2], dim=1))
        return x


class AlgebraicLatentSpaceDecoder(nn.Module):
    def __init__(self, latent_dim, signal_dim, n_layers, hidden_dim):
        super().__init__()
        self.l1 = nn.Linear(in_features=latent_dim, out_features=hidden_dim)
        self.alg = nn.ModuleList([Algebraic(in_features=hidden_dim, n_powers=5) for _ in range(n_layers)])
        self.l2 = nn.Linear(in_features=hidden_dim, out_features=signal_dim)

    def forward(self, x):
        bs, n_points, latent_dim = x.shape
        x = x.view(-1, latent_dim)
        x = self.l1(x)

        for l in self.alg:
            x = l(x)
            x = F.relu(x)

        x = self.l2(x)
        x = x.view(bs, n_points, -1)
        return x
