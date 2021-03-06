import torch
import torch.nn as nn


class LinearNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(in_features=input_dim, out_features=output_dim, bias=False)

    def forward(self, x):
        # x: (bs, input_dim)
        x = self.linear(x)  # x: (bs, output_dim)
        return x


class FCNet(nn.Module):
    @staticmethod
    def _get_linear(input_dim, output_dim, normalized, bias):
        if normalized:
            return nn.utils.parametrizations.spectral_norm(nn.Linear(input_dim, output_dim, bias=bias))
        else:
            return nn.Linear(input_dim, output_dim, bias=bias)

    def __init__(self, input_dim, output_dim, n_layers, hidden_dim, activation, normalized, dropouts=[], last_bias=True):
        super().__init__()
        self.activation = activation

        if n_layers == 1:
            self.layers = nn.ModuleList([self._get_linear(input_dim, output_dim, normalized, bias=last_bias)])
        elif n_layers == 2:
            self.layers = nn.ModuleList([self._get_linear(input_dim, hidden_dim, normalized, bias=True),
                                         self._get_linear(hidden_dim, output_dim, normalized, bias=last_bias)])
        else:
            self.layers = nn.ModuleList([self._get_linear(input_dim, hidden_dim, normalized, bias=True)] \
                                        + [self._get_linear(hidden_dim, hidden_dim, normalized, bias=True) for _ in
                                           range(n_layers - 2)] \
                                        + [self._get_linear(hidden_dim, output_dim, normalized, bias=last_bias)])

        self.dropouts = [nn.Dropout(p=dropout) for dropout in dropouts]

    def forward(self, x):
        # x: (bs, input_dim)
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            if self.activation == 'tanh':
                x = torch.tanh(x)
            else:
                x = torch.relu(x)

            if i < len(self.dropouts):
                x = self.dropouts[i](x)

        x = self.layers[-1](x)
        return x  # x: (bs, output_dim)
