import torch.nn as nn


class FCRHS(nn.Module):
  def __init__(self, system_dim, n_layers, hidden_dim):
    super().__init__()
    if n_layers == 1:
      self.layers = nn.ModuleList([nn.Linear(system_dim, system_dim)])
    elif n_layers == 2:
      self.layers = nn.ModuleList([nn.Linear(system_dim, hidden_dim),
                                   nn.Linear(hidden_dim, system_dim)])
    else:
      self.layers = nn.ModuleList([nn.Linear(system_dim, hidden_dim)] \
                                  + [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers-2)] \
                                  + [nn.Linear(hidden_dim, system_dim)])
    

  def forward(self, t, x):
    for layer in self.layers[:-1]:
      x = layer(x)
      x = nn.Tanh()(x)

    x = self.layers[-1](x)
    
    return x
