import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
from model_parts import DoubleConv, Up, Down, OutConv

class LatentSpaceDecoder(nn.Module):
  def __init__(self, latent_dim, signal_dim, n_layers, hidden_dim):
    super().__init__()
    if n_layers == 1:
      self.layers = nn.ModuleList([nn.Linear(latent_dim, signal_dim)])
    elif n_layers == 2:
      self.layers = nn.ModuleList([nn.Linear(latent_dim, hidden_dim),
                                nn.Linear(hidden_dim, signal_dim)])
    else:
      self.layers = nn.ModuleList([nn.Linear(latent_dim, hidden_dim)] \
                               + [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers-2)] \
                               + [nn.Linear(hidden_dim, signal_dim)])

  def forward(self, x):
    for layer in self.layers[:-1]:
      x = layer(x)
      x = nn.Tanh()(x)
    x =  self.layers[-1](x)
    return x


class LatentSpaceEncoder(nn.Module):
  def __init__(self, latent_dim, signal_dim, n_layers, hidden_channels):
    super().__init__()
    if n_layers == 1:
      self.conv_layers = nn.ModuleList([nn.Conv1d(signal_dim, latent_dim, kernel_size=3, padding=1)]) 
      self.res_layers = nn.ModuleList([nn.Conv1d(signal_dim, latent_dim, kernel_size=1)])

    elif n_layers == 2:
      self.conv_layers = nn.ModuleList([nn.Conv1d(signal_dim, hidden_channels, kernel_size=3, padding=1),
                                        nn.Conv1d(hidden_channels, latent_dim, kernel_size=3, padding=1)])
      self.res_layers = nn.ModuleList([nn.Conv1d(signal_dim, hidden_channels, kernel_size=1),
                                       nn.Conv1d(hidden_channels, latent_dim, kernel_size=1)])
      
    else:
      self.conv_layers = nn.ModuleList([nn.Conv1d(signal_dim, hidden_channels, kernel_size=3, padding=1)] \
                                       + [nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1) for _ in range(n_layers-2)] \
                                       + [nn.Conv1d(hidden_channels, latent_dim, kernel_size=3, padding=1)])
      self.res_layers = nn.ModuleList([nn.Conv1d(signal_dim, hidden_channels, kernel_size=1)] \
                                      + [nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1) for _ in range(n_layers-2)] \
                                      + [nn.Conv1d(hidden_channels, latent_dim, kernel_size=1)])

  def forward(self, x):
    for conv_layer, res_layer in zip(self.conv_layers[:-1], self.res_layers[:-1]):
      out = conv_layer(x)
      x = res_layer(x)
      x += out
      x = nn.Tanh()(x)

    out = self.conv_layers[-1](x)
    x = self.res_layers[-1](x)
    x += out

    return x


class UNetLikeLatentSpaceEncoder(nn.Module):
    def __init__(self, latent_dim, signal_dim, min_channels, n_layers, act='ReLU'):
        super(UNetLikeLatentSpaceEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.signal_dim = signal_dim
        self.min_channels = min_channels
        self.n_layers = n_layers
        self.act = act

        self.inc = DoubleConv(signal_dim, min_channels, act)

        self.down = nn.ModuleList([])
        n_channels = self.min_channels
        for _ in range(self.n_layers):
          self.down.append(Down(n_channels, 2 * n_channels, act))
          n_channels = 2 * n_channels

        self.up = nn.ModuleList([])
        for _ in range(self.n_layers):
          self.up.append(Up(n_channels, n_channels // 2, act))
          n_channels = n_channels // 2
      
        self.outc = OutConv(n_channels, self.latent_dim)

    def forward(self, x):
        down_out = [self.inc(x)]
        for layer in self.down:
          down_out.append(layer(down_out[-1]))

        x = down_out[-1]

        for layer, down_x in zip(self.up, down_out[-2::-1]):
          x = layer(x, down_x)

        x = self.outc(x)

        return x


class RHS(nn.Module):
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


class NODESolver(nn.Module):
  def __init__(self, latent_dim, signal_dim,
               encoder_n_layers, encoder_hidden_channels,
               decoder_n_layers, decoder_hidden_dim,
               rhs_n_layers, rhs_hidden_dim):
    super().__init__()

    self.decoder = LatentSpaceDecoder(latent_dim, signal_dim, decoder_n_layers, decoder_hidden_dim)
    # self.encoder = LatentSpaceEncoder(latent_dim, signal_dim, encoder_n_layers, encoder_hidden_channels)
    self.encoder = UNetLikeLatentSpaceEncoder(latent_dim, signal_dim, encoder_hidden_channels, encoder_n_layers)
    self.rhs = RHS(latent_dim, rhs_n_layers, rhs_hidden_dim)

  def forward(self, y, t):
    z0 = self.encoder(y)[:, :, 0]
    pred_z = odeint(self.rhs, z0, t).to(y.device)
    pred_y = self.decoder(pred_z).permute(1, 2, 0)
    return pred_y

  def autoencoder_forward(self, y): 
    z = self.encoder(y)
    pred_y = self.decoder(z.transpose(1, 2)).transpose(1, 2)
    return pred_y