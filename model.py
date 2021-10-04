import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint

from rhs_model import FCRHS
from decoder_model import FCLatentSpaceDecoder
from encoder_model import ConvLatentSpaceEncoder, UNetLikeLatentSpaceEncoder, TransformerLatentSpaceEncoder


class NODESolver(nn.Module):
  def __init__(self, latent_dim, signal_dim,
               encoder_n_layers, encoder_hidden_channels,
               decoder_n_layers, decoder_hidden_dim,
               rhs_n_layers, rhs_hidden_dim):
    super().__init__()

    self.decoder = FCLatentSpaceDecoder(latent_dim, signal_dim, decoder_n_layers, decoder_hidden_dim)
    # self.encoder = ConvLatentSpaceEncoder(latent_dim, signal_dim, encoder_n_layers, encoder_hidden_channels)
    # self.encoder = UNetLikeLatentSpaceEncoder(latent_dim, signal_dim, encoder_hidden_channels, encoder_n_layers)
    self.encoder = TransformerLatentSpaceEncoder(latent_dim, signal_dim, encoder_n_layers, 8, encoder_hidden_channels, 0, 'relu')
    self.rhs = FCRHS(latent_dim, rhs_n_layers, rhs_hidden_dim)

  def forward(self, y, t):
    z = self.encoder(y)
    pred_z = odeint(self.rhs, z[:, :, 0], t).to(y.device)
    pred_y = self.decoder(pred_z).permute(1, 2, 0)
    return z, pred_z.permute(1, 2, 0), pred_y

  def autoencoder_forward(self, y): 
    z = self.encoder(y)
    pred_y = self.decoder(z.transpose(1, 2)).transpose(1, 2)
    return pred_y
