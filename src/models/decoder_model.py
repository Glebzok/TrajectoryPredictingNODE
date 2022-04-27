import torch.nn as nn


class TrivialDecoder(nn.Module):
    def __init__(self):
        super(TrivialDecoder, self).__init__()

    def forward(self, z):
        # z: (n_samples, latent_dim, T)
        return z


class NontrivialDecoder(nn.Module):
    def __init__(self, decoder_net, dropout, **kwargs):
        super(NontrivialDecoder, self).__init__()
        self.decoder_net = decoder_net

        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None

    def forward(self, z):
        # z: (n_samples, latent_dim, T)
        if self.dropout is not None:
            z = self.dropout(z)
        return self.decoder_net(z)  # (n_samples, signal_dim, T)
