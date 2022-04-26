import torch.nn as nn


class TrivialDecoder(nn.Module):
    def __init__(self):
        super(TrivialDecoder, self).__init__()

    def forward(self, z):
        # z: (n_samples, latent_dim, T)
        return z


class NontrivialDecoder(nn.Module):
    def __init__(self, decoder_net):
        super(NontrivialDecoder, self).__init__()
        self.decoder_net = decoder_net

    def forward(self, z):
        # z: (n_samples, latent_dim, T)
        return self.decoder_net(z)  # (n_samples, signal_dim, T)
