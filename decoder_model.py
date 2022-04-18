import torch

from pointwise_nets import LinearNet, FCNet, AlgebraicNet, PointwiseTransformerNet, PointwisePermformerNet
from seq2seq_nets import UNetLikeConvNet, Seq2SeqTransformerNet


class SimpleLatentSpaceDecoder(LinearNet):
    def __init__(self, latent_dim, signal_dim):
        super().__init__(input_dim=latent_dim, output_dim=signal_dim)


class FCLatentSpaceDecoder(FCNet):
    def __init__(self, latent_dim, signal_dim, n_layers, hidden_dim):
        super().__init__(input_dim=latent_dim, output_dim=signal_dim, n_layers=n_layers, hidden_dim=hidden_dim)


class AlgebraicLatentSpaceDecoder(AlgebraicNet):
    def __init__(self, latent_dim, signal_dim, n_layers, hidden_dim):
        super().__init__(input_dim=latent_dim, output_dim=signal_dim, n_layers=n_layers, hidden_dim=hidden_dim)


class TransformerLatentSpaceDecoder(PointwiseTransformerNet):
    def __init__(self, latent_dim, signal_dim, n_layers, nhead, dim_feedforward, dropout, activation):
        super().__init__(input_dim=latent_dim, output_dim=signal_dim, n_layers=n_layers, nhead=nhead,
                         dim_feedforward=dim_feedforward, dropout=dropout, activation=activation)


class PermformerLatentSpaceDecoder(PointwisePermformerNet):
    def __init__(self, latent_dim, signal_dim, n_layers, nhead, dim_feedforward, dropout, activation):
        super().__init__(input_dim=latent_dim, output_dim=signal_dim, n_layers=n_layers, nhead=nhead,
                         dim_feedforward=dim_feedforward, dropout=dropout, activation=activation)


class UNetLikeConvLatentSpaceDecoder(UNetLikeConvNet):
    def __init__(self, latent_dim, signal_dim, min_channels, n_layers, act='ReLU', always_decrease_n_ch=False):
        super().__init__(input_dim=latent_dim, output_dim=signal_dim,
                         min_channels=min_channels, n_layers=n_layers, act=act, always_decrease_n_ch=always_decrease_n_ch)

    def forward(self, z):
        z = z.T[None, :, :] # (1, latent_dim, T)
        y = super().forward(z).permute(0, 2, 1)[0] # (1, signal_dim, T)
        return y


class Seq2SeqTransformerLatentSpaceDecoder(Seq2SeqTransformerNet):
    def __init__(self, latent_dim, signal_dim, n_layers, nhead, dim_feedforward, dropout, activation):
        super().__init__(input_dim=latent_dim, output_dim=signal_dim, n_layers=n_layers, nhead=nhead,
                         dim_feedforward=dim_feedforward, dropout=dropout, activation=activation)

    def forward(self, z):
        z = z.T[None, :, :] # (1, latent_dim, T)
        y = super().forward(z).permute(0, 2, 1)[0] # (1, signal_dim, T)
        return y
