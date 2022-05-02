from src.models.backbones.pointwise_nets import LinearNet, FCNet
from src.models.backbones.seq2seq_nets import UNetLikeConvNet, Seq2SeqTransformerNet, ShrinkingResNet, MixNet


class SimpleLatentSpaceDecoder(LinearNet):
    def __init__(self, latent_dim, signal_dim):
        super().__init__(input_dim=latent_dim, output_dim=signal_dim)

    def forward(self, z):
        # z: (n_samples, latent_dim, T)
        return super().forward(z.permute(0, 2, 1)).permute(0, 2, 1)  # (n_samples, signal_dim, T)


class FCLatentSpaceDecoder(FCNet):
    def __init__(self, latent_dim, signal_dim, n_layers, hidden_dim, activation, normalized, dropouts):
        super().__init__(input_dim=latent_dim, output_dim=signal_dim, n_layers=n_layers, hidden_dim=hidden_dim,
                         activation=activation, normalized=normalized, dropouts=dropouts)

    def forward(self, z):
        # z: (n_samples, latent_dim, T)
        return super().forward(z.permute(0, 2, 1)).permute(0, 2, 1)  # (n_samples, signal_dim, T)


class UNetLikeConvLatentSpaceDecoder(UNetLikeConvNet):
    def __init__(self, latent_dim, signal_dim, init_channels, n_layers, act, always_decrease_n_ch):
        super().__init__(input_dim=latent_dim, output_dim=signal_dim,
                         init_channels=init_channels, n_layers=n_layers, act=act,
                         always_decrease_n_ch=always_decrease_n_ch)


class ShrinkingResLatentSpaceDecoder(ShrinkingResNet):
    def __init__(self, latent_dim, signal_dim, n_layers):
        super().__init__(input_dim=latent_dim, output_dim=signal_dim, n_layers=n_layers)


class Seq2SeqTransformerLatentSpaceDecoder(Seq2SeqTransformerNet):
    def __init__(self, latent_dim, signal_dim, n_layers, nhead, dim_feedforward, dropout, activation):
        super().__init__(input_dim=latent_dim, output_dim=signal_dim, n_layers=n_layers, nhead=nhead,
                         dim_feedforward=dim_feedforward, dropout=dropout, activation=activation)


class MixNetLatentSpaceDecoder(MixNet):
    def __init__(self, latent_dim, signal_dim, n_layers, hidden_dim, activation, normalized, conv_index, dropouts):
        super().__init__(input_dim=latent_dim, output_dim=signal_dim, n_layers=n_layers,
                         hidden_dim=hidden_dim, activation=activation, normalized=normalized,
                         conv_index=conv_index, dropouts=dropouts)
