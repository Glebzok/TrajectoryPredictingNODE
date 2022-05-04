from src.models.backbones.pointwise_nets import FCNet
from src.models.backbones.seq2seq_nets import SingleLayerConvNet, ConvNet, UNetLikeConvNet, Seq2SeqTransformerNet


class FCLatentSpaceEncoder(FCNet):
    def __init__(self, latent_dim, signal_dim, n_layers, hidden_dim, activation, normalized, dropouts=[]):
        super().__init__(input_dim=signal_dim, output_dim=latent_dim,
                         n_layers=n_layers, hidden_dim=hidden_dim, activation=activation,
                         normalized=normalized, dropouts=dropouts)

    def forward(self, x):
        # x: (bs, input_dim, T)
        return super().forward(x.permute(0, 2, 1)).permute(0, 2, 1)


class SimpleLatentSpaceEncoder(SingleLayerConvNet):
    def __init__(self, latent_dim, signal_dim):
        super().__init__(input_dim=signal_dim, output_dim=latent_dim)


class ConvLatentSpaceEncoder(ConvNet):
    def __init__(self, latent_dim, signal_dim, n_layers, hidden_channels, activation):
        super().__init__(input_dim=signal_dim, output_dim=latent_dim,
                         n_layers=n_layers, hidden_channels=hidden_channels, activation=activation)


class UNetLikeLatentSpaceEncoder(UNetLikeConvNet):
    def __init__(self, latent_dim, signal_dim, init_channels, n_layers, activation):
        super().__init__(input_dim=signal_dim, output_dim=latent_dim,
                         init_channels=init_channels, n_layers=n_layers, act=activation, always_decrease_n_ch=False)


class TransformerLatentSpaceEncoder(Seq2SeqTransformerNet):
    def __init__(self, latent_dim, signal_dim, n_layers, nhead, dim_feedforward, dropout, activation):
        super().__init__(input_dim=signal_dim, output_dim=latent_dim, n_layers=n_layers, nhead=nhead,
                         dim_feedforward=dim_feedforward, dropout=dropout, activation=activation)
