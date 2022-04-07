from seq2seq_nets import SingleLayerConvNet, ConvNet, UNetLikeConvNet, Seq2SeqTransformerNet


class SimpleLatentSpaceEncoder(SingleLayerConvNet):
    def __init__(self, signal_dim, latent_dim):
        super().__init__(input_dim=signal_dim, output_dim=latent_dim)


class ConvLatentSpaceEncoder(ConvNet):
    def __init__(self, latent_dim, signal_dim, n_layers, hidden_channels):
        super().__init__(input_dim=signal_dim, output_dim=latent_dim,
                         n_layers=n_layers, hidden_channels=hidden_channels)


class UNetLikeLatentSpaceEncoder(UNetLikeConvNet):
    def __init__(self, latent_dim, signal_dim, min_channels, n_layers, act='ReLU'):
        super().__init__(input_dim=signal_dim, output_dim=latent_dim,
                         min_channels=min_channels, n_layers=n_layers, act=act)


class TransformerLatentSpaceEncoder(Seq2SeqTransformerNet):
    def __init__(self, latent_dim, signal_dim, n_layers, nhead, dim_feedforward, dropout, activation):
        super().__init__(input_dim=signal_dim, output_dim=latent_dim, n_layers=n_layers, nhead=nhead,
                         dim_feedforward=dim_feedforward, dropout=dropout, activation=activation)

# class RoFormerLatentSpaceEncoder(RoFormerNet):
#     def __init__(self, latent_dim, signal_dim, n_layers, nhead, dim_feedforward, dropout, activation):
#         super().__init__(input_dim=signal_dim, output_dim=latent_dim, n_layers=n_layers, nhead=nhead,
#                          dim_feedforward=dim_feedforward, dropout=dropout, activation=activation)
