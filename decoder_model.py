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
    def __init__(self, latent_dim, signal_dim, min_channels, n_layers, act='ReLU'):
        super().__init__(input_dim=latent_dim, output_dim=signal_dim,
                         min_channels=min_channels, n_layers=n_layers, act=act)

    def forward(self, z):
        # print(z.shape)
        # z (t, n_shooting_vars, latent_dim)
        # t, bs, latent_dim = z.shape
        # pred_z_flattend = torch.cat([z[:-1, :, :].permute(1, 0, 2).reshape(-1, latent_dim),
        #                              z[-1:, -1, :]], dim=0)  # (T, latent_dim)
        # print(pred_z_flattend.shape)

        # flattend_pred_y = super().forward(pred_z_flattend.T[None, :, :])[0].T # (T, signal_dim)
        # print(flattend_pred_y.shape)
        # signal_dim = flattend_pred_y.shape[-1]

        # pred_y = torch.zeros((t, bs, signal_dim), dtype=torch.float32, device=z.device) # (t, n_shooting_vars, signal_dim)
        # print(pred_y.shape)
        # pred_y[:-1, :, :] += flattend_pred_y[:-1, :].view(bs, t - 1, signal_dim).transpose(0, 1)
        # pred_y[-1:, -1, :] += flattend_pred_y[-1:, :]

        # z (T, latent_dim)
        z = z.T[None, :, :] # (1, latent_dim, T)
        y = super().forward(z).permute(0, 2, 1)[0] # (1, signal_dim, T)
        return y

        # return pred_y


class Seq2SeqTransformerLatentSpaceDecoder(Seq2SeqTransformerNet):
    def __init__(self, latent_dim, signal_dim, n_layers, nhead, dim_feedforward, dropout, activation):
        super().__init__(input_dim=latent_dim, output_dim=signal_dim, n_layers=n_layers, nhead=nhead,
                         dim_feedforward=dim_feedforward, dropout=dropout, activation=activation)
