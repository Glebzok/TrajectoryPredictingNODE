import torch
import torch.nn as nn


class NODEModel(nn.Module):
    def __init__(self, encoder_model, shooting_model, decoder_model):
        super().__init__()
        self.encoder_model = encoder_model
        self.shooting_model = shooting_model
        self.decoder_model = decoder_model

    def forward(self, t, y):
        # t: T
        # y : (signal_dim, T)

        z0 = self.encoder_model(y)  # (n_shooting_vars, latent_dim)

        z_pred = self.shooting_model(z0, t)  # (n_samples, n_shooting_vars, latent_dim, t)
        n_samples, latent_dim = z_pred.shape[0], z_pred.shape[2]

        z_pred_flattened = torch.cat([z_pred[:, :, :, :-1].permute(0, 1, 3, 2).reshape(n_samples, -1, latent_dim),
                                      z_pred[:, -1, :, -1:].permute(0, 2, 1)], dim=1)  # (n_samples, T, latent_dim)
        z_pred_flattened = z_pred_flattened.permute(0, 2, 1)  # (n_samples, latent_dim, T)

        y_pred = self.decoder_model(z_pred_flattened)  # (n_samples, signal_dim, T)

        return y_pred, z0, z_pred

    def inference(self, t, y):
        # t: T
        # y : (signal_dim, T)
        z0 = self.encoder_model(y)[:1]  # (1, latent_dim)
        z_pred = self.shooting_model.inference(z0, t)  # (latent_dim, T)
        y_pred = self.decoder_model(z_pred[None, ...])[0]  # (signal_dim, T)

        return y_pred, z0, z_pred
