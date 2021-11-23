import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint

from rhs_model import SimpleRHS, FCRHS
from decoder_model import SimpleLatentSpaceDecoder, FCLatentSpaceDecoder
from encoder_model import SimpleLatentSpaceEncoder


class SingleShooting(nn.Module):
    def __init__(self, signal_dim):
        super().__init__()
        self.signal_dim = signal_dim
        self.rhs = SimpleRHS(system_dim=signal_dim)

    def forward(self, t, y):
        # y : (signal_dim, T)
        pred_y = odeint(self.rhs, y[None, :, 0], t).to(y.device)  # (T, 1, signal_dim)
        pred_y = pred_y[:, 0, :].permute(1, 0)  # (signal_dim, T)
        return pred_y, None, None

    def inference(self, t, y):
        # y : (signal_dim, T)
        pred_y = odeint(self.rhs, y[None, :, 0], t).to(y.device)  # (T, 1, signal_dim)
        pred_y = pred_y[:, 0, :].permute(1, 0)  # (signal_dim, T)
        return pred_y


class LatentSingleShooting(nn.Module):
    def __init__(self, signal_dim, latent_dim):
        super().__init__()
        self.signal_dim = signal_dim
        # self.rhs = SimpleRHS(system_dim=latent_dim)
        self.rhs = FCRHS(system_dim=latent_dim, n_layers=5, hidden_dim=25)
        self.latent_dim = latent_dim
        self.z0 = nn.Parameter(torch.randn(1, latent_dim))
        # self.decoder = SimpleLatentSpaceDecoder(latent_dim=latent_dim, signal_dim=signal_dim)
        self.decoder = FCLatentSpaceDecoder(latent_dim=latent_dim, signal_dim=signal_dim, n_layers=5, hidden_dim=25)

    def forward(self, t, y):
        # y : (signal_dim, T)
        pred_z = odeint(self.rhs, self.z0, t).to(y.device)  # (T, 1, latent_dim)
        pred_y = self.decoder(pred_z)  # (T, 1, signal_dim)
        pred_y = pred_y[:, 0, :].permute(1, 0)  # (signal_dim, T)
        return pred_y, None, None

    def inference(self, t, y):
        # y : (signal_dim, T)
        pred_z = odeint(self.rhs, self.z0, t).to(y.device)  # (T, 1, latent_dim)
        pred_y = self.decoder(pred_z)  # (T, 1, signal_dim)
        pred_y = pred_y[:, 0, :].permute(1, 0)  # (signal_dim, T)
        return pred_y


class MultipleShooting(SingleShooting):
    def __init__(self, signal_dim, n_shooting_vars):
        super().__init__(signal_dim=signal_dim)
        self.n_shooting_vars = n_shooting_vars
        self.shooting_vars = None

    def forward(self, t, y):
        # y : (signal_dim, T)
        signal_dim, singal_length = y.shape

        if self.shooting_vars is None:
            shooting_times = torch.linspace(0, singal_length - 1, self.n_shooting_vars + 1,
                                            dtype=torch.long, device=y.device)[:-1]
            self.shooting_vars = nn.Parameter(torch.index_select(y, 1, index=shooting_times).permute(1, 0))  # (n_shooting_vars, signal_dim)

        subsignal_length = singal_length // self.n_shooting_vars
        pred_y = odeint(self.rhs, self.shooting_vars, t[:subsignal_length]).to(y.device)  # (t, n_shooting_vars, signal_dim)
        shooting_end_values = pred_y[-1, :, :]  # (n_shooting_vars, signal_dim)
        pred_y = pred_y.permute(1, 0, 2).reshape(-1, signal_dim)  # (T, signal_dim)
        pred_y = pred_y.permute(1, 0)  # (signal_dim, T)
        return pred_y, shooting_end_values[:-1, :], self.shooting_vars[1:, :]

    def inference(self, t, y):
        # y : (signal_dim, T)
        pred_y = odeint(self.rhs, self.shooting_vars[:1, :], t).to(y.device)  # (T, 1, signal_dim)
        pred_y = pred_y[:, 0, :].permute(1, 0)  # (signal_dim, T)
        return pred_y


class LatentMultipleShooting(LatentSingleShooting):
    def __init__(self, signal_dim, latent_dim, n_shooting_vars):
        super().__init__(signal_dim=signal_dim, latent_dim=latent_dim)
        self.n_shooting_vars = n_shooting_vars
        self.shooting_vars = None

    def forward(self, t, y):
        # y : (signal_dim, T)
        signal_dim, singal_length = y.shape

        if self.shooting_vars is None:
            self.shooting_vars = nn.Parameter(torch.rand(self.n_shooting_vars,
                                                         self.latent_dim,
                                                         device=y.device))  # (n_shooting_vars, latent_dim)
            # print(self.shooting_vars)

        subsignal_length = (singal_length - 1) // self.n_shooting_vars + 1
        pred_z = odeint(self.rhs, self.shooting_vars, t[:subsignal_length]).to(y.device)  # (t, n_shooting_vars, latent_dim)
        shooting_end_values = pred_z[-1, :, :]  # (n_shooting_vars, latent_dim)
        pred_y = self.decoder(pred_z)  # (t, n_shooting_vars, signal_dim)
        last_point = pred_y[-1:, -1, :]  # (1, signal_dim)
        pred_y = torch.cat([pred_y[:-1, :, :].permute(1, 0, 2).reshape(-1, signal_dim), last_point], dim=0)  # (T, signal_dim)
        pred_y = pred_y.permute(1, 0)  # (signal_dim, T)
        return pred_y, shooting_end_values[:-1, :], self.shooting_vars[1:, :]

    def inference(self, t, y):
        # y : (signal_dim, T)
        pred_z = odeint(self.rhs, self.shooting_vars[:1, :], t).to(y.device)  # (T, 1, latent_dim)
        pred_y = self.decoder(pred_z)  # (T, 1, signal_dim)
        pred_y = pred_y[:, 0, :].permute(1, 0)  # (signal_dim, T)
        return pred_y


class LatentDecoderMultipleShooting(LatentSingleShooting):
    def __init__(self, signal_dim, latent_dim, n_shooting_vars):
        super().__init__(signal_dim=signal_dim, latent_dim=latent_dim)
        self.n_shooting_vars = n_shooting_vars
        self.shooting_times = None
        self.encoder = SimpleLatentSpaceEncoder(signal_dim=signal_dim, latent_dim=latent_dim)

    def forward(self, t, y):
        # y : (signal_dim, T)
        signal_dim, singal_length = y.shape

        if self.shooting_times is None:
            self.shooting_times = torch.linspace(0, singal_length - 1, self.n_shooting_vars + 1, dtype=torch.long,
                                                 device=y.device)[:-1]

        subsignal_length = (singal_length - 1) // self.n_shooting_vars + 1
        z = self.encoder(y[None, :, :])[0]  # (latent_dim, T)
        # print(z.shape)
        z0 = torch.index_select(z, 1, index=self.shooting_times).permute(1, 0)  # (n_shooting_vars, latent_dim)
        pred_z = odeint(self.rhs, z0, t[:subsignal_length]).to(y.device)  # (t, n_shooting_vars, latent_dim)
        shooting_end_values = pred_z[-1, :, :]  # (n_shooting_vars, latent_dim)
        pred_y = self.decoder(pred_z)  # (t, n_shooting_vars, signal_dim)
        last_point = pred_y[-1:, -1, :]  # (1, signal_dim)
        pred_y = torch.cat([pred_y[:-1, :, :].permute(1, 0, 2).reshape(-1, signal_dim), last_point],
                           dim=0)  # (T, signal_dim)
        pred_y = pred_y.permute(1, 0)  # (signal_dim, T)
        return pred_y, shooting_end_values[:-1, :], z0[1:, :]

    def inference(self, t, y):
        # y : (signal_dim, T)
        z0 = self.encoder(y[None, :, :])[:, :, 0]  # (1, latent_dim)
        pred_z = odeint(self.rhs, z0, t).to(y.device)  # (T, 1, latent_dim)
        pred_y = self.decoder(pred_z)  # (T, 1, signal_dim)
        pred_y = pred_y[:, 0, :].permute(1, 0)  # (signal_dim, T)
        return pred_y