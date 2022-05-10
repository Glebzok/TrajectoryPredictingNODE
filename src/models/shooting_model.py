import torch
import torch.nn as nn


class DeterministicShooting(nn.Module):
    def __init__(self, rhs_net, odeint):
        super(DeterministicShooting, self).__init__()
        self.rhs_net = rhs_net
        self.odeint = odeint

    def forward(self, z0, t):
        # z0: n_shooting_vars x latent_dim
        # t:  T
        signal_length = t.shape[0]
        n_shooting_vars = z0.shape[0]
        subsignal_length = (signal_length - 1) // n_shooting_vars + 1
        z = self.odeint(self.rhs_net, z0, t[:subsignal_length])  # (t, n_shooting_vars, latent_dim)
        z = z.permute(1, 2, 0)[None, ...]  # (1, n_shooting_vars, latent_dim, t)
        return z

    def inference(self, z0, t):
        # z0: n_shooting_vars x latent_dim
        # t:  T
        return self.forward(z0[:1, :], t)[0, 0]  # (latent_dim, T)


class VariationalShooting(DeterministicShooting):
    def __init__(self, rhs_net, odeint, n_samples):
        super(VariationalShooting, self).__init__(rhs_net=rhs_net, odeint=odeint)
        self.n_samples = n_samples

    def forward(self, z0, t):
        # z0: n_shooting_vars x 2*latent_dim
        # t:  T
        signal_length = t.shape[0]
        n_shooting_vars = z0.shape[0]
        latent_dim = z0.shape[1] // 2

        z0_mu = z0[:, :latent_dim]
        z0_sigma = z0[:, latent_dim:]

        z0 = z0_mu[None, ...] \
             + torch.randn(self.n_samples, n_shooting_vars, latent_dim, device=z0.device) \
             * z0_sigma[None, ...]  # (n_samples, n_shooting_vars, latent_dim)

        z0 = z0.reshape(-1, latent_dim)  # (n_samples * n_shooting_vars, latent_dim)
        subsignal_length = (signal_length - 1) // n_shooting_vars + 1

        z = self.odeint(self.rhs_net, z0, t[:subsignal_length])  # (t, n_samples * n_shooting_vars, latent_dim)
        z = z.view(-1, self.n_samples, n_shooting_vars, latent_dim)  # (t, n_samples,  n_shooting_vars, latent_dim)
        z = z.permute(1, 2, 3, 0)  # (n_samples, n_shooting_vars, latent_dim, t)
        return z

    def inference(self, z0, t):
        # z0: n_shooting_vars x 2*latent_dim
        # t:  T
        latent_dim = z0.shape[1] // 2

        z0_mu = z0[:, :latent_dim]
        z = self.odeint(self.rhs_net, z0_mu[:1, :], t)[:, 0, :].T  # (latent_dim, T)
        return z
