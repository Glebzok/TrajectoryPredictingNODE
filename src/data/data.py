import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint as odeint

import numpy as np

import pickle as pkl
import imageio as iio

from src.data.abstract_data import AbstractTrajectory


class SinTrajectory(AbstractTrajectory):
    def __init__(self, noise_std, t0, T, n_points, signal_amp):
        super().__init__(t0=t0, T=T, n_points=n_points, noise_std=noise_std, signal_amp=signal_amp)

        self.signal_dim = 1
        self.visible_dims = [0]

    def __call__(self):
        t = torch.linspace(self.t0, self.T, self.n_points)
        y_clean = self.signal_amp * torch.sin(t).view(1, -1)
        y = self.generate_visible_trajectory(y_clean)

        return t, y_clean, y


class SpiralTrajectory(AbstractTrajectory):
    class RHS(nn.Module):
        def __init__(self):
            super().__init__()
            self.A = torch.tensor([[-0.1, 2.0],
                                   [-2.0, -0.1]], dtype=torch.float32)

        def forward(self, t, x):
            return (self.A @ (x.transpose(1, 0) ** 3)).transpose(1, 0)

    def __init__(self, noise_std, visible_dims, t0, T, n_points, signal_amp):
        super().__init__(t0=t0, T=T, n_points=n_points, noise_std=noise_std, signal_amp=signal_amp)

        self.signal_dim = 2
        self.visible_dims = list(visible_dims)

        self.rhs = self.RHS()

    def __call__(self):
        t = torch.linspace(self.t0, self.T, self.n_points)
        y0 = torch.tensor([1., 0.]).view(1, -1) * self.signal_amp
        y_clean = odeint(self.rhs, y0, t)[:, 0, self.visible_dims].permute(1, 0)
        y = self.generate_visible_trajectory(y_clean)

        return t, y_clean, y


class LorenzTrajectory(AbstractTrajectory):
    class RHS(nn.Module):
        def __init__(self, sigma, rho, beta):
            super().__init__()
            self.sigma = sigma
            self.rho = rho
            self.beta = beta

        def forward(self, t, state):
            x, y, z = state.transpose(1, 0)
            return torch.vstack([self.sigma * (y - x), x * (self.rho - z) - y, x * y - self.beta * z]).transpose(1, 0)

    def __init__(self, noise_std, visible_dims, t0, T, n_points, signal_amp, sigma,
                 rho, beta):
        super().__init__(t0=t0, T=T, n_points=n_points, noise_std=noise_std, signal_amp=signal_amp)

        self.signal_dim = 3
        self.visible_dims = list(visible_dims)

        self.rhs = self.RHS(sigma, rho, beta)

    def __call__(self):
        t = torch.linspace(self.t0, self.T, self.n_points)
        y0 = self.signal_amp * torch.tensor([1., 1., 1.]).view(1, 3)

        y_clean = odeint(self.rhs, y0, t)[:, 0, self.visible_dims].permute(1, 0)
        y = self.generate_visible_trajectory(y_clean)

        return t, y_clean, y


class CascadedTanksTrajectory(AbstractTrajectory):
    def __init__(self, noise_std, data_path):
        super().__init__(t0=0, T=12, n_points=802, noise_std=noise_std, signal_amp=10)

        self.data_path = data_path
        self.signal_dim = 2
        self.visible_dims = list([0, 1])

    def __call__(self):
        with open(self.data_path, 'rb') as f:
            data = pkl.load(f)

        y_clean, t = torch.tensor(data['y'], dtype=torch.float32), torch.tensor(data['t'], dtype=torch.float32)
        y = self.generate_visible_trajectory(y_clean)

        return t, y_clean, y


class PendulumTrajectory(AbstractTrajectory):
    def __init__(self, visible_dims, T, n_points, noise_std, m, l, lambd):
        super().__init__(t0=0, T=T, n_points=n_points, noise_std=noise_std, signal_amp=1)
        self.m = m
        self.l = l
        self.lambd = lambd

        self.rhs = self.RHS(m, l, lambd)

        self.signal_dim = 2
        self.visible_dims = list(visible_dims)

    class RHS(nn.Module):
        def __init__(self, m, l, lambd):
            super().__init__()

            self.l = l
            self.lambd = lambd
            self.m = m

            self.A = torch.tensor([[0., 1.],
                                   [-10. / l, -lambd / m]], dtype=torch.float32)

        def forward(self, t, x):
            res = torch.tensor([[x[:, 1],
                                 - 10. / self.l * torch.sin(x[:, 0]) - self.lambd / self.m * x[:, 1]]],
                               dtype=torch.float32)
            return res

    def __call__(self):
        t = torch.linspace(self.t0, self.T, self.n_points)
        y0 = torch.tensor([3.12, 0.]).view(1, -1)

        y_clean = odeint(self.rhs, y0, t)[:, 0, self.visible_dims].permute(1, 0)
        y = self.generate_visible_trajectory(y_clean)

        return t, y_clean, y


class FluidFlowTrajectory(AbstractTrajectory):
    def __init__(self, visible_dims, T, n_points, noise_std, mu, omega, A, lam):
        super().__init__(t0=0, T=T, n_points=n_points, noise_std=noise_std, signal_amp=1)

        self.mu = mu
        self.omega = omega
        self.A = A
        self.lam = lam

        self.rhs = self.RHS(mu=self.mu, omega=self.omega, A=self.A, lam=self.lam)

        self.signal_dim = 3
        self.visible_dims = list(visible_dims)

    class RHS(nn.Module):
        def __init__(self, mu, omega, A, lam):
            super().__init__()

            self.mu = mu
            self.omega = omega
            self.A = A
            self.lam = lam

        def forward(self, t, x):
            res = torch.tensor([[self.mu * x[:, 0] - self.omega * x[:, 1] + self.A * x[:, 0] * x[:, 2],
                                 self.omega * x[:, 0] + self.mu * x[:, 1] + self.A * x[:, 1] * x[:, 2],
                                 - self.lam * (x[:, 2] - x[:, 0] ** 2 - x[:, 1] ** 2)]], dtype=torch.float32)
            return res

    def __call__(self):
        t = torch.linspace(self.t0, self.T, self.n_points)
        y0 = torch.tensor([-.1, -.2, .05]).view(1, -1)

        y_clean = odeint(self.rhs, y0, t)[:, 0, self.visible_dims].permute(1, 0)
        y = self.generate_visible_trajectory(y_clean)

        return t, y_clean, y


class ShortKarmanVortexStreetTrajectory(AbstractTrajectory):
    def __init__(self, n_points, noise_std, data_path):
        super().__init__(t0=0, T=10., n_points=n_points, noise_std=noise_std, signal_amp=1)
        self.data = F.interpolate(torch.tensor(np.load(data_path)['snapshots'], dtype=torch.float32),
                                  n_points, mode='linear', align_corners=False)

        self.init_dim = self.data.shape[:2]
        self.signal_dim = self.init_dim[0] * self.init_dim[1]
        self.visible_dims = list(range(self.signal_dim))

    def __call__(self):
        t = torch.linspace(self.t0, self.T, self.n_points)
        y_clean = self.data.reshape(self.signal_dim, -1)
        y = self.generate_visible_trajectory(y_clean)

        return t, y_clean, y


class LongKarmanVortexStreetTrajectory(AbstractTrajectory):
    def __init__(self, n_points, noise_std, data_path):
        super().__init__(t0=0, T=10., n_points=n_points, noise_std=noise_std, signal_amp=1)

        self.data = np.stack(list(iio.get_reader(data_path, mode='I')))
        self.data = np.dot(self.data[..., :3], [0.2989, 0.5870, 0.1140])
        self.data = ((self.data[:, :123, :] ** 2 + self.data[:, 123:246, :] ** 2) ** 0.5)
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.data = F.interpolate(self.data[None, None, :, :], [n_points, 40, 200], mode='trilinear',
                                  align_corners=False)[0, 0].permute(1, 2, 0)

        self.init_dim = self.data.shape[:2]
        self.signal_dim = self.init_dim[0] * self.init_dim[1]
        self.visible_dims = list(range(self.signal_dim))

    def __call__(self):
        t = torch.linspace(self.t0, self.T, self.n_points)
        y_clean = self.data.reshape(self.signal_dim, -1)
        y = self.generate_visible_trajectory(y_clean)

        return t, y_clean, y


class ToyTrajectory(AbstractTrajectory):
    def __init__(self, T, n_points, noise_std):
        super().__init__(t0=0, T=T, n_points=n_points, noise_std=noise_std, signal_amp=1)

        self.init_dim = None
        self.signal_dim = 128
        self.visible_dims = list(range(self.signal_dim))

    def __call__(self):
        f1 = lambda x, t: 1. / torch.cosh(x + 3) * torch.exp(2.3j * t)
        f2 = lambda x, t: 2. / torch.cosh(x) * torch.tanh(x) * torch.exp(2.8j * t)

        x = torch.linspace(-5, 5, self.signal_dim)
        t = torch.linspace(0, self.T, self.n_points)

        xgrid, tgrid = torch.meshgrid(x, t)

        y_clean = (f1(xgrid, tgrid) + f2(xgrid, tgrid)).real
        y = self.generate_visible_trajectory(y_clean)

        return t, y_clean, y
