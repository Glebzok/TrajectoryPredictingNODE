import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint

import numpy as np
import pandas as pd
from math import pi, log, e

import pickle as pkl

import wandb


class Trajectory():
    def __init__(self, t0, T, n_points, noise_std, signal_amp, **kwargs):
        self.t0 = t0
        self.T = T
        self.n_points = n_points
        self.noise_std = noise_std
        self.signal_amp = signal_amp

        self.signal_dim = None
        self.visible_dims = None

    def generate_visible_trajectory(self, y_clean):
        # y_clean (signal_dim, T)
        y = y_clean + torch.randn_like(y_clean) * self.noise_std
        return y

    def __call__(self):
        raise NotImplementedError()

    def log_prediction_results(self, model, t_train, y_clean_train, y_train, z_pred, y_pred, t_test, y_clean_test, y_test):
        y_inference, z_inference = model.inference(torch.cat([t_train, t_test]), y_train)  # (signal_dim, T), (latent_dim, T)
        y_train_inference = y_inference[:, :t_train.shape[0]]
        y_test_inference = y_inference[:, t_train.shape[0]:]
        z_train_inference = z_inference[:, :t_train.shape[0]]
        z_test_inference = z_inference[:, t_train.shape[0]:]

        t_train, y_clean_train, y_train, z_pred, y_pred, y_train_inference, t_test, y_clean_test, y_test, \
            y_test_inference, z_train_inference, z_test_inference = \
                t_train.cpu(), y_clean_train.cpu(), y_train.cpu(), z_pred.detach().cpu(), y_pred.detach().cpu(), \
                y_train_inference.detach().cpu(), t_test.cpu(), y_clean_test.cpu(), y_test.cpu(),\
                y_test_inference.detach().cpu(), z_train_inference.detach().cpu(), z_test_inference.detach().cpu()

        train_log_table = pd.DataFrame(np.concatenate([t_train.numpy().reshape(-1, 1),
                                                       y_clean_train.numpy().T,
                                                       y_train.numpy().T,
                                                       y_pred.numpy().T,
                                                       y_train_inference.numpy().T], axis=1),
                                       columns=['t'] \
                                               + ['y_true_clean_y%i' % i for i in self.visible_dims] \
                                               + ['y_true_noisy_y%i' % i for i in self.visible_dims] \
                                               + ['y_pred_shooting_y%i' % i for i in self.visible_dims] \
                                               + ['y_pred_inference_y%i' % i for i in self.visible_dims])

        train_log_table = pd.melt(train_log_table, id_vars=['t'], value_name='y', var_name='description')
        train_log_table['stage'] = 'train'

        test_log_table = pd.DataFrame(np.concatenate([t_test.numpy().reshape(-1, 1),
                                                      y_clean_test.numpy().T,
                                                      y_test.numpy().T,
                                                      y_test_inference.numpy().T], axis=1),
                                      columns=['t'] \
                                              + ['y_true_clean_y%i' % i for i in self.visible_dims] \
                                              + ['y_true_noisy_y%i' % i for i in self.visible_dims] \
                                              + ['y_pred_inference_y%i' % i for i in self.visible_dims])

        test_log_table = pd.melt(test_log_table, id_vars=['t'], value_name='y', var_name='description')
        test_log_table['stage'] = 'test'

        log_table = pd.concat([train_log_table, test_log_table])
        log_table['type'] = log_table['description'].str.split('_').str[1]
        log_table['subtype'] = log_table['description'].str.split('_').str[2]
        log_table['variable'] = log_table['description'].str.split('_').str[3]

        log_table = pd.pivot_table(log_table.drop(columns=['description']),
                                   values='y',index=['t', 'stage', 'type', 'subtype'],
                                   columns=['variable']).reset_index()

        log_table = wandb.Table(dataframe=log_table)

        signals = \
            {'true': {'t_train': t_train.numpy(),
                      't_test': t_test.numpy(),
                      'y_clean_train': y_clean_train.numpy(),
                      'y_train': y_train.numpy(),
                      'y_clean_test': y_clean_test.numpy(),
                      'y_test': y_test.numpy()},
             'pred': {'y_train_pred': y_pred.numpy(),
                      'z_train_pred': z_pred.numpy(),
                      'y_train_inference': y_train_inference.numpy(),
                      'y_test_inference': y_test_inference.numpy(),
                      'z_train_inference': z_train_inference.numpy(),
                      'z_test_inference': z_test_inference.numpy()}}

        return log_table, signals


class SinTrajectory(Trajectory):
    def __init__(self, noise_std, t0=0, T=8 * pi, n_points=800, signal_amp=1):
        super().__init__(t0=t0, T=T, n_points=n_points, noise_std=noise_std, signal_amp=signal_amp)

        self.signal_dim = 1
        self.visible_dims = [0]

    def __call__(self):
        t = torch.linspace(self.t0, self.T, self.n_points)
        y_clean = self.signal_amp * torch.sin(t).view(1, -1) # y_clean (signal_dim, T)
        y = self.generate_visible_trajectory(y_clean)

        return t, y_clean, y


class SpiralTrajectory(Trajectory):
    class RHS(nn.Module):
        def __init__(self):
            super().__init__()
            self.A = torch.tensor([[-0.1, 2.0],
                                   [-2.0, -0.1]], dtype=torch.float32)

        def forward(self, t, x):
            return (self.A @ (x.transpose(1, 0) ** 3)).transpose(1, 0)

    def __init__(self, noise_std, visible_dims=(0, 1), t0=0, T=150, n_points=5000, signal_amp=2):
        super().__init__(t0=t0, T=T, n_points=n_points, noise_std=noise_std, signal_amp=signal_amp)

        self.signal_dim = 2
        self.visible_dims = list(visible_dims)

        self.rhs = self.RHS()

    def __call__(self):
        # t = torch.linspace(self.t0, self.T, self.n_points)
        t = torch.logspace(0, log(self.T + 1), self.n_points, base=e) - 1
        y0 = torch.tensor([1., 0.]).view(1, -1) * self.signal_amp
        y_clean = odeint(self.rhs, y0, t)[:, 0, self.visible_dims].permute(1, 0) # (#visible_dims, T)
        y = self.generate_visible_trajectory(y_clean)

        t = torch.log(t + 1) / log(self.T + 1) * self.T

        return t, y_clean, y


class LorenzTrajectory(Trajectory):
    class RHS(nn.Module):
        def __init__(self, sigma, rho, beta):
            super().__init__()
            self.sigma = sigma
            self.rho = rho
            self.beta = beta

        def forward(self, t, state):
            x, y, z = state.transpose(1, 0)
            return torch.vstack([self.sigma * (y - x), x * (self.rho - z) - y, x * y - self.beta * z]).transpose(1, 0)

    def __init__(self, noise_std, visible_dims=(0, 1, 2), t0=0, T=50, n_points=4000, signal_amp=10, sigma=10.0,
                 rho=28.0, beta=8.0 / 3.0):
        super().__init__(t0=t0, T=T, n_points=n_points, noise_std=noise_std, signal_amp=signal_amp)

        self.signal_dim = 3
        self.visible_dims = list(visible_dims)

        self.rhs = self.RHS(sigma, rho, beta)

    def __call__(self):
        t = torch.linspace(self.t0, self.T, self.n_points)
        y0 = self.signal_amp * torch.tensor([1., 1., 1.]).view(1, 3)

        y_clean = odeint(self.rhs, y0, t)[:, 0, self.visible_dims].permute(1, 0) # (#visible_dims, T)
        y = self.generate_visible_trajectory(y_clean)

        return t, y_clean, y


class CascadedTanksTrajectory(Trajectory):
    def __init__(self):
        super().__init__(t0=0, T=12, n_points=802, noise_std=0, signal_amp=10)

        self.signal_dim = 2
        self.visible_dims = list([0, 1])

    def __call__(self):
        with open('./cascaded_tanks.pkl', 'rb') as f:
            data = pkl.load(f)

        y, t = torch.tensor(data['y'], dtype=torch.float32), torch.tensor(data['t'], dtype=torch.float32)

        return t, y, y
