import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint

import numpy as np
import pandas as pd
from math import pi, log, e
import matplotlib.pyplot as plt

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

    def log_prediction_table(self,
                             t_train, y_clean_train, y_train, y_pred, y_train_inference,
                             t_test, y_clean_test, y_test, y_test_inference):
        train_log_table = pd.DataFrame(np.concatenate([t_train.reshape(-1, 1),
                                                       y_clean_train.T,
                                                       y_train.T,
                                                       y_pred.T,
                                                       y_train_inference.T], axis=1),
                                       columns=['t'] \
                                               + ['y_true_clean_y%i' % i for i in self.visible_dims] \
                                               + ['y_true_noisy_y%i' % i for i in self.visible_dims] \
                                               + ['y_pred_shooting_y%i' % i for i in self.visible_dims] \
                                               + ['y_pred_inference_y%i' % i for i in self.visible_dims])

        train_log_table = pd.melt(train_log_table, id_vars=['t'], value_name='y', var_name='description')
        train_log_table['stage'] = 'train'

        test_log_table = pd.DataFrame(np.concatenate([t_test.reshape(-1, 1),
                                                      y_clean_test.T,
                                                      y_test.T,
                                                      y_test_inference.T], axis=1),
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
                                   values='y', index=['t', 'stage', 'type', 'subtype'],
                                   columns=['variable']).reset_index()

        log_table = wandb.Table(dataframe=log_table)
        return log_table

    def log_prediction_image(self,
                             y_clean_train, y_train, y_pred, y_train_inference,
                             y_clean_test, y_test, y_test_inference):

        train_len, test_len = y_train.shape[-1], y_test.shape[-1]

        dim = y_train.shape[0]

        log_image = np.zeros([dim * 4, train_len + test_len, 1])
        log_image[:, :train_len, 0] = np.concatenate([y_clean_train,
                                                      y_train,
                                                      y_pred,
                                                      y_train_inference])
        log_image[:, train_len:, 0] = np.concatenate([y_clean_test,
                                                      y_test,
                                                      np.zeros_like(y_test),
                                                      y_test_inference])
        log_image = wandb.Image(log_image)
        return log_image

    def log_prediction_gif(self,
                           y_clean_train, y_train, y_pred, y_train_inference,
                           y_clean_test, y_test, y_test_inference):
        train_len, test_len = y_train.shape[-1], y_test.shape[-1]
        max_len = max([train_len, test_len])

        height, width = self.init_dim

        log_video = np.zeros([height * 4, width * 2, max_len])
        log_video[:, :width, :train_len] = np.concatenate([y_clean_train.reshape([*self.init_dim, -1]),
                                                           y_train.reshape([*self.init_dim, -1]),
                                                           y_pred.reshape([*self.init_dim, -1]),
                                                           y_train_inference.reshape([*self.init_dim, -1])])
        log_video[:, width:, :test_len] = np.concatenate([y_clean_test.reshape([*self.init_dim, -1]),
                                                          y_test.reshape([*self.init_dim, -1]),
                                                          np.zeros_like(y_test).reshape([*self.init_dim, -1]),
                                                          y_test_inference.reshape([*self.init_dim, -1])])

        log_video = log_video[:, :, :, None].transpose([2, 3, 0, 1])
        log_video = ((log_video - log_video.min()) / (log_video.max() - log_video.min()) * 255.).astype('uint8')
        log_video = wandb.Video(log_video)

        return log_video

    def log_spectrum(self, model):
        eigv = np.linalg.eigvals(model.rhs.linear.weight.detach().cpu().numpy())

        spectrum_table = wandb.Table(data=[[x, y] for (x, y) in zip(eigv.real, eigv.imag)], columns=["Re", "Im"])

        return spectrum_table

    def log_latent_trajectories(self, t_train, z_pred, t_test, z_train_inference, z_test_inference):
        points_per_shooting_var, n_shooting_vars, latent_dim = z_pred.shape
        points_per_shooting_var -= 1

        last_point = z_pred[-1:, -1, :]  # (1, latent_dim)
        z_pred = np.concatenate([z_pred[:-1, :, :].transpose([1, 0, 2]).reshape(-1, latent_dim), last_point],
                                axis=0).T  # (latent_dim, T)

        plt.rcParams.update({'font.size': 22})

        shooting_fig = plt.figure(figsize=(20, 10), num=1, clear=True)
        for z in z_pred:
            plt.plot(t_train, z)
            plt.scatter(t_train[:-1:points_per_shooting_var], z[:-1:points_per_shooting_var])
        plt.xlabel('$t$')
        plt.ylabel('$z_i(t)$')
        plt.title('Latent shooting trajectories')
        plt.grid()
        shooting_image = wandb.Image(shooting_fig)

        inference_fig = plt.figure(figsize=(20, 10), num=1, clear=True)
        for z_train, z_test in zip(z_train_inference, z_test_inference):
            plt.plot(t_train, z_train)
            plt.plot(t_test, z_test)
        plt.xlabel('$t$')
        plt.ylabel('$z_i(t)$')
        plt.title('Latent inference trajectories')
        plt.grid()
        inference_image = wandb.Image(inference_fig)

        return shooting_image, inference_image

    def log_prediction_results(self, model, t_train, y_clean_train, y_train, z_pred, y_pred, t_test, y_clean_test, y_test):
        y_inference, z_inference = model.inference(torch.cat([t_train, t_test]), y_train)  # (signal_dim, T), (latent_dim, T)
        y_train_inference = y_inference[:, :t_train.shape[0]]
        y_test_inference = y_inference[:, t_train.shape[0]:]
        z_train_inference = z_inference[:, :t_train.shape[0]]
        z_test_inference = z_inference[:, t_train.shape[0]:]

        t_train, y_clean_train, y_train, z_pred, y_pred, y_train_inference, t_test, y_clean_test, y_test, \
            y_test_inference, z_train_inference, z_test_inference = \
                t_train.cpu().numpy(), y_clean_train.cpu().numpy(), y_train.cpu().numpy(),\
                z_pred.detach().cpu().numpy(), y_pred.detach().cpu().numpy(), \
                y_train_inference.detach().cpu().numpy(), t_test.cpu().numpy(),\
                y_clean_test.cpu().numpy(), y_test.cpu().numpy(),\
                y_test_inference.detach().cpu().numpy(),\
                z_train_inference.detach().cpu().numpy(), z_test_inference.detach().cpu().numpy()

        signals = \
            {'true': {'t_train': t_train,
                      't_test': t_test,
                      'y_clean_train': y_clean_train,
                      'y_train': y_train,
                      'y_clean_test': y_clean_test,
                      'y_test': y_test},
             'pred': {'y_train_pred': y_pred,
                      'z_train_pred': z_pred,
                      'y_train_inference': y_train_inference,
                      'y_test_inference': y_test_inference,
                      'z_train_inference': z_train_inference,
                      'z_test_inference': z_test_inference}}

        if len(self.visible_dims) <= 3:
            log_table = self.log_prediction_table(t_train, y_clean_train, y_train, y_pred, y_train_inference,
                                                  t_test, y_clean_test, y_test, y_test_inference)
        else:
            log_table = None

        if self.signal_dim > 3:
            if self.init_dim is None:
                log_video = None
                log_image = self.log_prediction_image(y_clean_train, y_train, y_pred, y_train_inference,
                                                      y_clean_test, y_test, y_test_inference)
            else:
                log_video = self.log_prediction_gif(y_clean_train, y_train, y_pred, y_train_inference,
                                                    y_clean_test, y_test, y_test_inference)
                log_image = None
        else:
            log_video = None

        spectrum_table = self.log_spectrum(model)
        shooting_latent_trajectories, inference_latent_trajectories = \
            self.log_latent_trajectories(t_train, z_pred, t_test, z_train_inference, z_test_inference)

        return log_table, log_video, log_image, spectrum_table,\
               shooting_latent_trajectories, inference_latent_trajectories, signals


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
        t = torch.linspace(self.t0, self.T, self.n_points)
        # t = torch.logspace(0, log(self.T + 1), self.n_points, base=e) - 1
        y0 = torch.tensor([1., 0.]).view(1, -1) * self.signal_amp
        y_clean = odeint(self.rhs, y0, t)[:, 0, self.visible_dims].permute(1, 0) # (#visible_dims, T)
        y = self.generate_visible_trajectory(y_clean)

        # t = torch.log(t + 1) / log(self.T + 1) * self.T

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


class PendulumTrajectory(Trajectory):
    def __init__(self, visible_dims=(0, 1), T=100, n_points=802, noise_std=0., m=5., l=10., lambd=0.05):
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

        y_clean = odeint(self.rhs, y0, t)[:, 0, self.visible_dims].permute(1, 0)  # (#visible_dims, T)
        y = self.generate_visible_trajectory(y_clean)

        return t, y_clean, y


class FluidFlowTrajectory(Trajectory):
    def __init__(self, visible_dims=(0, 1, 2), T=50, n_points=802, noise_std=0., mu=0.1, omega=1, A=-0.1, lam=10):
        super().__init__(t0=0, T=T, n_points=n_points, noise_std=noise_std, signal_amp=1)

        self.mu = mu
        self.omega = omega
        self.A = A
        self.lam = lam

        self.rhs = self.RHS(mu=self.mu, omega=self.omega, A=self.A, lam=self.lam)

        self.signal_dim = 2
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

        y_clean = odeint(self.rhs, y0, t)[:, 0, self.visible_dims].permute(1, 0)  # (#visible_dims, T)
        y = self.generate_visible_trajectory(y_clean)

        return t, y_clean, y


class KarmanVortexStreet(Trajectory):
    def __init__(self, noise_std=0.):
        super().__init__(t0=0, T=100., n_points=98, noise_std=noise_std, signal_amp=1)

        self.data = torch.tensor(np.load('karman_snapshots.npz')['snapshots'], dtype=torch.float32)[:, :, :-3]
        self.init_dim = self.data.shape[:2]

        self.signal_dim = self.init_dim[0] * self.init_dim[1]
        self.visible_dims = list(range(self.signal_dim))

    def __call__(self):
        t = torch.linspace(self.t0, self.T, self.n_points)
        y_clean = self.data.reshape(self.signal_dim, -1)
        y = self.generate_visible_trajectory(y_clean)

        return t, y_clean, y


class ToyDataset(Trajectory):
    def __init__(self, T=4 * np.pi, n_points=402, noise_std=0.):
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
