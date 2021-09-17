import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

import wandb

class DataGenerator():
  def __init__(self, trajectory_len, batch_size,
               rand_p, rand_q, rand_max_amp, rand_noise_amp, **kwargs):
    super().__init__()
    self.trajectory_len = trajectory_len
    self.batch_size = batch_size

    self.rand_p = rand_p
    self.rand_q = rand_q
    self.rand_max_amp = rand_max_amp
    self.rand_noise_amp = rand_noise_amp

    self.dim = None

  def generate_signal_batch(self, **kwargs):
    raise NotImplementedError()

  def generate_random_signal_batch(self):
    a = ((torch.rand((self.batch_size * self.dim, self.rand_p)) * 2) - 1) * self.rand_max_amp
    b = ((torch.rand((self.batch_size * self.dim, self.rand_p)) * 2) - 1) * self.rand_max_amp
    c = ((torch.rand((self.batch_size * self.dim, self.rand_q)) * 2) - 1) * self.rand_max_amp

    t = torch.linspace(0, 2 * 3.14, self.trajectory_len)
    t_inc = t * torch.arange(self.rand_p).view(-1, 1)
    t_pow = t ** torch.arange(self.rand_q).view(-1, 1)

    y = a @ torch.sin(t_inc) + b @ torch.cos(t_inc) + c @ t_pow
    y_noise = y + torch.randn_like(y) * self.rand_noise_amp
    return y.view(self.batch_size, self.dim, -1), y_noise.view(self.batch_size, self.dim, -1)

  def forward(self):
    return self.generate_random_signal_batch()

  def log_reconstruction_results(self, rand_y, rand_y_noise, rand_y_rec):
    n_samples = 3

    log_table = np.stack([rand_y.detach().cpu()[:n_samples],
                         rand_y_noise.detach().cpu()[:n_samples],
                         rand_y_rec.detach().cpu()[:n_samples]]).transpose(2, 0, 1, 3) # (n_dim, 3, n_samples, T)

    T = log_table.shape[-1]

    t = np.tile(np.arange(T, dtype=np.int).reshape(1, 1, 1, -1), (1, 3, n_samples, 1))
    signal_types = np.tile(np.arange(3, dtype=np.int).reshape(1, -1, 1, 1), (1, 1, n_samples, T))
    sample_ids = np.tile(np.arange(n_samples, dtype=np.int).reshape(1, 1, -1, 1), (1, 3, 1, T))

    log_table = np.concatenate([signal_types, sample_ids, t, log_table], axis=0) # (n_dim + 3, 3, n_samples, T)
    log_table = log_table.transpose(1, 2, 3, 0).reshape(-1, self.dim+3)

    n_rows = log_table.shape[0]

    dim_labels = ['y'] if self.dim == 1 else ['x', 'y', 'z']
    log_table = pd.DataFrame(log_table, columns=['signal_type', 'sample_id', 't'] + dim_labels)
    log_table['signal_type'] = log_table['signal_type'].map({0: 'Signal', 1: 'Signal+noise', 2: 'Reconstructed'})
    log_table = wandb.Table(dataframe=log_table)

    return log_table

  def log_approximation_results(self, model, batch_t, batch_y):
    n_samples = 3

    batch_t = torch.linspace(0, 2 * batch_t.max(), 2 * batch_t.shape[0], device=batch_t.device)
    pred_y = model(batch_y, batch_t).detach().cpu()
    batch_y = batch_y.detach().cpu()

    T = batch_t.shape[0]

    log_table = np.stack([batch_y[:n_samples],
                          pred_y[:n_samples][:, :, :T//2],
                          pred_y[:n_samples][:, :, T//2:]], axis=0).transpose(2, 0, 1, 3) # (n_dim, 3, n_samples, T)

    T = T // 2

    t = np.tile(np.arange(T, dtype=np.int).reshape(1, 1, 1, -1), (1, 3, n_samples, 1))
    t[:, -1, :, :] =  t[:, -1, :, :] + T
    signal_types = np.tile(np.arange(3, dtype=np.int).reshape(1, -1, 1, 1), (1, 1, n_samples, T))
    sample_ids = np.tile(np.arange(n_samples, dtype=np.int).reshape(1, 1, -1, 1), (1, 3, 1, T))


    log_table = np.concatenate([signal_types, sample_ids, t, log_table], axis=0) # (n_dim + 3, 3, n_samples, T)
    log_table = log_table.transpose(1, 2, 3, 0).reshape(-1, self.dim+3)

    n_rows = log_table.shape[0]

    dim_labels = ['y'] if self.dim == 1 else ['x', 'y', 'z']
    log_table = pd.DataFrame(log_table, columns=['signal_type', 'sample_id', 't'] + dim_labels)
    log_table['signal_type'] = log_table['signal_type'].map({0: 'Signal', 1: 'Approximated', 2: 'Extrapolated'})
    log_table = wandb.Table(dataframe=log_table)

    return log_table


class SinDataGenerator(DataGenerator):
  def __init__(self,
               trajectory_len=100, batch_size=32,
               rand_p=3, rand_q=1, rand_max_amp=1, rand_noise_amp=0.1,
               signal_t_min=0, signal_t_max=100, signal_noise_amp=0.1, signal_max_amp=1, **kwargs):
    
    super().__init__(trajectory_len=trajectory_len, batch_size=batch_size,
                     rand_p=rand_p, rand_q=rand_q, rand_max_amp=rand_max_amp, rand_noise_amp=rand_noise_amp)

    self.signal_t_min = signal_t_min
    self.signal_t_max = signal_t_max
    self.signal_noise_amp = signal_noise_amp
    self.signal_max_amp = signal_max_amp

    self.dim = 1

  def generate_signal_batch(self):
    t0 = torch.rand(self.batch_size) * 2 * 3.14
    amp = torch.rand(self.batch_size) * self.signal_max_amp
    t = torch.linspace(self.signal_t_min, self.signal_t_max, self.trajectory_len)
    y = torch.stack([amp_ * torch.sin(t + t0_).T for t0_, amp_ in zip(t0, amp)], dim=0).view(self.batch_size, 1, -1)
    y += torch.rand_like(y) * self.signal_noise_amp

    return t, y


class LorenzRHS(nn.Module):
    def __init__(self, sigma, rho, beta):
      super().__init__()
      self.sigma = sigma
      self.rho = rho
      self.beta = beta

    def forward(self, t, state):
      x, y, z = state.transpose(1, 0) 
      return torch.vstack([self.sigma * (y - x), x * (self.rho - z) - y, x * y - self.beta * z]).transpose(1, 0)


class LorenzDataGenerator(DataGenerator):
  def __init__(self,
               trajectory_len=1000, batch_size=32,
               rand_p=3, rand_q=1, rand_max_amp=1, rand_noise_amp=0.,
               signal_t_min=0, signal_t_max=5, signal_noise_amp=0.,
               signal_max_amp=10, sigma=10.0, rho=28.0, beta=8.0/3.0, **kwargs):
    
    super().__init__(trajectory_len=trajectory_len, batch_size=batch_size,
                     rand_p=rand_p, rand_q=rand_q, rand_max_amp=rand_max_amp, rand_noise_amp=rand_noise_amp)

    self.signal_t_min = signal_t_min
    self.signal_t_max = signal_t_max
    self.signal_noise_amp = signal_noise_amp
    self.signal_max_amp = signal_max_amp
        
    self.rho = rho
    self.rhs = LorenzRHS(sigma, rho, beta)

    self.dim = 3

  def generate_signal_batch(self):
    s0 = (torch.rand((self.batch_size, 3)) - 0.5) * 2 * self.signal_max_amp + torch.tensor([0., 0., self.rho]).view(1, 3)
    t = torch.linspace(self.signal_t_min, self.signal_t_max, self.trajectory_len)

    y = (odeint(self.rhs, s0, t).permute(1, 2, 0) - torch.tensor([0., 0., self.rho]).view(1, 3, 1)) / 10
    y += torch.rand_like(y) * self.signal_noise_amp
    return t, y
