import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

  def plot_reconstruction_results(self, rand_y, rand_y_noise, rand_y_rec):
    fig, ax = plt.subplots(1, self.dim, figsize=(8*self.dim, 5), squeeze=False)

    clrs = plt.cm.get_cmap('prism', 7)

    for signal_dim in range(self.dim):
      for num, true, true_noise, pred in zip(range(rand_y.shape[0])[:3], rand_y.detach().cpu()[:3, signal_dim, :], rand_y_noise.detach().cpu()[:3, signal_dim, :], rand_y_rec.detach().cpu()[:3][:3, signal_dim, :]):
        ax[0][signal_dim].plot(true, ls='', c=clrs(num), label='signal')
        ax[0][signal_dim].plot(true_noise, ls=':', c=clrs(num), label='signal + noise')
        ax[0][signal_dim].plot(pred, ls='--', c=clrs(num), label='reconstructed')

    return fig

  def plot_approximation_results(self, model, batch_t, batch_y):
    raise NotImplementedError()


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

  def plot_approximation_results(self, model, batch_t, batch_y):
    fig = plt.figure(figsize=(8, 5))

    clrs = plt.cm.get_cmap('prism', 7)

    batch_t = torch.linspace(0, 2 * batch_t.max(), 2 * batch_t.shape[0], device=batch_t.device)
    pred_y = model(batch_y, batch_t).detach().cpu()

    batch_y = batch_y.detach().cpu()
    
    for num in range(3):
      plt.plot(batch_y[num][0], ls='-', c=clrs(num), label='signal')
      plt.plot(pred_y[num][0], c=clrs(num), ls='--', label='predicted')

    return fig

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
        
    self.rhs = LorenzRHS(sigma, rho, beta)

    self.dim = 3

  def generate_signal_batch(self):
    s0 = (torch.rand((self.batch_size, 3)) - 0.5) * 2 * self.signal_max_amp
    t = torch.linspace(self.signal_t_min, self.signal_t_max, self.trajectory_len)

    y = odeint(self.rhs, s0, t).permute(1, 2, 0)
    y += torch.rand_like(y) * self.signal_noise_amp
    return t, y

  def plot_approximation_results(self, model, batch_t, batch_y):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca(projection="3d")

    clrs = plt.cm.get_cmap('prism', 7)

    batch_t = torch.linspace(0, 2 * batch_t.max(), 2 * batch_t.shape[0], device=batch_t.device)
    pred_y = model(batch_y, batch_t).detach().cpu()

    batch_y = batch_y.detach().cpu()
    
    for num in range(3):
      ax.plot(batch_y[num, 0, :], batch_y[num, 1, :], batch_y[num, 2, :], ls='-', c=clrs(num), label='signal', alpha=0.5)
      ax.plot(pred_y[num, 0, :], pred_y[num, 1, :], pred_y[num, 2, :], ls='--', c=clrs(num), label='predicted', alpha=0.5)

    return fig