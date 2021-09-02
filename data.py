import torch

class DataGenerator():
  def __init__(self, trajectory_len=100, batch_size=32, signal_t_min=0, signal_t_max=100, signal_noise_amp=0.1, signal_max_amp=1, rand_p=1, rand_q=1, rand_max_amp=1, rand_noise_amp=0.1, **kwargs):
    super().__init__()
    self.trajectory_len = trajectory_len
    self.batch_size = batch_size

    self.signal_t_min = signal_t_min
    self.signal_t_max = signal_t_max
    self.signal_noise_amp = signal_noise_amp
    self.signal_max_amp = signal_max_amp

    self.rand_p = rand_p
    self.rand_q = rand_q
    self.rand_max_amp = rand_max_amp
    self.rand_noise_amp = rand_noise_amp

  def generate_signal_batch(self):
    t0 = torch.rand(self.batch_size) * 2 * 3.14
    amp = torch.rand(self.batch_size) * self.signal_max_amp
    t = torch.linspace(self.signal_t_min, self.signal_t_max, self.trajectory_len)
    y = torch.stack([amp_ * torch.sin(t + t0_).T for t0_, amp_ in zip(t0, amp)], dim=0)
    y += torch.rand_like(y) * self.signal_noise_amp

    return t, y

  def generate_random_signal_batch(self):
    a = ((torch.rand((self.batch_size, self.rand_p)) * 2) - 1) * self.rand_max_amp
    b = ((torch.rand((self.batch_size, self.rand_p)) * 2) - 1) * self.rand_max_amp
    c = ((torch.rand((self.batch_size, self.rand_q)) * 2) - 1) * self.rand_max_amp

    t = torch.linspace(0, 2 * 3.14, self.trajectory_len)
    t_inc = t * torch.arange(self.rand_p).view(-1, 1)
    t_pow = t ** torch.arange(self.rand_q).view(-1, 1)

    y = a @ torch.sin(t_inc) + b @ torch.cos(t_inc) + c @ t_pow
    y_noise = y + torch.randn_like(y) * self.rand_noise_amp
    return y, y_noise

  def forward(self):
    return self.generate_random_signal_batch()