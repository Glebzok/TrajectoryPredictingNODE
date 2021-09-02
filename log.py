
import matplotlib.pyplot as plt
import torch

def plot_reconstruction_results(rand_y, rand_y_noise, rand_y_rec):

  fig = plt.figure(figsize=(8, 5))

  clrs = plt.cm.get_cmap('prism', 7)

  for num, true, true_noise, pred in zip(range(rand_y.shape[0])[:3], rand_y.detach().cpu()[:3], rand_y_noise.detach().cpu()[:3], rand_y_rec.detach().cpu()[:3]):
    if num == 0:
      plt.plot(true, ls='', c=clrs(num), marker='o', ms=2, label='signal')
      plt.plot(true_noise, c=clrs(num), label='signal + noise')
      plt.plot(pred, c=clrs(num), ls='--', label='reconstructed')
    else:
      plt.plot(true, ls='', c=clrs(num), marker='o', ms=2)
      plt.plot(true_noise, c=clrs(num))
      plt.plot(pred, c=clrs(num), ls='--')

  return fig

def plot_approximation_results(model, batch_t, batch_y):

  fig = plt.figure(figsize=(8, 5))

  clrs = plt.cm.get_cmap('prism', 7)

  batch_t = torch.linspace(0, 2 * batch_t.max(), 2 * batch_t.shape[0], device=batch_t.device)
  pred_y = model(batch_y, batch_t).detach().cpu()

  batch_y = batch_y.detach().cpu()
  
  for num in range(3):
    if num == 0:
      plt.plot(batch_y[num], ls='-', c=clrs(num), label='signal')
      plt.plot(pred_y[num], c=clrs(num), ls='--', label='predicted')
    else:
      plt.plot(batch_y[num], ls='-', c=clrs(num))
      plt.plot(pred_y[num], c=clrs(num), ls='--')

  return fig