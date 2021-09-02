import torch
import matplotlib.pyplot as plt
import wandb
from torchdiffeq import odeint_adjoint as odeint

from log import plot_approximation_results, plot_reconstruction_results

def train(model, optimizer, data_generator, node_criterion, rec_criterion, device, config):

  for itr in range(config['n_iter']):

    batch_t, batch_y = data_generator.generate_signal_batch()
    batch_t, batch_y = batch_t.to(device), batch_y.to(device)

    rand_y, rand_y_noise = data_generator.generate_random_signal_batch()
    rand_y, rand_y_noise = rand_y.to(device), rand_y_noise.to(device)

    if itr == 0:
      torch.onnx.export(model.encoder, batch_y[:, None, :], 'encoder.onnx')
      wandb.save('encoder.onnx')

      z0 = model.encoder(batch_y[:, None, :])[:, :, 0]
      torch.onnx.export(model.rhs, (batch_t, z0), 'rhs.onnx')
      wandb.save('rhs.onnx')

      pred_z = odeint(model.rhs, z0, batch_t).to(batch_y.device)
      torch.onnx.export(model.decoder, pred_z, 'decoder.onnx')
      wandb.save('decoder.onnx')

    for _ in range(config['n_batch_steps']):
      pred_y = model(batch_y, batch_t)
      node_loss = node_criterion(pred_y, batch_y)
      
      rand_y_rec = model.autoencoder_forward(rand_y_noise)
      rec_loss = rec_criterion(rand_y_rec, rand_y)
      
      loss = node_loss + config['lambd'] * rec_loss

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()


    if itr % 10 == 0:

      reconstruction_fig = plot_reconstruction_results(rand_y, rand_y_noise, rand_y_rec)
      approximation_fig = plot_approximation_results(model, batch_t, batch_y)


      
      wandb.log({'step': itr, 'node_loss': node_loss.item(), 'rec_loss': rec_loss.item(), 'max_pred_amplitude': torch.max(torch.abs(pred_y.cpu().detach())),
                  'reconstruction_fig': reconstruction_fig, 'approximation_fig': approximation_fig
                  })
      
      reconstruction_fig.clear()
      approximation_fig.clear()
      plt.close(reconstruction_fig)
      plt.close(approximation_fig)

      print(itr, node_loss.item(), rec_loss.item(), torch.max(torch.abs(pred_y.cpu().detach())))