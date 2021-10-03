import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
from torchdiffeq import odeint_adjoint as odeint


class Trainer():
  def __init__(self, model, optimizer, data_generator, node_criterion, rec_criterion, rhs_criterion):
    self.model = model
    self.optimizer = optimizer
    self.data_generator = data_generator
    self.node_criterion = node_criterion
    self.rec_criterion = rec_criterion
    self.rhs_criterion = rhs_criterion

  def log_model(self, t, y):
    with torch.no_grad():
      z = self.model.encoder(y)

      torch.onnx.export(self.model.encoder, y, 'encoder.onnx')
      wandb.save('encoder.onnx')

      torch.onnx.export(self.model.rhs, (t, z[:, :, 0]), 'rhs.onnx')
      wandb.save('rhs.onnx')

      torch.onnx.export(self.model.decoder, z, 'decoder.onnx')
      wandb.save('decoder.onnx')

  def log_step(self, itr, t, y, rand_y, rand_y_noise, rand_y_rec, losses_dict):
    with torch.no_grad():
      reconstruction_table = self.data_generator.log_reconstruction_results(rand_y, rand_y_noise, rand_y_rec)
      approximation_table = self.data_generator.log_approximation_results(self.model, t, y)
      
      wandb.log(dict(losses_dict, **{'step': itr,'reconstruction_table': reconstruction_table, 'approximation_table': approximation_table}))

  def train(self, device, config):
    for itr in tqdm(range(config['n_iter'])):

      batch_t, batch_y = self.data_generator.generate_signal_batch()
      batch_t, batch_y = batch_t.to(device), batch_y.to(device)

      rand_y, rand_y_noise = self.data_generator.generate_random_signal_batch()
      rand_y, rand_y_noise = rand_y.to(device), rand_y_noise.to(device)

      if itr == 0:
        self.log_model(batch_t, batch_y)
        
      z, pred_z, pred_y = self.model(batch_y, batch_t)
      rand_y_rec = self.model.autoencoder_forward(rand_y_noise)

      rhs_loss = self.rhs_criterion(pred_z, z)
      node_loss = self.node_criterion(pred_y, batch_y)
      rec_loss = self.rec_criterion(rand_y_rec, rand_y)
      
      loss = node_loss + config['lambd1'] * rec_loss + config['lambd2'] * rhs_loss

      self.optimizer.zero_grad()
      loss.backward()

      if itr % 10 == 0:
        losses = {'node_loss': node_loss.item(), 'rec_loss': rec_loss.item(), 'rhs_loss': rhs_loss.item(),
                  'max_pred_amplitude': torch.max(torch.abs(pred_y.cpu().detach()))}
        
        self.log_step(itr, batch_t, batch_y, rand_y, rand_y_noise, rand_y_rec, losses)

      self.optimizer.step()
