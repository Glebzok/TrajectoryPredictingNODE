import torch
import matplotlib.pyplot as plt
import wandb
from torchdiffeq import odeint_adjoint as odeint


def train(model, optimizer, data_generator, node_criterion, rec_criterion, rhs_criterion, device, config):

  for itr in range(config['n_iter']):

    batch_t, batch_y = data_generator.generate_signal_batch()
    batch_t, batch_y = batch_t.to(device), batch_y.to(device)

    rand_y, rand_y_noise = data_generator.generate_random_signal_batch()
    rand_y, rand_y_noise = rand_y.to(device), rand_y_noise.to(device)

    if itr == 0:
      with torch.no_grad():
        torch.onnx.export(model.encoder, batch_y, 'encoder.onnx')
        wandb.save('encoder.onnx')

        z0 = model.encoder(batch_y)[:, :, 0]
        torch.onnx.export(model.rhs, (batch_t, z0), 'rhs.onnx')
        wandb.save('rhs.onnx')

        pred_z = odeint(model.rhs, z0, batch_t).to(batch_y.device)
        torch.onnx.export(model.decoder, pred_z, 'decoder.onnx')
        wandb.save('decoder.onnx')

    z = model.encoder(batch_y)
    pred_z = odeint(model.rhs, z[:, :, 0], batch_t).to(batch_y.device)
    rhs_loss = rhs_criterion(pred_z.permute(1, 2, 0), z)

    pred_y = model.decoder(pred_z).permute(1, 2, 0)
    node_loss = node_criterion(pred_y, batch_y)
    
    rand_y_rec = model.autoencoder_forward(rand_y_noise)
    rec_loss = rec_criterion(rand_y_rec, rand_y)
    
    loss = node_loss + config['lambd1'] * rec_loss + config['lambd2'] * rhs_loss

    optimizer.zero_grad()
    loss.backward()

    if itr % 10 == 0:

      reconstruction_table = data_generator.log_reconstruction_results(rand_y, rand_y_noise, rand_y_rec)
      approximation_table = data_generator.log_approximation_results(model, batch_t, batch_y)


      
      wandb.log({'step': itr,
                 'node_loss': node_loss.item(), 'rec_loss': rec_loss.item(), 'rhs_loss': rhs_loss.item(),
                 'max_pred_amplitude': torch.max(torch.abs(pred_y.cpu().detach())),
                 'reconstruction_table': reconstruction_table, 'approximation_table': approximation_table
                  })

      print(itr, node_loss.item(), rec_loss.item(), rhs_loss.item(), torch.max(torch.abs(pred_y.cpu().detach())).item())


    optimizer.step()
