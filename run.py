import torch
import torch.nn as nn
import wandb

from model import NODESolver
from data import DataGenerator
from train import train


if __name__ == '__main__':
  wandb.login()

  device = 'cuda' if torch.cuda.is_available() else 'cpu'


  training_params = {'lambd': 0.2, 'n_iter': 10000, 'n_batch_steps': 1, 'lr':1e-3}

  data_params = {'latent_dim': 5, 'signal_dim': 1,
                'trajectory_len': 200, 'batch_size': 32, 'signal_max_amp':3,
                'signal_t_min': 0, 'signal_t_max': 4*3.14, 'signal_noise_amp':0.2,
                'rand_p':3, 'rand_q':0, 'rand_max_amp':0, 'rand_noise_amp':0}

  model_params = {'encoder_n_layers': 3, 'encoder_hidden_channels': 5,
                  'decoder_n_layers': 3, 'decoder_hidden_dim': 5,
                  'rhs_n_layers': 3, 'rhs_hidden_dim': 5}


  model = NODESolver(latent_dim=data_params['latent_dim'], signal_dim=data_params['signal_dim'],
                    **model_params).to(device)

  data_generator = DataGenerator(**data_params)

  optimizer = torch.optim.Adam(model.parameters(), lr=training_params['lr'])

  node_criterion = nn.L1Loss()
  rec_criterion = nn.L1Loss()


  config = {**training_params, **data_params, **model_params, 'opimizer': optimizer.__class__.__name__,
            'node_criterion': node_criterion.__class__.__name__, 'rec_criterion': rec_criterion.__class__.__name__}

  wandb.init(project='Sinus approximation',
            notes='testing',
            tags=['noiseless', 'testing'],
            config=config)

  wandb.watch(model)

train(model, optimizer, data_generator, node_criterion, rec_criterion, device, config)