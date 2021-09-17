import argparse

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]="6"

import torch
import torch.nn as nn
import wandb

import numpy as np
import random

from model import NODESolver
from data import SinDataGenerator, LorenzDataGenerator
from train import train
from callbacks import ensure_clean_worktree, get_commit_hash

DATASET = 'LORENZ'

seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
# Torch
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# Numpy 
np.random.seed(seed)
# Python
random.seed(seed)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--dev", help="running in dev mode", action="store_true")
  args = parser.parse_args()

  wandb.login()

  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  training_params = {'lambd': 0.2, 'n_iter': 10000, 'n_batch_steps': 1, 'lr': 1e-3}
  
  if DATASET == 'SIN':   

    data_params = {'latent_dim': 5, 'signal_dim': 1,
                  'trajectory_len': 200, 'batch_size': 256, 'signal_max_amp': 3,
                  'signal_t_min': 0, 'signal_t_max': 4*3.14, 'signal_noise_amp': 0.2,
                  'rand_p': 3, 'rand_q': 0, 'rand_max_amp': 1, 'rand_noise_amp': 0.2}

    model_params = {'encoder_n_layers': 3, 'encoder_hidden_channels': 5,
                    'decoder_n_layers': 3, 'decoder_hidden_dim': 5,
                    'rhs_n_layers': 3, 'rhs_hidden_dim': 5}

    data_generator = SinDataGenerator(**data_params)

  elif DATASET == 'LORENZ':

    data_params = {'latent_dim': 5, 'signal_dim': 3,
                  'trajectory_len': 1000, 'batch_size': 256, 'signal_max_amp': 10,
                  'sigma': 10.0, 'rho':28.0, 'beta':8.0/3.0,
                  'signal_t_min': 0, 'signal_t_max': 5, 'signal_noise_amp': 0,
                  'rand_p': 3, 'rand_q': 0, 'rand_max_amp': 1, 'rand_noise_amp': 0}

    model_params = {'encoder_n_layers': 3, 'encoder_hidden_channels': 5,
                    'decoder_n_layers': 3, 'decoder_hidden_dim': 5,
                    'rhs_n_layers': 3, 'rhs_hidden_dim': 5}

    data_generator = LorenzDataGenerator(**data_params)

  else:
    raise ValueError('Wrong dataset name')

  model = NODESolver(latent_dim=data_params['latent_dim'], signal_dim=data_params['signal_dim'],
                    **model_params).to(device)


  optimizer = torch.optim.Adam(model.parameters(), lr=training_params['lr'])

  node_criterion = nn.L1Loss()
  rec_criterion = nn.L1Loss()


  config = {**training_params, **data_params, **model_params, 'opimizer': optimizer.__class__.__name__,
            'node_criterion': node_criterion.__class__.__name__, 'rec_criterion': rec_criterion.__class__.__name__}

  if args.dev:
    config['n_iter'] = 1
    mode = 'offline'

  else:
    ensure_clean_worktree()
    mode = 'online'  

  experiment_name = 'Naively scaled Lorenz with new plots'

  wandb.init(project='Sinus approximation',
              notes='testing',
              tags=[],
              config=config,
              name=experiment_name,
              mode=mode)
  
  wandb.watch(model)
  train(model, optimizer, data_generator, node_criterion, rec_criterion, device, config)