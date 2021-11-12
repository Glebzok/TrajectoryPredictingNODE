import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import torch
import torch.nn as nn
import wandb

import numpy as np
import random

from single_trajectory_data import LorenzTrajectory, SinTrajectory
from train import SingleTrajectoryTrainer
from shooting_model import SingleShooting, LatentMultipleShooting

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
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    DATASET = 'SIN'
    experiment_name = 'test'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_params = {'T_train': 4*3.14}

    training_params = {'lambda1': 0.1, 'lambda2': 0.1, 'lambda3': 0.1,
                       'n_iter': 800, 'lr': 1e-2,
                       'logging_interval': 1, 'shooting_lambda_step': 1e-2}

    # trajectory = LorenzTrajectory(1, (0, 1, 2), T=10)
    trajectory = SinTrajectory(noise_std=0.1, T=8*3.14, n_points=402)

    # shooting = SingleShooting(len(trajectory.visible_dims))
    shooting = LatentMultipleShooting(signal_dim=1, latent_dim=5, n_shooting_vars=10)

    config = {**training_params, **data_params}

    wandb.init(project='Sinus approximation',
               notes='',
               tags=['SingleTrajectory'],
               config=config,
               name=experiment_name,
               mode='online')

    wandb.watch(shooting)
    SingleTrajectoryTrainer(trajectory, shooting, config).train(device)
