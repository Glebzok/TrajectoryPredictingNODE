import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import torch
import torch.nn as nn
import wandb

import numpy as np
import random

from single_trajectory_data import LorenzTrajectory, SinTrajectory, SpiralTrajectory
from train import SingleTrajectoryTrainer
from shooting_model import SingleShooting, LatentSingleShooting, LatentMultipleShooting, LatentMultipleInterShooting, VariationalLatentMultipleShooting

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
    os.environ["CUDA_VISIBLE_DEVICES"] = '6'
    DATASET = 'SIN'
    experiment_name = 'Sin 0.1 noise multiple shooting 20 vars, 2T, 1e-2 lr 0.999 scheduler Adam 1e-3 wd 2 layers, linear rhs, Tanh decoder, 100 latent dim'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_params = {'T_train': 4*3.14}
    # data_params = {'T_train': 12}

    training_params = training_params = {'lambda1': 1e-9, 'lambda2': 1e-9, 'lambda3': 1e-9, 'lambda4': 0,
                       'l2_lambda': 0,
                       'n_iter': 32000, 'lr': 1e-2,
                       'logging_interval': 20, 'shooting_lambda_step': 1e-5}

    # trajectory = LorenzTrajectory(1, (0, 1, 2), T=10)
    trajectory = SinTrajectory(noise_std=0.1, T=8*3.14, n_points=402)
    # trajectory = SpiralTrajectory(noise_std=0.1, T=24, n_points=802, visible_dims=[0, 1])

    # shooting = LatentSingleShooting(signal_dim=1, latent_dim=20)
    # shooting = SingleShooting(len(trajectory.visible_dims))
    shooting = LatentMultipleShooting(signal_dim=1, latent_dim=100, n_shooting_vars=20)
    # shooting = LatentMultipleInterShooting(signal_dim=2, latent_dim=15, n_shooting_vars=41)
    # shooting = VariationalLatentMultipleShooting(signal_dim=2, latent_dim=100, n_shooting_vars=20, n_samples=256)

    config = {**training_params, **data_params}

    wandb.init(project='Sinus approximation',
               notes='',
               tags=['SingleTrajectory'],
               config=config,
               name=experiment_name,
               mode='online')

    wandb.watch(shooting)
    SingleTrajectoryTrainer(trajectory, shooting, config).train(device)
