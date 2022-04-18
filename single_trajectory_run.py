import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import torch
import torch.nn as nn
import wandb

import numpy as np
import random

from single_trajectory_data import LorenzTrajectory, SinTrajectory, SpiralTrajectory, CascadedTanksTrajectory, PendulumTrajectory, FluidFlowTrajectory, KarmanVortexStreet, ToyDataset
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
    os.environ["CUDA_VISIBLE_DEVICES"] = '4'
    DATASET = 'SPIRAL'
    # experiment_name = 'SPIRAL div 3 points (expt) multiple shooting 5 vars even more steps, 4T, 1e-4 lr 1e-3 wd, linear rhs, 5 layer 0.3 dropout (add linear layers) new permformer decoder, 50 latent dim, 1e-3 incr, norm init W, log t (uniform grid)'
    # experiment_name = 'test weight scaling 2'
    # experiment_name = 'FluidFlow (expt) multiple shooting 5 vars even more steps, 8T, 1e-4 lr 1e-3 wd, linear rhs, 5 layer fc decoder, 50 latent dim, 1e-3 incr, no norm W, log-norm penalty 1e2 lambda'
    # experiment_name = 'test new logging'
    experiment_name = 'Old Karman 0.8 402 points multiple shooting 5 vars, 1e-4 lr, linear rhs, 5 layer 50 hd fc decoder, 200 latent dim, 1e-3 incr, stablev3 random projection A'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'

    training_params = training_params = {'lambda1': 1e-9, 'lambda2': 1e-9, 'lambda3': 1e-9, 'lambda4': 0,
                       'l2_lambda': 0, 'log_norm_lambda': 0,
                       'n_iter': 512000, 'lr': 1e-4,
                       'logging_interval': 500, 'shooting_lambda_step': 1e-3}

    # trajectory = LorenzTrajectory(0, (0, 1, 2), T=12, n_points=402)
    # trajectory = PendulumTrajectory(n_points=402)
    # trajectory = FluidFlowTrajectory()
    trajectory = KarmanVortexStreet()
    # trajectory = ToyDataset(T=8*np.pi)
    # trajectory = CascadedTanksTrajectory()
    # trajectory = SinTrajectory(noise_std=0, T=8*3.14, n_points=402)
    # trajectory = SpiralTrajectory(noise_std=0, T=24, n_points=402, visible_dims=[0, 1])

    # data_params = {'T_train': 4*3.14}
    # data_params = {'T_train': 6}
    # data_params = {'T_train': 12}
    # data_params = {'T_train': 25}
    # data_params = {'T_train': 20}
    data_params = {'T_train': trajectory.T * 0.8}

    # shooting = LatentSingleShooting(signal_dim=1, latent_dim=20)
    # shooting = SingleShooting(len(trajectory.visible_dims))
    shooting = LatentMultipleShooting(signal_dim=trajectory.signal_dim, latent_dim=200, T=trajectory.T, n_shooting_vars=5)
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
    SingleTrajectoryTrainer(trajectory, shooting, config, experiment_name).train(device)
