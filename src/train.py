import torch
import torch.nn.functional as F
from tqdm import tqdm
import pickle as pkl
import wandb
import os
from itertools import starmap

from src.data.preprocessing import train_test_split


class Trainer(object):
    def __init__(self,
                 trajectory, node_model, optimizer,
                 train_frac, n_iter, logging_interval,
                 lambda1, lambda2, lambda3, lambda4, lambda5, shooting_lambda_step,
                 device, experiment_name, project_name, notes, tags, config, mode,
                 log_dir,
                 scaling, normalize_t, normalize_rhs_loss):
        self.trajectory = trajectory
        self.node_model = node_model
        self.optimizer = optimizer

        self.train_frac = train_frac
        self.n_iter = n_iter
        self.logging_interval = logging_interval

        self.device = device

        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda4 = lambda4
        self.lambda5 = lambda5
        if lambda5 > 0:
            self.v = list(torch.randn((weight.shape[1], 1), device=self.device) \
                          for weight in list(filter(lambda param: param.dim() == 2,
                                                    self.node_model.shooting_model.rhs_net.dynamics.parameters())))

        self.shooting_lambda_step = shooting_lambda_step

        self.experiment_name = experiment_name
        self.project_name = project_name
        self.notes = notes
        self.tags = tags
        self.config = config
        self.mode = mode

        self.log_dir = log_dir

        self.scaling = scaling
        self.normalize_t = normalize_t
        self.normalize_rhs_loss = normalize_rhs_loss

    @staticmethod
    def power_iteration(A, v):
        u = A @ v
        v = A.T @ u
        v = v / torch.linalg.norm(v)
        sigma_sq = torch.linalg.norm(u) ** 2

        return v, sigma_sq

    def calc_loss(self, y, y_pred, z0_pred, z_pred):
        # y: (signal_dim, T)
        # y_pred: (n_samples, signal_dim, T)
        # z0_pred: (n_shooting_vars, latent_dim)
        # z_pred: (n_samples, n_shooting_vars, latent_dim, t)
        n_samples, latent_dim = z_pred.shape[0], z_pred.shape[-2]

        rec_loss = F.mse_loss(y_pred, y.expand(n_samples, -1, -1))
        loss = rec_loss
        losses = {'Reconstruction loss': rec_loss.item()}

        if self.lambda1 > 0:
            shooting_latent_loss = F.mse_loss(z_pred[:, 1:, :, 0], z_pred[:, :-1, :, -1])
            loss += self.lambda1 * shooting_latent_loss
            self.lambda1 += self.shooting_lambda_step
            losses['Shooting latent loss'] = shooting_latent_loss.item()

        if self.lambda2 > 0:
            z_pred_flattened_left = torch.cat(
                [z_pred[:, :, :, :-1].permute(0, 1, 3, 2).reshape(n_samples, -1, latent_dim),
                 z_pred[:, -1, :, -1:].permute(0, 2, 1)], dim=1)  # (n_samples, T, latent_dim)
            z_pred_flattened_left = z_pred_flattened_left.permute(0, 2, 1)  # (n_samples, latent_dim, T)

            z_pred_flattened_right = torch.cat([z_pred[:, 0, :, :1].permute(0, 2, 1),
                                                z_pred[:, :, :, 1:].permute(0, 1, 3, 2).reshape(n_samples, -1,
                                                                                                latent_dim)],
                                               dim=1)  # (n_samples, T, latent_dim)
            z_pred_flattened_right = z_pred_flattened_right.permute(0, 2, 1)  # (n_samples, latent_dim, T)

            y_pred_shooting_left = self.node_model.decoder_model(z_pred_flattened_left)  # (n_samples, signal_dim, T)
            y_pred_shooting_right = self.node_model.decoder_model(z_pred_flattened_right)  # (n_samples, signal_dim, T)

            shooting_loss = F.mse_loss(y_pred_shooting_left, y_pred_shooting_right)

            loss += self.lambda2 * shooting_loss
            self.lambda2 += self.shooting_lambda_step
            losses['Shooting loss'] = shooting_loss.item()

        if self.lambda3 > 0:
            shooting_rhs_loss = F.mse_loss(self.node_model.shooting_model.rhs_net(None, z_pred[:, 1:, :, 0]),
                                           self.node_model.shooting_model.rhs_net(None, z_pred[:, :-1, :, -1]))
            if self.normalize_rhs_loss:
                shooting_rhs_loss /= (self.node_model.shooting_model.rhs_net.dynamics.norm ** 2.)

            loss += self.lambda3 * shooting_rhs_loss
            self.lambda3 += self.shooting_lambda_step
            losses['Shooting RHS loss'] = shooting_rhs_loss.item()

        if self.lambda4 > 0:
            z_pred_sigma = z0_pred[:, latent_dim:]
            sigma_trace = (z_pred_sigma ** 2).sum(dim=1)  # shooting_vars
            var_loss = ((sigma_trace - z_pred_sigma.shape[1] / 16.) ** 2).mean()

            loss += self.lambda4 * var_loss
            losses['Sigma divergence'] = var_loss.item()

        if self.lambda5 > 0:
            weights = list(map(lambda weight: weight.data,
                               filter(lambda param: param.dim() == 2,
                                      self.node_model.shooting_model.rhs_net.dynamics.parameters())))
            self.v, sigmas_sq = list(zip(*starmap(self.power_iteration, zip(weights, self.v))))
            spectral_penalty = sum(sigmas_sq) / len(sigmas_sq)

            loss += self.lambda5 * spectral_penalty
            losses['Spectral penalty'] = spectral_penalty.item()

        return loss, losses

    def log_step(self,
                 itr,
                 t_train, y_clean_train, y_train,
                 z_pred, y_pred,
                 t_test, y_clean_test, y_test,
                 losses_dict):
        with torch.no_grad():
            # t_train : (T_train,)
            # t_test : (T_test,
            # y_clean_train / y_train : (signal_dim, T_train)
            # y_clean_test / y_test : (signal_dim, T_test)
            # z_pred : (n_samples, n_shooting_vars, latent_dim, t)
            # y_pred : (n_samples, signal_dim, T_train)
            y_pred = y_pred[0]  # (signal_dim, T_train)
            z_pred = z_pred[0]  # (n_shooting_vars, latent_dim, t)

            prediction_table, prediction_video, prediction_image, spectrum_table, \
            shooting_latent_trajectories, inference_latent_trajectories, signals_dict, val_losses = \
                self.trajectory.log_prediction_results(self.node_model,
                                                       t_train, y_clean_train, y_train, z_pred, y_pred,
                                                       t_test, y_clean_test, y_test)

            log_dict = dict(losses_dict, **val_losses)
            log_dict['step'] = itr
            if prediction_table is not None:
                log_dict['prediction_results'] = prediction_table
            if prediction_video is not None:
                log_dict['prediction_results_gif'] = prediction_video
            if prediction_image is not None:
                log_dict['prediction_image'] = prediction_image
            log_dict['spectrum'] = spectrum_table
            log_dict['shooting_latent_trajectories'] = shooting_latent_trajectories
            log_dict['inference_latent_trajectories'] = inference_latent_trajectories

            wandb.log(log_dict)

        return signals_dict

    def save_model(self, itr, signals_dict):
        with open(f'{self.log_dir}/{wandb.run.id}_{self.experiment_name}/{itr}_true.pkl', 'wb') as f:
            pkl.dump(signals_dict['true'], f)

        with open(f'{self.log_dir}/{wandb.run.id}_{self.experiment_name}/{itr}_pred.pkl', 'wb') as f:
            pkl.dump(signals_dict['pred'], f)

        torch.save(self.node_model.state_dict(), f'{self.log_dir}/{wandb.run.id}_{self.experiment_name}/{itr}_model.pt')

    def train(self):
        wandb.init(project=self.project_name,
                   notes=self.notes,
                   tags=self.tags,
                   config=self.config,
                   name=self.experiment_name,
                   mode=self.mode)
        wandb.watch(self.node_model, log_freq=100, log_graph=True)

        t, y_clean, y = self.trajectory()

        t_train, y_clean_train, y_train, t_test, y_clean_test, y_test = \
            train_test_split(t=t, y_clean=y_clean, y=y, train_frac=self.train_frac,
                             normalize_t=self.normalize_t, scaling=self.scaling)

        t_train, y_clean_train, y_train, t_test, y_clean_test, y_test = \
            t_train.to(self.device), y_clean_train.to(self.device), y_train.to(self.device), \
            t_test.to(self.device), y_clean_test.to(self.device), y_test.to(self.device)

        for itr in tqdm(range(self.n_iter)):
            if itr == 0:
                if not os.path.exists(f'{self.log_dir}/{wandb.run.id}_{self.experiment_name}'):
                    os.mkdir(f'{self.log_dir}/{wandb.run.id}_{self.experiment_name}')

            def closure():
                y_pred, z0_pred, z_pred = self.node_model(t_train, y_train)
                loss, step_losses = self.calc_loss(y_train, y_pred, z0_pred, z_pred)

                self.optimizer.zero_grad()
                loss.backward()
                return loss

            if itr % self.logging_interval == 0:

                y_pred, z0_pred, z_pred = self.node_model(t_train, y_train)
                loss, step_losses = self.calc_loss(y_train, y_pred, z0_pred, z_pred)

                self.node_model.eval()

                signals_dict = \
                    self.log_step(itr,
                                  t_train, y_clean_train, y_train,
                                  z_pred, y_pred,
                                  t_test, y_clean_test, y_test,
                                  step_losses)

                if itr % (100 * self.logging_interval) == 0:
                    self.save_model(itr, signals_dict)

                self.node_model.train()

            self.optimizer.step(closure)
