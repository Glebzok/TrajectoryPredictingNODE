import torch
import torch.nn.functional as F

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import wandb


class AbstractTrajectory():
    def __init__(self, t0, T, n_points, noise_std, signal_amp, **kwargs):
        self.t0 = t0
        self.T = T
        self.n_points = n_points
        self.noise_std = noise_std
        self.signal_amp = signal_amp

        self.signal_dim = None
        self.visible_dims = None

    def generate_visible_trajectory(self, y_clean):
        y = y_clean + torch.randn_like(y_clean) * self.noise_std
        return y

    def __call__(self):
        raise NotImplementedError()

    def log_prediction_table(self,
                             t_train, y_clean_train, y_train, y_pred, y_train_inference,
                             t_test, y_clean_test, y_test, y_test_inference):
        # t_train : (T_train,)
        # t_test : (T_test,
        # y_clean_train / y_train : (signal_dim, T_train)
        # y_clean_test / y_test : (signal_dim, T_test)
        # y_pred : (signal_dim, T_train)
        # y_train_inference: (signal_dim, T_train)
        # y_test_inference: (signal_dim, T_test)
        train_log_table = pd.DataFrame(np.concatenate([t_train.reshape(-1, 1),
                                                       y_clean_train.T,
                                                       y_train.T,
                                                       y_pred.T,
                                                       y_train_inference.T], axis=1),
                                       columns=['t'] \
                                               + ['y_true_clean_y%i' % i for i in self.visible_dims] \
                                               + ['y_true_noisy_y%i' % i for i in self.visible_dims] \
                                               + ['y_pred_shooting_y%i' % i for i in self.visible_dims] \
                                               + ['y_pred_inference_y%i' % i for i in self.visible_dims])

        train_log_table = pd.melt(train_log_table, id_vars=['t'], value_name='y', var_name='description')
        train_log_table['stage'] = 'train'

        test_log_table = pd.DataFrame(np.concatenate([t_test.reshape(-1, 1),
                                                      y_clean_test.T,
                                                      y_test.T,
                                                      y_test_inference.T], axis=1),
                                      columns=['t'] \
                                              + ['y_true_clean_y%i' % i for i in self.visible_dims] \
                                              + ['y_true_noisy_y%i' % i for i in self.visible_dims] \
                                              + ['y_pred_inference_y%i' % i for i in self.visible_dims])

        test_log_table = pd.melt(test_log_table, id_vars=['t'], value_name='y', var_name='description')
        test_log_table['stage'] = 'test'

        log_table = pd.concat([train_log_table, test_log_table])
        log_table['type'] = log_table['description'].str.split('_').str[1]
        log_table['subtype'] = log_table['description'].str.split('_').str[2]
        log_table['variable'] = log_table['description'].str.split('_').str[3]

        log_table = pd.pivot_table(log_table.drop(columns=['description']),
                                   values='y', index=['t', 'stage', 'type', 'subtype'],
                                   columns=['variable']).reset_index()

        log_table = wandb.Table(dataframe=log_table)
        return log_table

    @staticmethod
    def log_prediction_image(y_clean_train, y_train, y_pred, y_train_inference,
                             y_clean_test, y_test, y_test_inference):

        train_len, test_len = y_train.shape[-1], y_test.shape[-1]

        dim = y_train.shape[0]

        log_image = np.zeros([dim * 4, train_len + test_len, 1])
        log_image[:, :train_len, 0] = np.concatenate([y_clean_train,
                                                      y_train,
                                                      y_pred,
                                                      y_train_inference])
        log_image[:, train_len:, 0] = np.concatenate([y_clean_test,
                                                      y_test,
                                                      np.zeros_like(y_test),
                                                      y_test_inference])
        log_image = wandb.Image(log_image)
        return log_image

    def log_prediction_gif(self,
                           y_clean_train, y_train, y_pred, y_train_inference,
                           y_clean_test, y_test, y_test_inference):
        train_len, test_len = y_train.shape[-1], y_test.shape[-1]
        max_len = max([train_len, test_len])

        height, width = self.init_dim

        log_video = np.zeros([height * 4, width * 2, max_len])
        log_video[:, :width, :train_len] = np.concatenate([y_clean_train.reshape([*self.init_dim, -1]),
                                                           y_train.reshape([*self.init_dim, -1]),
                                                           y_pred.reshape([*self.init_dim, -1]),
                                                           y_train_inference.reshape([*self.init_dim, -1])])
        log_video[:, width:, :test_len] = np.concatenate([y_clean_test.reshape([*self.init_dim, -1]),
                                                          y_test.reshape([*self.init_dim, -1]),
                                                          np.zeros_like(y_test).reshape([*self.init_dim, -1]),
                                                          y_test_inference.reshape([*self.init_dim, -1])])

        log_video = np.repeat(log_video[:, :, :, None], 3, axis=3).transpose([2, 3, 0, 1])
        log_video = ((log_video - log_video.min()) / (log_video.max() - log_video.min()) * 255.).astype('uint8')
        log_video = wandb.Video(log_video, fps=12, format='mp4')

        return log_video

    @staticmethod
    def log_spectrum(model):
        eigv = np.linalg.eigvals(model.shooting_model.rhs_net.dynamics.weight.detach().cpu().numpy())

        spectrum_table = wandb.Table(data=[[x, y] for (x, y) in zip(eigv.real, eigv.imag)], columns=["Re", "Im"])

        return spectrum_table

    @staticmethod
    def log_latent_trajectories(t_train, z_pred, t_test, z_train_inference, z_test_inference):
        z_pred = z_pred.transpose(2, 0, 1)
        points_per_shooting_var, n_shooting_vars, latent_dim = z_pred.shape
        points_per_shooting_var -= 1

        last_point = z_pred[-1:, -1, :]  # (1, latent_dim)
        z_pred = np.concatenate([z_pred[:-1, :, :].transpose([1, 0, 2]).reshape(-1, latent_dim), last_point],
                                axis=0).T  # (latent_dim, T)

        plt.rcParams.update({'font.size': 22})

        shooting_fig = plt.figure(figsize=(20, 10), num=1, clear=True)
        for z in z_pred:
            plt.plot(t_train, z)
            plt.scatter(t_train[:-1:points_per_shooting_var], z[:-1:points_per_shooting_var])
        plt.xlabel('$t$')
        plt.ylabel('$z_i(t)$')
        plt.title('Latent shooting trajectories')
        plt.grid()
        shooting_image = wandb.Image(shooting_fig)

        inference_fig = plt.figure(figsize=(20, 10), num=1, clear=True)
        for z_train, z_test in zip(z_train_inference, z_test_inference):
            plt.plot(t_train, z_train)
            plt.plot(t_test, z_test)
        plt.xlabel('$t$')
        plt.ylabel('$z_i(t)$')
        plt.title('Latent inference trajectories')
        plt.grid()
        inference_image = wandb.Image(inference_fig)

        return shooting_image, inference_image

    @staticmethod
    def calc_val_losses(y_train, y_test, y_train_inference, y_test_inference):
        train_inference_loss = F.mse_loss(y_train, y_train_inference)
        test_inference_loss = F.mse_loss(y_test, y_test_inference)

        return {'Train Inference Reconstruction loss': train_inference_loss.item(),
                'Test Inference Reconstruction loss': test_inference_loss.item()}

    def log_prediction_results(self, model, t_train, y_clean_train, y_train, z_pred, y_pred, t_test, y_clean_test,
                               y_test):
        y_inference, _, z_inference = model.inference(torch.cat([t_train, t_test]),
                                                      y_train)  # (signal_dim, T), (latent_dim, T)
        y_train_inference = y_inference[:, :t_train.shape[0]]
        y_test_inference = y_inference[:, t_train.shape[0]:]
        z_train_inference = z_inference[:, :t_train.shape[0]]
        z_test_inference = z_inference[:, t_train.shape[0]:]

        val_losses = self.calc_val_losses(y_train, y_test, y_train_inference, y_test_inference)

        t_train, y_clean_train, y_train, z_pred, y_pred, y_train_inference, t_test, y_clean_test, y_test, \
        y_test_inference, z_train_inference, z_test_inference = \
            t_train.cpu().numpy(), y_clean_train.cpu().numpy(), y_train.cpu().numpy(), \
            z_pred.detach().cpu().numpy(), y_pred.detach().cpu().numpy(), \
            y_train_inference.detach().cpu().numpy(), t_test.cpu().numpy(), \
            y_clean_test.cpu().numpy(), y_test.cpu().numpy(), \
            y_test_inference.detach().cpu().numpy(), \
            z_train_inference.detach().cpu().numpy(), z_test_inference.detach().cpu().numpy()

        signals = \
            {'true': {'t_train': t_train,
                      't_test': t_test,
                      'y_clean_train': y_clean_train,
                      'y_train': y_train,
                      'y_clean_test': y_clean_test,
                      'y_test': y_test},
             'pred': {'y_train_pred': y_pred,
                      'z_train_pred': z_pred,
                      'y_train_inference': y_train_inference,
                      'y_test_inference': y_test_inference,
                      'z_train_inference': z_train_inference,
                      'z_test_inference': z_test_inference}}

        if len(self.visible_dims) <= 3:
            log_table = self.log_prediction_table(t_train, y_clean_train, y_train, y_pred, y_train_inference,
                                                  t_test, y_clean_test, y_test, y_test_inference)
        else:
            log_table = None

        if self.signal_dim > 3:
            if self.init_dim is None:
                log_video = None
                log_image = self.log_prediction_image(y_clean_train, y_train, y_pred, y_train_inference,
                                                      y_clean_test, y_test, y_test_inference)
            else:
                log_video = self.log_prediction_gif(y_clean_train, y_train, y_pred, y_train_inference,
                                                    y_clean_test, y_test, y_test_inference)
                log_image = None
        else:
            log_video = None
            log_image = None

        spectrum_table = self.log_spectrum(model)
        shooting_latent_trajectories, inference_latent_trajectories = \
            self.log_latent_trajectories(t_train, z_pred, t_test, z_train_inference, z_test_inference)

        return log_table, log_video, log_image, spectrum_table, \
               shooting_latent_trajectories, inference_latent_trajectories, signals, val_losses
