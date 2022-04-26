import torch
import torch.nn.functional as F
from tqdm import tqdm
import pickle as pkl
import wandb
import os


class Trainer:
    def __init__(self, trajectory, shooting, config, experiment_name):
        self.trajectory = trajectory
        self.shooting = shooting
        self.optimizer = None
        self.config = config
        self.experiment_name = experiment_name

        self.lambda1 = config['lambda1']
        self.lambda2 = config['lambda2']
        self.lambda3 = config['lambda3']
        self.lambda4 = config['lambda4']

    def log_model(self):
        with torch.no_grad():
            if hasattr(self.shooting, 'encoder'):
                x = torch.randn((1, self.shooting.signal_dim, 10))
                torch.onnx.export(self.shooting.encoder.to('cpu'), x, 'encoder.onnx', opset_version=11)
                wandb.save('encoder.onnx')

            if hasattr(self.shooting, 'rhs'):
                x = torch.randn((1, self.shooting.rhs.system_dim))
                torch.onnx.export(self.shooting.rhs.to('cpu'), (None, x), 'rhs.onnx')
                wandb.save('rhs.onnx')

            if hasattr(self.shooting, 'decoder'):
                x = torch.randn((10, 1, self.shooting.latent_dim))
                torch.onnx.export(self.shooting.decoder.to('cpu'), x, 'decoder.onnx')
                wandb.save('decoder.onnx')

    def log_step(self, itr, t_train, y_clean_train, y_train, z_pred, y_pred, t_test, y_clean_test, y_test, losses_dict):
        with torch.no_grad():
            if len(y_pred.shape) == 3:
                y_pred = y_pred[0]
            prediction_table, prediction_video, prediction_image, spectrum_table,\
            shooting_latent_trajectories, inference_latent_trajectories, signals_dict = \
                self.trajectory.log_prediction_results(self.shooting,
                                                       t_train, y_clean_train, y_train, z_pred, y_pred,
                                                       t_test, y_clean_test, y_test)

            log_dict = losses_dict
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

        with open(f'./model/{self.experiment_name}/{itr}_true.pkl', 'wb') as f:
            pkl.dump(signals_dict['true'], f)

        with open(f'./model/{self.experiment_name}/{itr}_pred.pkl', 'wb') as f:
            pkl.dump(signals_dict['pred'], f)

        # torch.save(self.shooting, f'./model/{self.experiment_name}/{itr}_model.pt')
        with open(f'./model/{self.experiment_name}/{itr}_model.pkl', 'wb') as f:
            pkl.dump(self.shooting.rhs.linear.weight, f)

    @staticmethod
    def train_test_split(t, y_clean, y, T_train):



        # !!!!! ADD SCALING



        N = (t <= T_train).sum()
        t_train, t_test = t[:N], t[N:]
        y_clean_train, y_clean_test = y_clean[:, :N], y_clean[:, N:]
        y_train, y_test = y[:, :N], y[:, N:]
        # print(t_train.shape, y_train.shape, t_test.shape, y_test.shape)
        return t_train, y_clean_train, y_train, t_test, y_clean_test, y_test

    def calc_loss(self, y, pred_y, z_pred):
        #z_pred (t, n_shooting_vars, latent_dim)
        rec_loss = F.mse_loss(y, pred_y)
        loss = rec_loss
        losses = {'Train Reconstruction loss': rec_loss.item()}
        if self.lambda1 > 0:
            shooting_latent_loss = F.mse_loss(z_pred[0, 1:, :], z_pred[-1, :-1, :])
            loss += self.lambda1 * shooting_latent_loss
            self.lambda1 += self.config['shooting_lambda_step']
            losses['Shooting latent loss'] = shooting_latent_loss.item()
        if self.lambda2 > 0:
            latent_dim = z_pred.shape[-1]
            pred_z_flattend_left = torch.cat([z_pred[:-1, :, :].permute(1, 0, 2).reshape(-1, latent_dim),
                                              z_pred[-1:, -1, :]], dim=0)  # (T, latent_dim)
            pred_z_flattend_right = torch.cat([z_pred[:1, 0, :],
                                               z_pred[1:, :, :].permute(1, 0, 2).reshape(-1, latent_dim)], dim=0)  # (T, latent_dim)

            y_pred_shooting_left = self.shooting.decoder(pred_z_flattend_left) # (T, signal_dim)
            y_pred_shooting_right = self.shooting.decoder(pred_z_flattend_right)  # (T, signal_dim)

            shooting_loss = F.mse_loss(y_pred_shooting_left, y_pred_shooting_right)

            loss += self.lambda2 * shooting_loss
            self.lambda2 += self.config['shooting_lambda_step']
            losses['Shooting loss'] = shooting_loss.item()
        if self.lambda3 > 0:
            shooting_rhs_loss = F.mse_loss(self.shooting.rhs(None, z_pred[0, 1:, :]),
                                           self.shooting.rhs(None, z_pred[-1, :-1, :]))
            loss += self.lambda3 * shooting_rhs_loss
            self.lambda3 += self.config['shooting_lambda_step']
            losses['Shooting RHS loss'] = shooting_rhs_loss.item()

        if self.lambda4 > 0:
            sigma_trace = (torch.exp(self.shooting.shooting_vars_sigma) ** 2).sum(dim=1)  # shooting_vars
            var_loss = ((sigma_trace - self.shooting.shooting_vars_sigma.shape[1]) ** 2).mean()

            loss += self.lambda4 * var_loss
            losses['sigma_trace'] = var_loss.item()

        if self.l2_lambda > 0:
            l2_reg = torch.tensor(0., device=loss.device)
            for param in self.shooting.parameters():
                l2_reg += torch.linalg.norm(param)

            loss += self.l2_lambda * l2_reg
            losses['L2 loss'] = l2_reg.item()

        if self.log_norm_lambda > 0:
            log_norm = F.relu(torch.linalg.eigvalsh(self.shooting.rhs.linear.weight + self.shooting.rhs.linear.weight.T)[-1])

            loss += self.log_norm_lambda * log_norm
            losses['log_norm'] = log_norm.item()

        return loss, losses

    def train(self, device):
        t, y_clean, y = self.trajectory()
        t_train, y_clean_train, y_train, t_test, y_clean_test, y_test = \
            self.train_test_split(t=t, y_clean=y_clean, y=y, T_train=self.config['T_train'])
        t_train, y_clean_train, y_train, t_test, y_clean_test, y_test = \
            t_train.to(device), y_clean_train.to(device), y_train.to(device), \
            t_test.to(device), y_clean_test.to(device), y_test.to(device)

        self.shooting = self.shooting.to(device)
        self.shooting.rhs.decoder = self.shooting.decoder
        _ = self.shooting(t_train, y_train)
        # self.optimizer = torch.optim.Adam(self.shooting.parameters(), lr=self.config['lr'], weight_decay=0)
        self.optimizer = torch.optim.Adam([{'params': self.shooting.decoder.parameters(), 'weight_decay': 0},
                                           {'params': self.shooting.shooting_vars, 'weight_decay': 0},
                                           {'params': self.shooting.rhs.linear.parameters(), 'weight_decay': 0}] \
                                          + (
                                              [{'params' : self.shooting.rhs.controller.parameters(), 'weight_decay': 0}]
                                              if hasattr(self.shooting.rhs, 'controller')
                                              else []
                                          ), lr=self.config['lr'])
        # self.optimizer = torch.optim.LBFGS(self.shooting.parameters(), lr=self.config['lr'], tolerance_change=1e-30)
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)

        # with torch.autograd.detect_anomaly():
        for itr in tqdm(range(self.config['n_iter'])):

            self.logged = False

            if itr == 0:
                # self.log_model()
                # self.shooting = self.shooting.to(device)
                if not os.path.exists(f'./model/{self.experiment_name}'):
                    os.mkdir(f'./model/{self.experiment_name}')

            y_pred, z_pred = self.shooting(t_train, y_train)
            loss, step_losses = self.calc_loss(y_train, y_pred, z_pred)

            self.optimizer.zero_grad()
            loss.backward()

            if (itr % self.config['logging_interval'] == 0) and (not self.logged):
                # print(itr)
                self.shooting.eval()
                signals_dict = \
                    self.log_step(itr, t_train, y_clean_train, y_train, z_pred, y_pred, t_test, y_clean_test, y_test,
                                  step_losses)

                if itr % (100 * self.config['logging_interval']) == 0:
                    # print(itr, itr)
                    self.save_model(itr, signals_dict)

                self.shooting.train()
                self.logged = True

            # if itr == 60:
            #     print('l', shooting_begin_values.detach())
            #     print('r', shooting_end_values.detach())
            #     print('lrhs', self.shooting.rhs(None, shooting_begin_values).detach())
            #     print('rrhs', self.shooting.rhs(None, shooting_end_values).detach())


            self.optimizer.step()
            # self.scheduler.step()