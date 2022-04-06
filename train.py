import torch
import torch.nn.functional as F
from tqdm import tqdm
import pickle as pkl
import wandb
import os


class Trainer():
    def __init__(self, model, optimizer, scheduler, data_generator, node_criterion, rec_criterion, rhs_criterion):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.data_generator = data_generator
        self.node_criterion = node_criterion
        self.rec_criterion = rec_criterion
        self.rhs_criterion = rhs_criterion

    def log_model(self, t, y):
        with torch.no_grad():
            z = self.model.encoder(y)

            torch.onnx.export(self.model.encoder.to('cpu'), y.to('cpu'), 'encoder.onnx', opset_version=11)
            self.model.encoder.to(y.device)
            wandb.save('encoder.onnx')

            torch.onnx.export(self.model.rhs, (t, z[:, :, 0]), 'rhs.onnx')
            wandb.save('rhs.onnx')

            torch.onnx.export(self.model.decoder, z, 'decoder.onnx')
            wandb.save('decoder.onnx')

    def log_step(self, itr, t, y, rand_y, rand_y_noise, rand_y_rec, losses_dict):
        with torch.no_grad():
            reconstruction_table = self.data_generator.log_reconstruction_results(rand_y, rand_y_noise, rand_y_rec)
            approximation_table = self.data_generator.log_approximation_results(self.model, t, y)

            wandb.log(dict(losses_dict, **{'step': itr, 'reconstruction_table': reconstruction_table,
                                           'approximation_table': approximation_table}))

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
            self.scheduler.step()


class SingleTrajectoryTrainer:
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
        self.l2_lambda = config['l2_lambda']
        self.log_norm_lambda = config['log_norm_lambda']

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
            prediction_table, prediction_video, spectrum_table,\
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
        N = (t <= T_train).sum()
        t_train, t_test = t[:N], t[N:]
        y_clean_train, y_clean_test = y_clean[:, :N], y_clean[:, N:]
        y_train, y_test = y[:, :N], y[:, N:]
        # print(t_train.shape, y_train.shape, t_test.shape, y_test.shape)
        return t_train, y_clean_train, y_train, t_test, y_clean_test, y_test

    def calc_loss(self, y, pred_y, shooting_begin_values, shooting_end_values):
        rec_loss = F.mse_loss(y, pred_y)
        loss = rec_loss
        losses = {'Reconstruction loss': rec_loss.item()}
        if self.lambda1 > 0:
            shooting_latent_loss = F.mse_loss(shooting_begin_values, shooting_end_values)
            loss += self.lambda1 * shooting_latent_loss
            self.lambda1 += self.config['shooting_lambda_step']
            losses['Shooting latent loss'] = shooting_latent_loss.item()
        if self.lambda2 > 0:
            shooting_loss = F.mse_loss(self.shooting.decoder(shooting_begin_values[:, None, :]),
                                       self.shooting.decoder(shooting_end_values[:, None, :]))
            loss += self.lambda2 * shooting_loss
            self.lambda2 += self.config['shooting_lambda_step']
            losses['Shooting loss'] = shooting_loss.item()
        if self.lambda3 > 0:
            shooting_rhs_loss = F.mse_loss(self.shooting.rhs(None, shooting_begin_values),
                                           self.shooting.rhs(None, shooting_end_values))
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
        _ = self.shooting(t_train, y_train)
        self.optimizer = torch.optim.Adam(self.shooting.parameters(), lr=self.config['lr'], weight_decay=1e-3)
        # self.optimizer = torch.optim.LBFGS(self.shooting.parameters(), lr=self.config['lr'], tolerance_change=1e-30)
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)

        # with torch.autograd.detect_anomaly():
        for itr in tqdm(range(self.config['n_iter'])):

            self.logged = False

            if itr == 0:
                # self.log_model()
                self.shooting = self.shooting.to(device)
                if not os.path.exists(f'./model/{self.experiment_name}'):
                    os.mkdir(f'./model/{self.experiment_name}')

            y_pred, z_pred, shooting_end_values, shooting_begin_values = self.shooting(t_train, y_train)
            loss, step_losses = self.calc_loss(y_train, y_pred, shooting_begin_values, shooting_end_values)

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
