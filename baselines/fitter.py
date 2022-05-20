import torch
import torch.nn.functional as F
import wandb

from src.data.preprocessing import train_test_split


class Fitter:
    def __init__(self,
                 trajectory,
                 model,
                 train_frac,
                 experiment_name, project_name, notes, tags, config, mode):
        self.trajectory = trajectory
        self.model = model

        self.train_frac = train_frac

        self.experiment_name = experiment_name
        self.project_name = project_name
        self.notes = notes
        self.tags = tags
        self.config = config
        self.mode = mode

    def log_step(self,
                 t_train, y_clean_train, y_train,
                 y_pred,
                 t_test, y_clean_test, y_test,
                 losses_dict
                 ):

        prediction_table, prediction_video, prediction_image, _, _, _, signals_dict, val_losses = \
            self.trajectory.log_prediction_results(self.model,
                                                   t_train, y_clean_train, y_train, None, y_pred,
                                                   t_test, y_clean_test, y_test)

        log_dict = dict(losses_dict, **val_losses)
        log_dict['step'] = 0
        if prediction_table is not None:
            log_dict['prediction_results'] = prediction_table
        if prediction_video is not None:
            log_dict['prediction_results_gif'] = prediction_video
        if prediction_image is not None:
            log_dict['prediction_image'] = prediction_image

        wandb.log(log_dict)

    def fit(self):
        wandb.init(project=self.project_name,
                   notes=self.notes,
                   tags=self.tags,
                   config=self.config,
                   name=self.experiment_name,
                   mode=self.mode)

        t, y_clean, y = self.trajectory()

        t_train, y_clean_train, y_train, t_test, y_clean_test, y_test = \
            train_test_split(t=t, y_clean=y_clean, y=y, train_frac=self.train_frac,
                             normalize_t=False, scaling='total')

        self.model.fit(t_train, y_train)

        if torch.is_complex(y_train):
            y_clean_train, y_train, y_clean_test, y_test = \
                y_clean_train.real, y_train.real, y_clean_test.real, y_test.real

        y_pred = self.model(t_train, y_train)[0]

        rec_loss = F.mse_loss(y_pred, y_train)

        self.log_step(t_train, y_clean_train, y_train,
                      y_pred,
                      t_test, y_clean_test, y_test,
                      {'Reconstruction loss': rec_loss})
