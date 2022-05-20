import torch
import torch.nn as nn
import numpy as np

import hydra
from omegaconf import DictConfig

from utils.seed import seed_everything
from utils.create import create_trajectory, create_fitter, create_reservoir_model


class ReservoirModelWrapper(nn.Module):
    def __init__(self, model, warm_steps):
        super().__init__()
        self.model = model
        self.warm_steps = warm_steps

    def fit(self, t, y):
        self.model = self.model.fit(y.T[:-1], y.T[1:], warmup=self.warm_steps)

    def forward(self, t, y):
        warmup_y = self.model.run(y[:, :self.warm_steps].T, reset=True)
        pred = np.zeros((y.shape[0], t.shape[0]))
        pred[:, :self.warm_steps] = warmup_y.T

        y_ = warmup_y[-1].reshape(1, -1)
        for i in range(t.shape[0] - self.warm_steps):
            y_ = self.model(y_)
            pred[:, i + self.warm_steps] = y_

        pred = torch.tensor(pred, dtype=torch.float32)
        return pred, None, None

    def inference(self, t, y):
        return self.forward(t, y)


@hydra.main(config_path="../configs", config_name="reservoir_toy_config.yaml")
def app(cfg: DictConfig):

    seed_everything(cfg.environment.SEED)

    trajectory = create_trajectory(cfg.trajectory)

    reservoir_model = ReservoirModelWrapper(create_reservoir_model(cfg.model, seed=cfg.environment.SEED), warm_steps=cfg.training.warm_steps)

    fitter = create_fitter(config=cfg, fitter_config=cfg.fitter, trajectory=trajectory, model=reservoir_model)
    fitter.fit()


if __name__ == '__main__':
    app()
