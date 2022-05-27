import torch
import torch.nn as nn

from math import floor
import hydra
from omegaconf import DictConfig

from utils.seed import seed_everything
from utils.create import create_trajectory, create_fitter, create_universal


class DMDModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def fit(self, t, y):
        self.t0, self.T = t[0].item(), t[-1].item()
        self.dt = (self.T - self.t0) / (t.shape[0] - 1)
        self.model.fit(y.T)

    def forward(self, t, y):
        self.model.dmd_time['t0'] = floor((t[0].item() - self.t0) / self.dt)
        self.model.dmd_time['tend'] = floor((t[-1].item() - self.t0) / self.dt)

        pred = torch.tensor(self.model.reconstructed_data.real, dtype=torch.float32)

        return pred, None, None

    def inference(self, t, y):
        return self.forward(t, y)


@hydra.main(config_path="../configs", config_name="dmd_karman_config.yaml")
def app(cfg: DictConfig):

    seed_everything(cfg.environment.SEED)

    trajectory = create_trajectory(cfg.trajectory)

    dmd_model = DMDModelWrapper(create_universal(cfg.model))

    fitter = create_fitter(config=cfg, fitter_config=cfg.fitter, trajectory=trajectory, model=dmd_model)
    fitter.fit()


if __name__ == '__main__':
    app()
