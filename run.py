import hydra
from omegaconf import DictConfig
import os

from src.models.model import NODEModel
from utils.create import create_optimizer, create_trajectory, create_encoder, create_shooting_model, \
    create_decoder_model, create_trainer
from utils.seed import seed_everything


@hydra.main(config_path="./configs", config_name="sin_config.yaml")
def app(cfg: DictConfig):
    device = cfg.environment.device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.environment.GPU_id

    seed_everything(cfg.environment.SEED)

    trajectory = create_trajectory(cfg.trajectory)

    encoder = create_encoder(encoder_config=cfg.model.encoder,
                             shooting_config=cfg.model.shooting,
                             latent_dim=cfg.model.config.latent_dim,
                             signal_dim=trajectory.signal_dim,
                             n_shooting_vars=cfg.model.config.n_shooting_vars)

    decoder = create_decoder_model(decoder_config=cfg.model.decoder,
                                   latent_dim=cfg.model.config.latent_dim,
                                   signal_dim=trajectory.signal_dim)

    shooting = create_shooting_model(shooting_config=cfg.model.shooting,
                                     latent_dim=cfg.model.config.latent_dim,
                                     signal_dim=trajectory.signal_dim,
                                     decoder=decoder)

    model = NODEModel(encoder, shooting, decoder).to(device)
    optimizer = create_optimizer(cfg.optimizer,
                                 cfg.model.encoder.config, cfg.model.shooting, cfg.model.decoder.config,
                                 model)

    trainer = create_trainer(config=cfg,
                             trainer_config=cfg.trainer,
                             trajectory=trajectory,
                             node_model=model,
                             optimizer=optimizer)

    trainer.train()


if __name__ == '__main__':
    app()
