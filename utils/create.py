import importlib
import inspect
from omegaconf import DictConfig, OmegaConf
from functools import partial
import pandas as pd
import torch.nn as nn


def get_attr_from_module(module, attr):
    try:
        mdl = importlib.import_module(module)
    except Exception:
        raise RuntimeError(f'failed to import module {module}')

    try:
        result = getattr(mdl, attr)
    except Exception:
        raise RuntimeError(f'attribute {attr} not found in {module}')

    return result


def create_trajectory(trajectory_config: DictConfig):
    trajectory_type = get_attr_from_module(trajectory_config.module, trajectory_config.type)
    trajectory = trajectory_type(**trajectory_config.config)
    return trajectory


def create_encoder(encoder_config: DictConfig, shooting_config:DictConfig,
                   latent_dim, signal_dim, n_shooting_vars):
    encoder_type = get_attr_from_module(encoder_config.module, encoder_config.type)

    if shooting_config.type == 'VariationalShooting':
        latent_dim = latent_dim * 2

    if encoder_config.type == 'NontrivialEncoder':
        encoder_net_type = get_attr_from_module(encoder_config.config.encoder_net.module,
                                                encoder_config.config.encoder_net.type)

        encoder_net = encoder_net_type(latent_dim=latent_dim, signal_dim=signal_dim,
                                       **encoder_config.config.encoder_net.config)
        encoder = encoder_type(n_shooting_vars=n_shooting_vars, encoder_net=encoder_net)

    elif encoder_config.type == 'DirectOptimizationEncoder':
        encoder = encoder_type(n_shooting_vars=n_shooting_vars, latent_dim=latent_dim,
                               init_distribution=encoder_config.config.init_distribution)

    else:
        encoder = encoder_type(n_shooting_vars=n_shooting_vars)
    return encoder


def create_decoder_model(decoder_config: DictConfig, latent_dim, signal_dim):
    decoder_model_type = get_attr_from_module(decoder_config.module, decoder_config.type)
    if decoder_config.type == 'TrivialDecoder':
        decoder_model = decoder_model_type()
    else:
        decoder_net_type = get_attr_from_module(decoder_config.decoder_net.module, decoder_config.decoder_net.type)
        decoder_net = decoder_net_type(latent_dim=latent_dim, signal_dim=signal_dim,
                                       **decoder_config.decoder_net.config)
        decoder_model = decoder_model_type(decoder_net, **decoder_config.config)

    return decoder_model


def create_shooting_model(shooting_config: DictConfig, latent_dim, signal_dim, decoder):
    shooting_model_type = get_attr_from_module(shooting_config.module, shooting_config.type)
    rhs_net_type = get_attr_from_module(shooting_config.rhs_net.module, shooting_config.rhs_net.type)

    # if shooting_config.type == 'VariationalShooting':
    #     latent_dim = latent_dim

    if shooting_config.rhs_net.type != 'FCRHS':
        linear_type = get_attr_from_module(shooting_config.linear.module, shooting_config.linear.type)
        linear = linear_type(n=latent_dim, **shooting_config.linear.config)

        rhs_net = rhs_net_type(signal_dim=signal_dim, system_dim=latent_dim, decoder=decoder, linear=linear,
                               **shooting_config.rhs_net.config)
    else:
        rhs_net = rhs_net_type(signal_dim=signal_dim, system_dim=latent_dim, decoder=decoder,
                               **shooting_config.rhs_net.config)

    odeint = get_attr_from_module(shooting_config.rhs_net.odeint.module, shooting_config.rhs_net.odeint.type)
    shooting = shooting_model_type(rhs_net=rhs_net, odeint=odeint, **shooting_config.config)

    return shooting


def create_optimizer(opt_config: DictConfig,
                     encoder_config: DictConfig,
                     shooting_config: DictConfig,
                     decoder_config: DictConfig,
                     model: nn.Module):
    opt_type = get_attr_from_module(opt_config.module, opt_config.type)
    if opt_config.type == 'LBFGS':
        params = model.parameters()
    else:
        params = [{'params': model.encoder_model.parameters(), 'weight_decay': encoder_config.weight_decay},
                  {'params': model.decoder_model.parameters(), 'weight_decay': decoder_config.weight_decay},
                  {'params': model.shooting_model.rhs_net.dynamics.parameters(),
                   'weight_decay': shooting_config.rhs_net.config.dynamics_weight_decay}]

        if hasattr(model.shooting_model.rhs_net, 'controller'):
            params.append({'params': model.shooting_model.rhs_net.controller.parameters(),
                           'weight_decay': shooting_config.rhs_net.config.controller_weight_decay})

    opt = opt_type(params=params, **opt_config.config)
    return opt


def create_trainer(config: DictConfig,
                   trainer_config: DictConfig,
                   trajectory, node_model, optimizer):
    trainer_type = get_attr_from_module(trainer_config.module, trainer_config.type)

    normalized_config = pd.json_normalize(OmegaConf.to_container(config)).to_dict(orient='records')[0]
    trainer = trainer_type(trajectory=trajectory, node_model=node_model, optimizer=optimizer,
                           device=config.environment.device,
                           **trainer_config.config, config=normalized_config)
    return trainer


def create_universal(config: DictConfig):
    par_type = get_attr_from_module(config.module, config.type)
    if inspect.isclass(par_type):
        par = par_type(**config.config)
    else:
        par = partial(par_type, **config.config) if 'config' in config.keys() else par_type

    return par


def create_fitter(config: DictConfig,
                  fitter_config: DictConfig,
                  trajectory, model):
    fitter_type = get_attr_from_module(fitter_config.module, fitter_config.type)

    normalized_config = pd.json_normalize(OmegaConf.to_container(config)).to_dict(orient='records')[0]
    fitter = fitter_type(trajectory=trajectory, model=model,
                         **fitter_config.config, config=normalized_config)
    return fitter


def create_reservoir_model(reservoir_config: DictConfig, seed):
    reservoir_type = get_attr_from_module(reservoir_config.module, reservoir_config.reservoir.type)
    readout_type = get_attr_from_module(reservoir_config.module, reservoir_config.readout.type)

    reservoir = reservoir_type(**reservoir_config.reservoir.config, seed=seed)
    readout = readout_type(**reservoir_config.readout.config)

    model = reservoir >> readout
    return model
