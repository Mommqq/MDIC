import torch
from config import ConfigMDIC as cfg_MDIC
from diffusers import PNDMScheduler
from torch.nn import Linear


def find_linear_layers(module, prefix=''):
    """get names of all linear layers"""
    linear_layer_names = []
    for name, child_module in module.named_children():
        if name.startswith('conv_in_extended'):
            linear_layer_names.append(prefix + name)
        if isinstance(child_module, Linear):
            linear_layer_names.append(prefix + name)
        else:
            linear_layer_names.extend(find_linear_layers(child_module, prefix + name + '.'))
    return linear_layer_names

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


def get_pred_original_sample(noise_scheduler, timesteps, sample, model_output):
    """get predicted x_0 from inputs"""
    prediction_type = noise_scheduler.config.prediction_type
    alphas_cumprod = noise_scheduler.alphas_cumprod

    # 1. compute sqrt_alpha_prod, sqrt_one_minus_alpha_prod
    sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
    sqrt_alpha_prod = sqrt_alpha_prod.flatten()
    while len(sqrt_alpha_prod.shape) < len(sample.shape):
        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
    while len(sqrt_one_minus_alpha_prod.shape) < len(sample.shape):
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

    if prediction_type == "epsilon":
        pred_original_sample = (sample - sqrt_one_minus_alpha_prod * model_output) / sqrt_alpha_prod
    elif prediction_type == "v_prediction":
        pred_original_sample = sqrt_alpha_prod * sample - sqrt_one_minus_alpha_prod * model_output
       
    else:
        raise ValueError(
            f"prediction_type given as {prediction_type} must be one of `epsilon`, or"
            " `v_prediction` for the DDPMScheduler."
        )
    return pred_original_sample


def update_scheduler(pipe):
    """update scheduler + prediction type (for SD v1.5 only)"""
    if cfg_MDIC.prediction_type == "v_prediction":
        # bypass frozen dict
        pipe.scheduler = PNDMScheduler(
            num_train_timesteps=pipe.scheduler.config['num_train_timesteps'],
            beta_start=pipe.scheduler.config['beta_start'],
            beta_end=pipe.scheduler.config['beta_end'],
            beta_schedule=pipe.scheduler.config['beta_schedule'],
            trained_betas=pipe.scheduler.config['trained_betas'],
            skip_prk_steps=pipe.scheduler.config['skip_prk_steps'],
            set_alpha_to_one=pipe.scheduler.config['set_alpha_to_one'],
            prediction_type="v_prediction",
            timestep_spacing=pipe.scheduler.config['timestep_spacing'],
            steps_offset=pipe.scheduler.config['steps_offset'],
        )
