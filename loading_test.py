import os

import torch
from diffusers import AutoencoderKL
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import create_vae_diffusers_config, \
    convert_ldm_vae_checkpoint
from omegaconf import OmegaConf

vae = torch.load('kl-f8-anime2.ckpt')
if 'state_dict' in vae:
    vae = vae['state_dict']
dirname = os.path.dirname(__file__)
original_config = OmegaConf.load(os.path.join(dirname, 'utils/default_config.yaml'))
vae_config = create_vae_diffusers_config(original_config, image_size=512)
new_vae = {}
for k, v in vae.items():
    new_vae['first_stage_model.' + k] = v
converted_vae_checkpoint = convert_ldm_vae_checkpoint(new_vae, vae_config)
vae = AutoencoderKL(**vae_config)
vae.load_state_dict(converted_vae_checkpoint)
print(vae)
