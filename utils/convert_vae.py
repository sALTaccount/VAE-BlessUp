import os
from diffusers import AutoencoderKL
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import create_vae_diffusers_config, \
    convert_ldm_vae_checkpoint
from omegaconf import OmegaConf


def diffusers_to_compvis(vae):
    #  Based on huggingface diffusers conversion scripts https://github.com/huggingface/diffusers/blob/main/scripts/
    vae_conversion_map = [
        # (stable-diffusion, HF Diffusers)
        ("nin_shortcut", "conv_shortcut"),
        ("norm_out", "conv_norm_out"),
        ("mid.attn_1.", "mid_block.attentions.0."),
    ]

    for i in range(4):
        # down_blocks have two resnets
        for j in range(2):
            hf_down_prefix = f"encoder.down_blocks.{i}.resnets.{j}."
            sd_down_prefix = f"encoder.down.{i}.block.{j}."
            vae_conversion_map.append((sd_down_prefix, hf_down_prefix))

        if i < 3:
            hf_downsample_prefix = f"down_blocks.{i}.downsamplers.0."
            sd_downsample_prefix = f"down.{i}.downsample."
            vae_conversion_map.append((sd_downsample_prefix, hf_downsample_prefix))

            hf_upsample_prefix = f"up_blocks.{i}.upsamplers.0."
            sd_upsample_prefix = f"up.{3 - i}.upsample."
            vae_conversion_map.append((sd_upsample_prefix, hf_upsample_prefix))

        # up_blocks have three resnets
        # also, up blocks in hf are numbered in reverse from sd
        for j in range(3):
            hf_up_prefix = f"decoder.up_blocks.{i}.resnets.{j}."
            sd_up_prefix = f"decoder.up.{3 - i}.block.{j}."
            vae_conversion_map.append((sd_up_prefix, hf_up_prefix))

    # this part accounts for mid blocks in both the encoder and the decoder
    for i in range(2):
        hf_mid_res_prefix = f"mid_block.resnets.{i}."
        sd_mid_res_prefix = f"mid.block_{i + 1}."
        vae_conversion_map.append((sd_mid_res_prefix, hf_mid_res_prefix))

    vae_conversion_map_attn = [
        # (stable-diffusion, HF Diffusers)
        ("norm.", "group_norm."),
        ("q.", "query."),
        ("k.", "key."),
        ("v.", "value."),
        ("proj_out.", "proj_attn."),
    ]

    def reshape_weight_for_sd(w):
        # convert HF linear weights to SD conv2d weights
        return w.reshape(*w.shape, 1, 1)

    def convert_vae_state_dict(vae_state_dict):
        mapping = {k: k for k in vae_state_dict.keys()}
        for k, v in mapping.items():
            for sd_part, hf_part in vae_conversion_map:
                v = v.replace(hf_part, sd_part)
            mapping[k] = v
        for k, v in mapping.items():
            if "attentions" in k:
                for sd_part, hf_part in vae_conversion_map_attn:
                    v = v.replace(hf_part, sd_part)
                mapping[k] = v
        new_state_dict = {v: vae_state_dict[k] for k, v in mapping.items()}
        weights_to_convert = ["q", "k", "v", "proj_out"]
        for k, v in new_state_dict.items():
            for weight_name in weights_to_convert:
                if f"mid.attn_1.{weight_name}.weight" in k:
                    print(f"Reshaping {k} for SD format")
                    new_state_dict[k] = reshape_weight_for_sd(v)
        return new_state_dict

    vae_state_dict = vae.state_dict()
    new_state_dict = convert_vae_state_dict(vae_state_dict)
    return {'state_dict': new_state_dict}


def compvis_to_diffusers(vae):
    if 'state_dict' in vae:
        vae = vae['state_dict']
    dirname = os.path.dirname(__file__)
    original_config = OmegaConf.load(os.path.join(dirname, 'default_config.yaml'))
    vae_config = create_vae_diffusers_config(original_config, image_size=512)
    new_vae = {}
    for k, v in vae.items():
        new_vae['first_stage_model.' + k] = v
    converted_vae_checkpoint = convert_ldm_vae_checkpoint(new_vae, vae_config)
    vae = AutoencoderKL(**vae_config)
    vae.load_state_dict(converted_vae_checkpoint)
    return vae
