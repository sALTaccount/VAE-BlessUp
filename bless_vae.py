import os

import torch
from diffusers import AutoencoderKL
from torch import nn
import argparse

import utils.convert_vae

parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str, required=True)
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--output_path', type=str, required=True)
parser.add_argument('--output_type', type=str, required=True)
parser.add_argument('--contrast', type=float)
parser.add_argument('--contrast_operation', type=str, default='mul')
parser.add_argument('--brightness', type=float)
parser.add_argument('--brightness_operation', type=str, default='add')
parser.add_argument('--patch_encoder', action='store_true')

args = parser.parse_args()

if not args.contrast and not args.brightness:
    raise ValueError('Must specify at least one of contrast or brightness to modify')

print('Loading model...')
if args.model_type == 'compvis':
    vae = torch.load(args.model_path)
    vae = utils.convert_vae.compvis_to_diffusers(vae)
elif args.model_type == 'diffusers':
    vae = AutoencoderKL.from_pretrained(args.model_path, subfolder='vae', latent_channels=4)
else:
    raise ValueError('Invalid model type')

print('Applying modifications...')
if args.contrast is not None:
    # weight affects contrast
    if args.contrast_operation == 'add':
        vae.decoder.conv_out.weight = nn.Parameter(vae.decoder.conv_out.weight + args.contrast)
        if args.patch_encoder:
            vae.encoder.conv_in.weight = nn.Parameter(vae.encoder.conv_in.weight - args.contrast)
    elif args.contrast_operation == 'mul':
        vae.decoder.conv_out.weight = nn.Parameter(vae.decoder.conv_out.weight * args.contrast)
        if args.patch_encoder:
            vae.encoder.conv_in.weight = nn.Parameter(vae.encoder.conv_in.weight / args.contrast)
    else:
        raise ValueError('Invalid contrast operation')

if args.brightness is not None:
    # bias affects brightness
    if args.brightness_operation == 'add':
        vae.decoder.conv_out.weight = nn.Parameter(vae.decoder.conv_out.weight + args.contrast)
        if args.patch_encoder:
            vae.encoder.conv_in.weight = nn.Parameter(vae.encoder.conv_in.weight - args.contrast)
    elif args.brightness_operation == 'mul':
        vae.decoder.conv_out.weight = nn.Parameter(vae.decoder.conv_out.weight * args.contrast)
        if args.patch_encoder:
            vae.encoder.conv_in.weight = nn.Parameter(vae.encoder.conv_in.weight / args.contrast)
    else:
        raise ValueError('Invalid brightness operation')

print('Saving...')
if args.output_type == 'diffusers':
    vae.save_pretrained(args.output_path)
elif args.output_type == 'compvis':
    new_vae = utils.convert_vae.diffusers_to_compvis(vae)
    torch.save(new_vae, args.output_path)
print('Done!')
