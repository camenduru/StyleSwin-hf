#!/usr/bin/env python

from __future__ import annotations

import argparse
import functools
import os
import sys

if os.environ.get('SYSTEM') == 'spaces':
    os.system("sed -i '14,21d' StyleSwin/op/fused_act.py")
    os.system("sed -i '12,19d' StyleSwin/op/upfirdn2d.py")

sys.path.insert(0, 'StyleSwin')

import gradio as gr
import huggingface_hub
import numpy as np
import PIL.Image
import torch
import torch.nn as nn
from models.generator import Generator

REPO_URL = 'https://github.com/microsoft/StyleSwin'
TITLE = 'microsoft/StyleSwin'
DESCRIPTION = f'A demo for {REPO_URL}'
SAMPLE_IMAGE_DIR = 'https://huggingface.co/spaces/hysts/StyleSwin/resolve/main/samples'
ARTICLE = f'''## Generated images
### CelebA-HQ
- size: 1024x1024
- seed: 0-99
![CelebA-HQ samples]({SAMPLE_IMAGE_DIR}/celeba-hq.jpg)
### FFHQ
- size: 1024x1024
- seed: 0-99
![FFHQ samples]({SAMPLE_IMAGE_DIR}/ffhq.jpg)
### LSUN Church
- size: 256x256
- seed: 0-99
![LSUN Church samples]({SAMPLE_IMAGE_DIR}/lsun-church.jpg)
'''

TOKEN = os.environ['TOKEN']

MODEL_REPO = 'hysts/StyleSwin'
MODEL_NAMES = [
    'CelebAHQ_256',
    'FFHQ_256',
    'LSUNChurch_256',
    'CelebAHQ_1024',
    'FFHQ_1024',
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--theme', type=str)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--port', type=int)
    parser.add_argument('--disable-queue',
                        dest='enable_queue',
                        action='store_false')
    parser.add_argument('--allow-flagging', type=str, default='never')
    parser.add_argument('--allow-screenshot', action='store_true')
    return parser.parse_args()


def load_model(model_name: str, device: torch.device) -> nn.Module:
    size = int(model_name.split('_')[1])
    channel_multiplier = 1 if size == 1024 else 2
    model = Generator(size,
                      style_dim=512,
                      n_mlp=8,
                      channel_multiplier=channel_multiplier)
    ckpt_path = huggingface_hub.hf_hub_download(MODEL_REPO,
                                                f'models/{model_name}.pt',
                                                use_auth_token=TOKEN)
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['g_ema'])
    model.to(device)
    model.eval()
    return model


def generate_z(seed: int, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.random.RandomState(seed).randn(
        1, 512)).to(device).float()


def postprocess(tensors: torch.Tensor) -> torch.Tensor:
    assert tensors.dim() == 4
    tensors = tensors.cpu()
    std = torch.FloatTensor([0.229, 0.224, 0.225])[None, :, None, None]
    mean = torch.FloatTensor([0.485, 0.456, 0.406])[None, :, None, None]
    tensors = tensors * std + mean
    tensors = (tensors * 255).clamp(0, 255).to(torch.uint8)
    return tensors


@torch.inference_mode()
def generate_image(model_name: str, seed: int, model_dict: dict,
                   device: torch.device) -> PIL.Image.Image:
    model = model_dict[model_name]
    seed = int(np.clip(seed, 0, np.iinfo(np.uint32).max))
    z = generate_z(seed, device)
    out, _ = model(z)
    out = postprocess(out)
    out = out.numpy()[0].transpose(1, 2, 0)
    return PIL.Image.fromarray(out, 'RGB')


def main():
    gr.close_all()

    args = parse_args()
    device = torch.device(args.device)

    model_dict = {name: load_model(name, device) for name in MODEL_NAMES}

    func = functools.partial(generate_image,
                             model_dict=model_dict,
                             device=device)
    func = functools.update_wrapper(func, generate_image)

    gr.Interface(
        func,
        [
            gr.inputs.Radio(MODEL_NAMES,
                            type='value',
                            default='FFHQ_256',
                            label='Model',
                            optional=False),
            gr.inputs.Slider(0, 2147483647, step=1, default=0, label='Seed'),
        ],
        gr.outputs.Image(type='pil', label='Output'),
        title=TITLE,
        description=DESCRIPTION,
        article=ARTICLE,
        theme=args.theme,
        allow_screenshot=args.allow_screenshot,
        allow_flagging=args.allow_flagging,
        live=args.live,
    ).launch(
        enable_queue=args.enable_queue,
        server_port=args.port,
        share=args.share,
    )


if __name__ == '__main__':
    main()
