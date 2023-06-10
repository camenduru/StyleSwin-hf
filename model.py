from __future__ import annotations

import os
import pathlib
import sys

import huggingface_hub
import numpy as np
import torch
import torch.nn as nn

if os.getenv('SYSTEM') == 'spaces':
    os.system("sed -i '14,21d' StyleSwin/op/fused_act.py")
    os.system("sed -i '12,19d' StyleSwin/op/upfirdn2d.py")

current_dir = pathlib.Path(__file__).parent
submodule_dir = current_dir / 'StyleSwin'
sys.path.insert(0, submodule_dir.as_posix())

from models.generator import Generator


class Model:
    MODEL_NAMES = [
        'CelebAHQ_256',
        'FFHQ_256',
        'LSUNChurch_256',
        'CelebAHQ_1024',
        'FFHQ_1024',
    ]

    def __init__(self):
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self._download_all_models()
        self.model_name = self.MODEL_NAMES[3]
        self.model = self._load_model(self.model_name)

        self.std = torch.FloatTensor([0.229, 0.224,
                                      0.225])[None, :, None,
                                              None].to(self.device)
        self.mean = torch.FloatTensor([0.485, 0.456,
                                       0.406])[None, :, None,
                                               None].to(self.device)

    def _load_model(self, model_name: str) -> nn.Module:
        size = int(model_name.split('_')[1])
        channel_multiplier = 1 if size == 1024 else 2
        model = Generator(size,
                          style_dim=512,
                          n_mlp=8,
                          channel_multiplier=channel_multiplier)
        ckpt_path = huggingface_hub.hf_hub_download('public-data/StyleSwin',
                                                    f'models/{model_name}.pt')
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['g_ema'])
        model.to(self.device)
        model.eval()
        return model

    def set_model(self, model_name: str) -> None:
        if model_name == self.model_name:
            return
        self.model_name = model_name
        self.model = self._load_model(model_name)

    def _download_all_models(self):
        for name in self.MODEL_NAMES:
            self._load_model(name)

    def generate_z(self, seed: int) -> torch.Tensor:
        seed = int(np.clip(seed, 0, np.iinfo(np.uint32).max))
        z = np.random.RandomState(seed).randn(1, 512)
        return torch.from_numpy(z).float().to(self.device)

    def postprocess(self, tensors: torch.Tensor) -> np.ndarray:
        assert tensors.dim() == 4
        tensors = tensors * self.std + self.mean
        tensors = (tensors * 255).clamp(0, 255).to(torch.uint8)
        return tensors.permute(0, 2, 3, 1).cpu().numpy()

    @torch.inference_mode()
    def generate_image(self, seed: int) -> np.ndarray:
        z = self.generate_z(seed)
        out, _ = self.model(z)
        out = self.postprocess(out)
        return out[0]

    def set_model_and_generate_image(self, model_name: str,
                                     seed: int) -> np.ndarray:
        self.set_model(model_name)
        return self.generate_image(seed)
