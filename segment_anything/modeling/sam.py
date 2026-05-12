# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F
from icecream import ic

from typing import Any, Dict, List, Tuple

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder


class Sam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    def forward(self,adapter:torch.nn.Module,patchembed3d_half:torch.nn.Module,decode:torch.nn.Module,batched_input, multimask_output, image_size):
        if isinstance(batched_input, list):
            outputs = self.forward_test(batched_input, multimask_output)
        else:
            outputs = self.forward_train(adapter,patchembed3d_half,decode,batched_input, multimask_output, image_size)
        return outputs

    def forward_train(self,adapter:torch.nn.Module,patchembed3d_half:torch.nn.Module,decode:torch.nn.Module,batched_input, multimask_output, image_size):
        input_images = batched_input#self.preprocess(batched_input
        #batch = input_images.shape[0]
        #box = torch.tensor([[0,0,512,512]]*batch).float().cuda()
        image_embeddings = self.image_encoder(input_images,adapter,patchembed3d_half)
        res = decode(image_embeddings)

        return res#low_res_masks#outputs



    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

