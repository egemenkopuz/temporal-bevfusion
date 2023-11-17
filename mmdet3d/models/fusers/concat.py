from typing import List

import torch

from mmdet3d.models.builder import FUSERS

__all__ = ["ConcatFuser"]


@FUSERS.register_module()
class ConcatFuser:
    def __init__(self, dim: int = 1) -> None:
        self.dim = dim

    def __call__(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat(inputs, dim=self.dim)
