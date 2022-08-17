#!/usr/bin/python

"""
Customized bijectors.
Code adapted from https://github.com/deepmind/flows_for_atomic_solids
"""

from logging import Logger
from typing import Union, Tuple
import torch
from torch import Tensor
from torch.nn import Parameter
from nflows.transforms import Transform


class CircularShift(Transform):
    """Shift with wrapping around. Or, translation on a torus."""

    def __init__(self,
                 lower: Union[float, Tensor],
                 upper: Union[float, Tensor],
                 logger: Logger,
                 circular_shift_init=torch.zeros,
                 device='cuda'):
        super().__init__()
        self.logger = logger
        self.device = device
        if (not torch.is_tensor(lower)) and (not torch.is_tensor(upper)) and (lower >= upper):
            raise ValueError('`lower` must be less than `upper`.')

        try:
            width = upper - lower
        except TypeError as e:
            raise ValueError('`lower` and `upper` must be broadcastable to same '
                             f'shape, but `lower`={lower} and `upper`={upper}') from e

        self.shift = Parameter(
            data=circular_shift_init(size=(1, ), device=self.device, requires_grad=True))
        self.wrap = lambda x: torch.remainder(x - lower, width) + lower
        # self.shift = self.wrap(shift)

    def forward(self, inputs: Tensor, context=None) -> Tuple[Tensor, Tensor]:
        outputs = self.wrap(inputs + self.shift)
        logabsdet = torch.zeros_like(inputs)
        logabsdet = logabsdet.sum(dim=-1)
        return outputs, logabsdet

    def inverse(self, inputs: Tensor, context=None) -> Tuple[Tensor, Tensor]:
        outputs = self.wrap(inputs - self.shift)
        logabsdet = torch.zeros_like(inputs)
        logabsdet = logabsdet.sum(dim=-1)
        return outputs, logabsdet
