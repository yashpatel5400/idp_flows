#!/usr/bin/python

"""
Conditioner for IDP models.
Code adapted from https://github.com/deepmind/flows_for_atomic_solids
"""

from logging import Logger
from typing import Callable, Mapping, Any
import torch
from torch import Tensor
from torch.nn import (Linear, Module)


def circular(x: Tensor,
             lower: float,
             upper: float,
             num_frequencies: int,
             device: str) -> Tensor:
    """Maps angles to points on the unit circle.

    The mapping is such that the interval [lower, upper] is mapped to a full
    circle starting and ending at (1, 0). For num_frequencies > 1, the mapping
    also includes higher frequencies which are multiples of 2 pi/(lower-upper)
    so that [lower, upper] wraps around the unit circle multiple times.

    Args:
        x: array of shape [..., D].
        lower: lower limit, angles equal to this will be mapped to (1, 0).
        upper: upper limit, angles equal to this will be mapped to (1, 0).
        num_frequencies: number of frequencies to consider in the embedding.

    Returns:
        An array of shape [..., 2*num_frequencies*D].
    """
    base_frequency = 2. * torch.pi / (upper - lower)
    frequencies = base_frequency * torch.arange(1, num_frequencies+1, device=device)
    angles = frequencies * (x[..., None] - lower)
    # Reshape from [..., D, num_frequencies] to [..., D*num_frequencies].
    angles = angles.reshape(x.shape[:-1] + (-1,))
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return torch.concat([cos, sin], axis=-1)


class Conditioner(Module):
    """Make a conditioner for the coupling flow."""

    def __init__(self,
                 num_bijector_params: int,
                 lower: float,
                 upper: float,
                 angles: Tensor,
                 embedding_size: int,
                 conditioner_constructor: Callable[..., Any],
                 conditioner_kwargs: Mapping[str, Any],
                 num_frequencies: int,
                 logger: Logger,
                 device='cuda'):
        """
        This conditioner assumes that the input is of shape [..., N, D1]. 
        It returns an output of shape [..., N, K], where:
        K = `num_bijector_params`
        """
        super().__init__()
        self.logger = logger
        self.device = device
        self.linear1 = Linear(in_features=2 * num_frequencies * angles.shape[-1],
                              out_features=embedding_size,
                              device=self.device)
        self.conditioner = conditioner_constructor(
            d_model=embedding_size,
            device=self.device,
            **conditioner_kwargs)
        self.linear2 = Linear(in_features=embedding_size,
                              out_features=num_bijector_params,
                              device=self.device)
        self.circular_kwards = dict(
            lower=lower,
            upper=upper,
            num_frequencies=num_frequencies
        )

    def forward(self, input: Tensor):
        out = circular(input,
                       **self.circular_kwards,
                       device=self.device)
        out = self.linear1(out)
        out = self.conditioner(out, out)
        out = self.linear2(out)
        return out
