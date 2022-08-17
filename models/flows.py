#!/usr/bin/python

"""
Flow producer.
Code adapted from https://github.com/deepmind/flows_for_atomic_solids
"""

from logging import Logger
from typing import Mapping, Any
import torch
from torch import Tensor
from nflows.transforms import CompositeTransform
from models.bijectors import CircularShift
from models.split import SplitCoupling
from models.spline import RationalQuadraticSpline


def make_split_coupling_flow(
    angles: Tensor,
    lower: float,
    upper: float,
    num_layers: int,
    num_bins: int,
    conditioner: Mapping[str, Any],
    use_circular_shift: bool,
    logger: Logger,
    circular_shift_init=torch.zeros,
    device='cuda'
) -> CompositeTransform:
    """Create a flow that consists of a sequence of split coupling layers.

    All coupling layers use rational-quadratic splines. Each layer of the flow
    is composed of two split coupling bijectors, where each coupling bijector
    transforms a different part of the input.

    The flow maps to and from the range `[lower, upper]`, obeying periodic
    boundary conditions.

    Args:
      angles: a Tensor of shape (N) whose N is the batch size and 1 is angles.
      lower: lower range of the flow.
      upper: upper range of the flow.
      num_layers: the number of layers to use. Each layer consists of two split
        coupling bijectors, where each coupling bijector transforms a different
        part of the input.
      num_bins: number of bins to use in the rational-quadratic splines.
      conditioner: a Mapping containing 'constructor' and 'kwargs' keys that
        configures the conditioner used in the coupling layers.
      use_circular_shift: if True, add a learned circular shift between successive
        flow layers.
      circular_shift_init: initializer for the circular shifts.

    Returns:
      The flow, a nflow bijector.
    """

    def bijector_fn(params: Tensor):
        return RationalQuadraticSpline(
            params,
            range_min=lower,
            range_max=upper,
            boundary_slopes='circular',
            logger=logger,
            device=device,
            min_bin_size=(upper - lower) * 1e-4)

    layers = []
    for _ in range(num_layers):
        sublayers = []

        # Circular shift.
        if use_circular_shift:
            shift_layer = CircularShift(
                lower, upper, 
                circular_shift_init, 
                logger=logger, device=device)
            sublayers.append(shift_layer)

        # Coupling layer.
        coupling_layer = SplitCoupling(
            angles=angles,
            bijector=bijector_fn,
            logger=logger,
            conditioner=conditioner['constructor'](
                num_bijector_params=3 * num_bins + 1,
                lower=lower,
                upper=upper,
                angles=angles,
                logger=logger,
                device=device,
                **conditioner['kwargs'])
        )
        sublayers.append(coupling_layer)
        layers.append(CompositeTransform(sublayers))

    return CompositeTransform(layers)
