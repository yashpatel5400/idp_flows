#!/usr/bin/python

"""
split coupling bijector class.
Code adapted from https://github.com/deepmind/distrax
"""

from logging import Logger
from typing import Tuple
from torch import Tensor
from nflows.transforms import Transform


class SplitCoupling(Transform):
    """Split coupling bijector, with arbitrary conditioner & inner bijector.
    This coupling bijector splits the input array into two parts along a specified
    axis. One part remains unchanged, whereas the other part is transformed by an
    inner bijector conditioned on the unchanged part.
    Let `f` be a conditional bijector (the inner bijector) and `g` be a function
    (the conditioner). For `swap=False`, the split coupling bijector is defined as
    follows:
    - Forward:
      ```
      x = [x1, x2]
      y1 = x1
      y2 = f(x2; g(x1))
      y = [y1, y2]
      ```
    - Forward Jacobian log determinant:
      ```
      x = [x1, x2]
      log|det J(x)| = log|det df/dx2(x2; g(x1))|
      ```
    - Inverse:
      ```
      y = [y1, y2]
      x1 = y1
      x2 = f^{-1}(y2; g(y1))
      x = [x1, x2]
      ```
    - Inverse Jacobian log determinant:
      ```
      y = [y1, y2]
      log|det J(y)| = log|det df^{-1}/dy2(y2; g(y1))|
      ```
    Here, `[x1, x2]` is a partition of `x` along some axis. By default, `x1`
    remains unchanged and `x2` is transformed. If `swap=True`, `x2` will remain
    unchanged and `x1` will be transformed.
    """

    def __init__(self,
                 angles: Tensor,
                 conditioner,
                 bijector,
                 logger: Logger):
        """Initializes a SplitCoupling bijector.
        Args:
          conditioner: a function that computes the parameters of the inner bijector
            as a function of the unchanged part of the input. The output of the
            conditioner will be passed to `bijector` in order to obtain the inner
            bijector.
          bijector: a callable that returns the inner bijector that will be used to
            transform one of the two parts. The input to `bijector` is a set of
            parameters that can be used to configure the inner bijector. The
            `event_ndims_in` and `event_ndims_out` of the inner bijector must be
            equal, and less than or equal to `event_ndims`. If they are less than
            `event_ndims`, the remaining dimensions will be converted to event
            dimensions using `distrax.Block`.
        """
        super().__init__()
        self._angles = angles
        self._conditioner = conditioner
        self._bijector = bijector
        self._logger = logger

    @property
    def bijector(self):
        """The callable that returns the inner bijector of `SplitCoupling`."""
        return self._bijector

    @property
    def conditioner(self):
        """The conditioner function."""
        return self._conditioner

    def forward(self, x: Tensor, context=None) -> Tuple[Tensor, Tensor]:
        """Computes y = f(x) and log|det J(f)(x)|."""
        params = self._conditioner(self._angles)
        y, logdet = self._bijector(params).forward(x)
        return y, logdet

    def inverse(self, y: Tensor, context=None) -> Tuple[Tensor, Tensor]:
        """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
        params = self._conditioner(self._angles)
        x, logdet = self._bijector(params).inverse(y)
        return x, logdet
