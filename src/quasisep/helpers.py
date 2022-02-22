# -*- coding: utf-8 -*-
# mypy: ignore-errors

__all__ = ["handle_matvec_shapes", "qsm", "JAXArray"]

from functools import wraps
from typing import Any, Tuple

import jax.numpy as jnp

JAXArray = Any


def handle_matvec_shapes(func):
    @wraps(func)
    def wrapped(self, x, *args, **kwargs):
        vector = False
        if jnp.ndim(x) == 1:
            vector = True
            x = x[:, None]
        result = func(self, x, *args, **kwargs)
        if vector:
            return result[:, 0]
        return result

    return wrapped


def qsm(cls):
    def T(self):
        return self.transpose()

    def to_dense(self):
        return self.matmul(jnp.eye(self.shape[0]))

    def shape(self) -> Tuple[int, int]:
        n = self.diag.shape[0]
        return (n, n)

    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        return (-self).__add__(other)

    cls.T = property(T)
    cls.to_dense = to_dense
    if not hasattr(cls, "shape"):
        cls.shape = property(shape)
    cls.matmul = handle_matvec_shapes(cls.matmul)
    cls.is_qsm = True

    cls.__radd__ = __radd__
    cls.__rmul__ = __rmul__
    cls.__sub__ = __sub__
    cls.__rsub__ = __rsub__

    return cls
