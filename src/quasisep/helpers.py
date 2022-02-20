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

    cls.T = property(T)
    cls.to_dense = to_dense
    if not hasattr(cls, "shape"):
        cls.shape = property(shape)
    cls.matmul = handle_matvec_shapes(cls.matmul)
    return cls
