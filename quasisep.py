from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

JAXArray = Any


class TriQSM(NamedTuple):
    p: JAXArray
    q: JAXArray
    a: JAXArray

    def matmul(self, x: JAXArray, *, lower: bool = True) -> JAXArray:
        if lower:
            f = get_matmul_factor(self.q, self.a, x, False)
            return jax.vmap(jnp.dot)(self.p, f)
        else:
            f = get_matmul_factor(self.p, self.a, x, True)
            return jax.vmap(jnp.dot)(self.q, f)


class SquareQSM(NamedTuple):
    diag: JAXArray
    lower: TriQSM
    upper: TriQSM

    def matmul(self, x: JAXArray) -> JAXArray:
        return (
            self.diag * x
            + self.lower.matmul(x, lower=True)
            + self.upper.matmul(x, lower=False)
        )


def get_matmul_factor(q: JAXArray, a: JAXArray, x: JAXArray, reverse: bool) -> JAXArray:
    def impl(carry, data):
        fp, qp, ap, xp = carry
        qn, an, xn = data
        fn = ap @ fp + jnp.outer(qp, xp)
        return (fn, qn, an, xn), fn

    q1 = q[0]
    a1 = a[0]
    x1 = jnp.zeros_like(x[0])
    f1 = jnp.zeros_like(jnp.outer(q1, x1))
    init = (f1, q1, a1, x1)
    args = (q, a, x)
    return jax.lax.scan(impl, init, args, reverse=reverse)[1]
