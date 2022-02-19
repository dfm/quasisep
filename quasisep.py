from typing import Any, NamedTuple
from functools import partial

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
        if x.ndim == 1:
            x = x[:, None]
        return (
            self.diag[:, None] * x
            + self.lower.matmul(x, lower=True)
            + self.upper.matmul(x, lower=False)
        )

    def inv(self) -> "SquareQSM":
        d = self.diag
        p, q, a = self.lower
        h, g, b = self.upper

        def forward(carry, data):
            f = carry
            dk, pk, qk, ak, gk, hk, bk = data
            fhk = f @ hk
            fbk = f @ bk.T
            left = qk - ak @ fhk
            right = gk - pk @ fbk
            igk = 1 / (dk - pk @ fhk)
            sk = igk * left
            ellk = ak - jnp.outer(sk, pk)
            vk = igk * right
            delk = bk - jnp.outer(vk, hk)
            fk = ak @ fbk + igk * jnp.outer(left, right)
            return fk, (igk, sk, ellk, vk, delk)

        init = jnp.zeros_like(jnp.outer(q[0], g[0]))
        ig, s, ell, v, del_ = jax.lax.scan(forward, init, (d, p, q, a, g, h, b))[1]

        def backward(carry, data):
            z = carry
            igk, pk, ak, hk, bk, sk, vk = data
            zsk = z @ sk
            zak = z @ ak
            lk = igk + vk @ zsk
            tk = vk @ zak - lk * pk
            uk = bk.T @ zsk - lk * hk
            zk = bk.T @ zak - jnp.outer(uk + lk * hk, pk) - jnp.outer(hk, tk)
            return zk, (lk, tk, uk)

        init = jnp.zeros_like(jnp.outer(h[-1], p[-1]))
        args = (ig, p, a, h, b, s, v)
        lam, t, u = jax.lax.scan(backward, init, args, reverse=True)[1]

        return SquareQSM(
            diag=lam,
            lower=TriQSM(p=t, q=s, a=ell),
            upper=TriQSM(p=u, q=v, a=del_),
        )


def get_matmul_factor(q: JAXArray, a: JAXArray, x: JAXArray, reverse: bool) -> JAXArray:
    def impl(carry, data, *, transpose: bool):
        fp, qp, ap, xp = carry
        qn, an, xn = data
        if transpose:
            fn = ap.T @ fp + jnp.outer(qp, xp)
        else:
            fn = ap @ fp + jnp.outer(qp, xp)
        return (fn, qn, an, xn), fn

    q1 = q[0]
    a1 = a[0]
    x1 = jnp.zeros_like(x[0])
    f1 = jnp.zeros_like(jnp.outer(q1, x1))
    init = (f1, q1, a1, x1)
    args = (q, a, x)
    _, f = jax.lax.scan(partial(impl, transpose=reverse), init, args, reverse=reverse)
    return f
