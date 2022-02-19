# -*- coding: utf-8 -*-

__all__ = ["StrictTriQSM", "TriQSM", "SquareQSM", "SymmQSM"]

from functools import partial, wraps
from typing import Any, NamedTuple

import jax
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


class StrictTriQSM(NamedTuple):
    p: JAXArray
    q: JAXArray
    a: JAXArray

    def to_dense(self, *, lower: bool = True):
        return self.matmul(jnp.eye(len(self.p)), lower=lower)

    @partial(jax.jit, static_argnames=["lower"])
    @handle_matvec_shapes
    def matmul(self, x: JAXArray, *, lower: bool = True) -> JAXArray:
        if lower:
            f = get_matmul_factor(self.q, self.a, x, False)
            return jax.vmap(jnp.dot)(self.p, f)
        else:
            f = get_matmul_factor(self.p, self.a, x, True)
            return jax.vmap(jnp.dot)(self.q, f)


class TriQSM(NamedTuple):
    diag: JAXArray
    lower: StrictTriQSM

    def to_dense(self, *, lower: bool = True):
        return self.matmul(jnp.eye(len(self.diag)), lower=lower)

    @partial(jax.jit, static_argnames=["lower"])
    @handle_matvec_shapes
    def matmul(self, x: JAXArray, *, lower: bool = True) -> JAXArray:
        return self.diag[:, None] * x + self.lower.matmul(x, lower=lower)

    @jax.jit
    def inv(self) -> "TriQSM":
        p, q, a = self.lower
        g = 1 / self.diag
        u = -g[:, None] * p
        v = g[:, None] * q
        b = a - jax.vmap(jnp.outer)(v, p)
        return TriQSM(diag=g, lower=StrictTriQSM(p=u, q=v, a=b))


class SquareQSM(NamedTuple):
    diag: JAXArray
    lower: StrictTriQSM
    upper: StrictTriQSM

    def to_dense(self):
        return self.matmul(jnp.eye(len(self.diag)))

    @jax.jit
    @handle_matvec_shapes
    def matmul(self, x: JAXArray) -> JAXArray:
        return (
            self.diag[:, None] * x
            + self.lower.matmul(x, lower=True)
            + self.upper.matmul(x, lower=False)
        )

    def qsmul(self, other: "SquareQSM") -> "SquareQSM":
        d1 = self.diag
        p1, q1, _ = self.lower
        g1, h1, _ = self.upper

        d2 = other.diag
        p2, q2, _ = other.lower
        g2, h2, _ = other.upper

        def forward(carry, data):
            phi = carry
            (d1, (p1, q1, a1), _), (d2, _, (g2, h2, b2)) = data
            alpha = a1 @ phi @ h2 + q1 * d2
            theta = p1 @ phi @ b2.T + d1 * g2
            return a1 @ phi @ b2.T + jnp.outer(q1, g2), (phi, alpha, theta)

        init = jnp.zeros_like(jnp.outer(q1[0], g2[0]))
        phi, alpha, theta = jax.lax.scan(forward, init, (self, other))[1]
        s = jnp.concatenate((alpha, q2), axis=-1)
        v = jnp.concatenate((g1, theta), axis=-1)

        def backward(carry, data):
            psi = carry
            (d1, _, (g1, h1, b1)), (d2, (p2, q2, a2), _) = data
            beta = d1 * p2 + g1 @ psi @ a2
            gamma = h1 * d2 + b1.T @ psi @ q2.T
            return b1.T @ psi @ a2 + jnp.outer(h1, p2), (psi, beta, gamma)

        init = jnp.zeros_like(jnp.outer(h1[-1], p2[-1]))
        psi, beta, gamma = jax.lax.scan(backward, init, (self, other), reverse=True)[1]
        t = jnp.concatenate((p1, beta), axis=-1)
        u = jnp.concatenate((gamma, h2), axis=-1)

        def calc(self, other, phi, psi):
            d1 = self.diag
            p1, q1, a1 = self.lower
            g1, h1, b1 = self.upper

            d2 = other.diag
            p2, q2, a2 = other.lower
            g2, h2, b2 = other.upper

            lam = p1 @ phi @ h2 + d1 * d2 + g1 @ psi @ q2
            ell = jnp.concatenate(
                (
                    jnp.concatenate((a1, jnp.outer(q1, p2)), axis=-1),
                    jnp.concatenate(
                        (jnp.zeros((a2.shape[0], a1.shape[1])), a2), axis=-1
                    ),
                ),
                axis=0,
            )
            delta = jnp.concatenate(
                (
                    jnp.concatenate((b1, jnp.outer(h1, g2)), axis=-1),
                    jnp.concatenate(
                        (jnp.zeros((b2.shape[0], b1.shape[1])), b2), axis=-1
                    ),
                ),
                axis=0,
            )
            return lam, ell, delta

        lam, ell, delta = jax.vmap(calc)(self, other, phi, psi)

        return SquareQSM(
            diag=lam,
            lower=StrictTriQSM(p=t, q=s, a=ell),
            upper=StrictTriQSM(p=v, q=u, a=delta),
        )

    @jax.jit
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
            lower=StrictTriQSM(p=t, q=s, a=ell),
            upper=StrictTriQSM(p=u, q=v, a=del_),
        )


class SymmQSM(NamedTuple):
    diag: JAXArray
    lower: StrictTriQSM

    def to_dense(self):
        return self.matmul(jnp.eye(len(self.diag)))

    @jax.jit
    @handle_matvec_shapes
    def matmul(self, x: JAXArray) -> JAXArray:
        return (
            self.diag[:, None] * x
            + self.lower.matmul(x, lower=True)
            + self.lower.matmul(x, lower=False)
        )

    @jax.jit
    def inv(self) -> "SymmQSM":
        d = self.diag
        p, q, a = self.lower

        def forward(carry, data):
            f = carry
            dk, pk, qk, ak = data
            fpk = f @ pk
            left = qk - ak @ fpk
            igk = 1 / (dk - pk @ fpk)
            sk = igk * left
            ellk = ak - jnp.outer(sk, pk)
            fk = ak @ f @ ak.T + igk * jnp.outer(left, left.T)
            return fk, (igk, sk, ellk)

        init = jnp.zeros_like(jnp.outer(q[0], q[0]))
        ig, s, ell = jax.lax.scan(forward, init, (d, p, q, a))[1]

        def backward(carry, data):
            z = carry
            igk, pk, ak, sk = data
            zak = z @ ak
            skzak = sk @ zak
            lk = igk + sk @ z @ sk
            tk = skzak - lk * pk
            zk = ak.T @ zak - jnp.outer(skzak, pk) - jnp.outer(pk, tk)
            return zk, (lk, tk)

        init = jnp.zeros_like(jnp.outer(p[-1], p[-1]))
        lam, t = jax.lax.scan(backward, init, (ig, p, a, s), reverse=True)[1]
        return SymmQSM(diag=lam, lower=StrictTriQSM(p=t, q=s, a=ell))

    @jax.jit
    def cholesky(self) -> TriQSM:
        d = self.diag
        p, q, a = self.lower

        def impl(carry, data):
            fp = carry
            dk, pk, qk, ak = data
            ck = jnp.sqrt(dk - pk @ fp @ pk)
            tmp = fp @ ak.T
            wk = (qk - pk @ tmp) / ck
            fk = ak @ tmp + jnp.outer(wk, wk)
            return fk, (ck, wk)

        init = jnp.zeros_like(jnp.outer(q[0], q[0]))
        _, (c, w) = jax.lax.scan(impl, init, (d, p, q, a))
        return TriQSM(diag=c, lower=StrictTriQSM(p=p, q=w, a=a))


@partial(jax.jit, static_argnames=["reverse"])
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
