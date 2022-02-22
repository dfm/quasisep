# -*- coding: utf-8 -*-

__all__ = [
    "StrictLowerTriQSM",
    "StrictUpperTriQSM",
    "LowerTriQSM",
    "UpperTriQSM",
    "SquareQSM",
    "SymmQSM",
]

from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp

from quasisep.helpers import JAXArray, handle_matvec_shapes, qsm


@qsm
class StrictLowerTriQSM(NamedTuple):
    p: JAXArray
    q: JAXArray
    a: JAXArray

    @property
    def shape(self) -> Tuple[int, int]:
        n = self.p.shape[0]
        return (n, n)

    def transpose(self) -> "StrictUpperTriQSM":
        return StrictUpperTriQSM(p=self.p, q=self.q, a=self.a)

    @jax.jit
    def matmul(self, x: JAXArray) -> JAXArray:
        def impl(f, data):  # type: ignore
            q, a, x = data
            return a @ f + jnp.outer(q, x), f

        init = jnp.zeros_like(jnp.outer(self.q[0], x[0]))
        _, f = jax.lax.scan(impl, init, (self.q, self.a, x))
        return jax.vmap(jnp.dot)(self.p, f)


@qsm
class StrictUpperTriQSM(NamedTuple):
    p: JAXArray
    q: JAXArray
    a: JAXArray

    @property
    def shape(self) -> Tuple[int, int]:
        n = self.p.shape[0]
        return (n, n)

    def transpose(self) -> "StrictLowerTriQSM":
        return StrictLowerTriQSM(p=self.p, q=self.q, a=self.a)

    @jax.jit
    def matmul(self, x: JAXArray) -> JAXArray:
        def impl(f, data):  # type: ignore
            p, a, x = data
            return a.T @ f + jnp.outer(p, x), f

        init = jnp.zeros_like(jnp.outer(self.p[-1], x[-1]))
        _, f = jax.lax.scan(impl, init, (self.p, self.a, x), reverse=True)
        return jax.vmap(jnp.dot)(self.q, f)


@qsm
class LowerTriQSM(NamedTuple):
    diag: JAXArray
    lower: StrictLowerTriQSM

    def transpose(self) -> "UpperTriQSM":
        return UpperTriQSM(diag=self.diag, upper=self.lower.transpose())

    @jax.jit
    def matmul(self, x: JAXArray) -> JAXArray:
        return self.diag[:, None] * x + self.lower.matmul(x)

    @jax.jit
    def inv(self) -> "LowerTriQSM":
        p, q, a = self.lower
        g = 1 / self.diag
        u = -g[:, None] * p
        v = g[:, None] * q
        b = a - jax.vmap(jnp.outer)(v, p)
        return LowerTriQSM(diag=g, lower=StrictLowerTriQSM(p=u, q=v, a=b))

    @jax.jit
    @handle_matvec_shapes
    def solve(self, y: JAXArray) -> JAXArray:
        def impl(fn, data):  # type: ignore
            (cn, (pn, wn, an)), yn = data
            xn = (yn - pn @ fn) / cn
            return an @ fn + jnp.outer(wn, xn), xn

        init = jnp.zeros_like(jnp.outer(self.lower.q[0], y[0]))
        _, x = jax.lax.scan(impl, init, (self, y))
        return x

    @jax.jit
    def qsmul(self, other: "SquareQSM") -> "SquareQSM":
        def calc_phi(phi, data):  # type: ignore
            a, b, q, g = data
            return a @ phi @ b.T + jnp.outer(q, g), phi

        init = jnp.zeros_like(jnp.outer(self.lower.q[0], other.upper.p[0]))
        args = (self.lower.a, other.upper.a, self.lower.q, other.upper.q)
        _, phi = jax.lax.scan(calc_phi, init, args)

        @jax.vmap
        def impl(self, other, phi) -> SquareQSM:  # type: ignore
            # Note: the order of g and h is flipped vs the paper!
            d1 = self.diag
            p1, q1, a1 = self.lower

            d2 = other.diag
            p2, q2, a2 = other.lower
            h2, g2, b2 = other.upper

            alpha = a1 @ phi @ h2 + q1 * d2
            theta = p1 @ phi @ b2.T + d1 * g2
            beta = d1 * p2

            s = jnp.concatenate((alpha, q2))
            v = theta
            t = jnp.concatenate((p1, beta))
            u = h2
            lam = p1 @ phi @ h2 + d1 * d2
            ell = jnp.concatenate(
                (
                    jnp.concatenate((a1, jnp.outer(q1, p2)), axis=-1),
                    jnp.concatenate(
                        (jnp.zeros((a2.shape[0], a1.shape[0])), a2), axis=-1
                    ),
                ),
                axis=0,
            )
            delta = b2
            return SquareQSM(
                diag=lam,
                lower=StrictLowerTriQSM(p=t, q=s, a=ell),
                upper=StrictUpperTriQSM(p=u, q=v, a=delta),
            )

        return impl(self, other, phi)


@qsm
class UpperTriQSM(NamedTuple):
    diag: JAXArray
    upper: StrictUpperTriQSM

    def transpose(self) -> "LowerTriQSM":
        return LowerTriQSM(diag=self.diag, lower=self.upper.transpose())

    @jax.jit
    def matmul(self, x: JAXArray) -> JAXArray:
        return self.diag[:, None] * x + self.upper.matmul(x)

    @jax.jit
    def inv(self) -> "UpperTriQSM":
        return self.transpose().inv().transpose()

    @jax.jit
    @handle_matvec_shapes
    def solve(self, y: JAXArray) -> JAXArray:
        def impl(fn, data):  # type: ignore
            (cn, (pn, wn, an)), yn = data
            xn = (yn - wn @ fn) / cn
            return an.T @ fn + jnp.outer(pn, xn), xn

        init = jnp.zeros_like(jnp.outer(self.upper.p[-1], y[-1]))
        _, x = jax.lax.scan(impl, init, (self, y), reverse=True)
        return x

    @jax.jit
    def qsmul(self, other: "SquareQSM") -> "SquareQSM":
        def calc_psi(psi, data):  # type: ignore
            a, b, q, g = data
            return a.T @ psi @ b + jnp.outer(q, g), psi

        init = jnp.zeros_like(jnp.outer(self.upper.q[-1], other.lower.p[-1]))
        args = (self.upper.a, other.lower.a, self.upper.p, other.lower.p)
        _, psi = jax.lax.scan(calc_psi, init, args, reverse=True)

        @jax.vmap
        def impl(self, other, psi) -> SquareQSM:  # type: ignore
            # Note: the order of g and h is flipped vs the paper!
            d1 = self.diag
            h1, g1, b1 = self.upper

            d2 = other.diag
            p2, q2, a2 = other.lower
            h2, g2, b2 = other.upper

            theta = d1 * g2
            beta = d1 * p2 + g1 @ psi @ a2
            eta = h1 * d2 + b1.T @ psi @ q2

            s = q2
            v = jnp.concatenate((g1, theta))
            t = beta
            u = jnp.concatenate((eta, h2))
            lam = d1 * d2 + g1 @ psi @ q2
            ell = a2
            delta = jnp.concatenate(
                (
                    jnp.concatenate(
                        (b1, jnp.zeros((b1.shape[0], b2.shape[0]))), axis=-1
                    ),
                    jnp.concatenate((jnp.outer(g2, h1), b2), axis=-1),
                ),
                axis=0,
            )
            return SquareQSM(
                diag=lam,
                lower=StrictLowerTriQSM(p=t, q=s, a=ell),
                upper=StrictUpperTriQSM(p=u, q=v, a=delta),
            )

        return impl(self, other, psi)


@qsm
class SquareQSM(NamedTuple):
    diag: JAXArray
    lower: StrictLowerTriQSM
    upper: StrictUpperTriQSM

    def transpose(self) -> "SquareQSM":
        return SquareQSM(
            diag=self.diag,
            lower=self.upper.transpose(),
            upper=self.lower.transpose(),
        )

    @jax.jit
    def matmul(self, x: JAXArray) -> JAXArray:
        return (
            self.diag[:, None] * x
            + self.lower.matmul(x)
            + self.upper.matmul(x)
        )

    @jax.jit
    def qsmul(self, other: "SquareQSM") -> "SquareQSM":
        def calc_phi(phi, data):  # type: ignore
            a, b, q, g = data
            return a @ phi @ b.T + jnp.outer(q, g), phi

        def calc_psi(psi, data):  # type: ignore
            a, b, q, g = data
            return a.T @ psi @ b + jnp.outer(q, g), psi

        init = jnp.zeros_like(jnp.outer(self.lower.q[0], other.upper.p[0]))
        args = (self.lower.a, other.upper.a, self.lower.q, other.upper.q)
        _, phi = jax.lax.scan(calc_phi, init, args)

        init = jnp.zeros_like(jnp.outer(self.upper.q[-1], other.lower.p[-1]))
        args = (self.upper.a, other.lower.a, self.upper.p, other.lower.p)
        _, psi = jax.lax.scan(calc_psi, init, args, reverse=True)

        @jax.vmap
        def impl(self, other, phi, psi) -> SquareQSM:  # type: ignore
            # Note: the order of g and h is flipped vs the paper!
            d1 = self.diag
            p1, q1, a1 = self.lower
            h1, g1, b1 = self.upper

            d2 = other.diag
            p2, q2, a2 = other.lower
            h2, g2, b2 = other.upper

            alpha = a1 @ phi @ h2 + q1 * d2
            theta = p1 @ phi @ b2.T + d1 * g2
            beta = d1 * p2 + g1 @ psi @ a2
            eta = h1 * d2 + b1.T @ psi @ q2

            s = jnp.concatenate((alpha, q2))
            v = jnp.concatenate((g1, theta))
            t = jnp.concatenate((p1, beta))
            u = jnp.concatenate((eta, h2))
            lam = p1 @ phi @ h2 + d1 * d2 + g1 @ psi @ q2
            ell = jnp.concatenate(
                (
                    jnp.concatenate((a1, jnp.outer(q1, p2)), axis=-1),
                    jnp.concatenate(
                        (jnp.zeros((a2.shape[0], a1.shape[0])), a2), axis=-1
                    ),
                ),
                axis=0,
            )
            delta = jnp.concatenate(
                (
                    jnp.concatenate(
                        (b1, jnp.zeros((b1.shape[0], b2.shape[0]))), axis=-1
                    ),
                    jnp.concatenate((jnp.outer(g2, h1), b2), axis=-1),
                ),
                axis=0,
            )
            return SquareQSM(
                diag=lam,
                lower=StrictLowerTriQSM(p=t, q=s, a=ell),
                upper=StrictUpperTriQSM(p=u, q=v, a=delta),
            )

        return impl(self, other, phi, psi)

    @jax.jit
    def gram(self) -> "SymmQSM":
        def calc_phi(phi, data):  # type: ignore
            a, q = data
            return a @ phi @ a.T + jnp.outer(q, q), phi

        def calc_psi(psi, data):  # type: ignore
            a, p = data
            return a.T @ psi @ a + jnp.outer(p, p), psi

        init = jnp.zeros_like(jnp.outer(self.upper.p[0], self.upper.p[0]))
        args = (self.upper.a, self.upper.q)
        _, phi = jax.lax.scan(calc_phi, init, args)

        init = jnp.zeros_like(jnp.outer(self.lower.p[-1], self.lower.p[-1]))
        args = (self.lower.a, self.lower.p)
        _, psi = jax.lax.scan(calc_psi, init, args, reverse=True)

        @jax.vmap
        def impl(self, phi, psi) -> SymmQSM:  # type: ignore
            # Note: the order of g and h is flipped vs the paper!
            d = self.diag
            p, q, a = self.lower
            h, g, b = self.upper

            alpha = b @ phi @ h + g * d
            beta = d * p + q @ psi @ a

            s = jnp.concatenate((alpha, q))
            t = jnp.concatenate((h, beta))
            lam = h @ phi @ h + jnp.square(d) + q @ psi @ q
            ell = jnp.concatenate(
                (
                    jnp.concatenate((b, jnp.outer(g, p)), axis=-1),
                    jnp.concatenate(
                        (jnp.zeros((a.shape[0], b.shape[0])), a), axis=-1
                    ),
                ),
                axis=0,
            )
            return SymmQSM(diag=lam, lower=StrictLowerTriQSM(p=t, q=s, a=ell))

        return impl(self, phi, psi)

    @jax.jit
    def inv(self) -> "SquareQSM":
        d = self.diag
        p, q, a = self.lower
        h, g, b = self.upper

        def forward(carry, data):  # type: ignore
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
        ig, s, ell, v, del_ = jax.lax.scan(
            forward, init, (d, p, q, a, g, h, b)
        )[1]

        def backward(carry, data):  # type: ignore
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
            lower=StrictLowerTriQSM(p=t, q=s, a=ell),
            upper=StrictUpperTriQSM(p=u, q=v, a=del_),
        )


@qsm
class SymmQSM(NamedTuple):
    diag: JAXArray
    lower: StrictLowerTriQSM

    def transpose(self) -> "SymmQSM":
        return self

    @jax.jit
    def matmul(self, x: JAXArray) -> JAXArray:
        return (
            self.diag[:, None] * x
            + self.lower.matmul(x)
            + self.lower.transpose().matmul(x)
        )

    @jax.jit
    def inv(self) -> "SymmQSM":
        d = self.diag
        p, q, a = self.lower

        def forward(carry, data):  # type: ignore
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

        def backward(carry, data):  # type: ignore
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
        return SymmQSM(diag=lam, lower=StrictLowerTriQSM(p=t, q=s, a=ell))

    @jax.jit
    def cholesky(self) -> LowerTriQSM:
        d = self.diag
        p, q, a = self.lower

        def impl(carry, data):  # type: ignore
            fp = carry
            dk, pk, qk, ak = data
            ck = jnp.sqrt(dk - pk @ fp @ pk)
            tmp = fp @ ak.T
            wk = (qk - pk @ tmp) / ck
            fk = ak @ tmp + jnp.outer(wk, wk)
            return fk, (ck, wk)

        init = jnp.zeros_like(jnp.outer(q[0], q[0]))
        _, (c, w) = jax.lax.scan(impl, init, (d, p, q, a))
        return LowerTriQSM(diag=c, lower=StrictLowerTriQSM(p=p, q=w, a=a))
