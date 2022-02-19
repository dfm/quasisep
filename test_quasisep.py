import pytest
import numpy as np
import jax
import jax.numpy as jnp

from quasisep import TriQSM

from jax.config import config

config.update("jax_enable_x64", True)


@pytest.fixture(params=["random", "celerite"])
def matrices(request):
    N = 100
    random = np.random.default_rng(1234)

    if request.param == "random":
        J = 5
        p = random.normal(size=(N, J))
        q = random.normal(size=(N, J))
        a = np.repeat(np.eye(J)[None, :, :], N, axis=0)
        l = np.tril(p @ q.T, -1)
        u = np.triu(q @ p.T, 1)

    elif request.param == "celerite":
        t = np.sort(random.uniform(0, 10, N))

        a = jnp.array([1.0, 2.5])
        b = jnp.array([0.5, 1.5])
        c = jnp.array([1.2, 0.5])
        d = jnp.array([0.5, 0.1])

        tau = np.abs(t[:, None] - t[None, :])[:, :, None]
        K = np.sum(
            np.exp(-c[None, None] * tau)
            * (
                a[None, None] * np.cos(d[None, None] * tau)
                + b[None, None] * np.sin(d[None, None] * tau)
            ),
            axis=-1,
        )
        l = np.tril(K, -1)
        u = np.triu(K, 1)

        cos = np.cos(d[None] * t[:, None])
        sin = np.sin(d[None] * t[:, None])
        p = np.concatenate(
            (
                a[None] * cos + b[None] * sin,
                a[None] * sin - b[None] * cos,
            ),
            axis=1,
        )
        q = np.concatenate((cos, sin), axis=1)
        c = np.append(c, c)
        dt = np.append(np.diff(t), 0)
        a = np.stack([np.diag(v) for v in np.exp(-c[None] * dt[:, None])], axis=0)
        q = np.einsum("nji,ni->nj", a, q)

    else:
        assert False

    v = random.normal(size=N)
    m = random.normal(size=(N, 4))
    return p, q, a, v, m, l, u


def test_tri_matmul(matrices):
    p, q, a, v, m, l, u = matrices
    N = len(p)
    mat = TriQSM(p=p, q=q, a=a)

    # Check multiplication into identity / to dense
    np.testing.assert_allclose(mat.matmul(np.eye(N)), l)
    np.testing.assert_allclose(mat.matmul(np.eye(N), lower=False), u)

    # Check matvec
    np.testing.assert_allclose(mat.matmul(v), l @ v[:, None])
    np.testing.assert_allclose(mat.matmul(v, lower=False), u @ v[:, None])

    # Check matmat
    np.testing.assert_allclose(mat.matmul(m), l @ m)
    np.testing.assert_allclose(mat.matmul(m, lower=False), u @ m)
