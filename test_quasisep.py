import pytest
import numpy as np
import jax.numpy as jnp

from quasisep import StrictTriQSM, TriQSM, SquareQSM, SymmQSM

from jax.config import config

config.update("jax_enable_x64", True)


@pytest.fixture(params=["random", "celerite"])
def matrices(request):
    N = 100
    random = np.random.default_rng(1234)
    diag = np.exp(random.normal(size=N))

    if request.param == "random":
        J = 5
        p = random.normal(size=(N, J))
        q = random.normal(size=(N, J))
        a = np.repeat(np.eye(J)[None, :, :], N, axis=0)
        l = np.tril(p @ q.T, -1)
        u = np.triu(q @ p.T, 1)
        diag += np.sum(p * q, axis=1)

    elif request.param == "celerite":
        t = np.sort(random.uniform(0, 10, N))

        a = np.array([1.0, 2.5])
        b = np.array([0.5, 1.5])
        c = np.array([1.2, 0.5])
        d = np.array([0.5, 0.1])

        tau = np.abs(t[:, None] - t[None, :])[:, :, None]
        K = np.sum(
            np.exp(-c[None, None] * tau)
            * (
                a[None, None] * np.cos(d[None, None] * tau)
                + b[None, None] * np.sin(d[None, None] * tau)
            ),
            axis=-1,
        )
        K[np.diag_indices_from(K)] += diag
        diag = np.diag(K)
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
    return diag, p, q, a, v, m, l, u


def test_strict_tri_matmul(matrices):
    _, p, q, a, v, m, l, u = matrices
    N = len(p)
    mat = StrictTriQSM(p=p, q=q, a=a)

    # Check multiplication into identity / to dense
    np.testing.assert_allclose(mat.matmul(np.eye(N)), l)
    np.testing.assert_allclose(mat.matmul(np.eye(N), lower=False), u)

    # Check matvec
    np.testing.assert_allclose(mat.matmul(v), l @ v[:, None])
    np.testing.assert_allclose(mat.matmul(v, lower=False), u @ v[:, None])

    # Check matmat
    np.testing.assert_allclose(mat.matmul(m), l @ m)
    np.testing.assert_allclose(mat.matmul(m, lower=False), u @ m)


def test_tri_matmul(matrices):
    diag, p, q, a, v, m, l, _ = matrices
    N = len(p)
    mat = TriQSM(diag=diag, lower=StrictTriQSM(p=p, q=q, a=a))
    dense = l + np.diag(diag)

    # Check multiplication into identity / to dense
    np.testing.assert_allclose(mat.matmul(np.eye(N)), dense)
    np.testing.assert_allclose(mat.matmul(np.eye(N), lower=False), dense.T)

    # Check matvec
    np.testing.assert_allclose(mat.matmul(v), dense @ v[:, None])
    np.testing.assert_allclose(mat.matmul(v, lower=False), dense.T @ v[:, None])

    # Check matmat
    np.testing.assert_allclose(mat.matmul(m), dense @ m)
    np.testing.assert_allclose(mat.matmul(m, lower=False), dense.T @ m)


@pytest.mark.parametrize("symm", [True, False])
def test_square_matmul(symm, matrices):
    diag, p, q, a, v, m, l, u = matrices
    N = len(p)
    if symm:
        mat = SymmQSM(diag=diag, lower=StrictTriQSM(p=p, q=q, a=a))
    else:
        mat = SquareQSM(
            diag=diag,
            lower=StrictTriQSM(p=p, q=q, a=a),
            upper=StrictTriQSM(p=p, q=q, a=a),
        )

    # Create and double check the dense reconstruction
    dense = mat.matmul(np.eye(N))
    np.testing.assert_allclose(np.tril(dense, -1), l)
    np.testing.assert_allclose(np.triu(dense, 1), u)
    np.testing.assert_allclose(np.diag(dense), diag)

    # Test matmuls
    np.testing.assert_allclose(mat.matmul(v), dense @ v[:, None])
    np.testing.assert_allclose(mat.matmul(m), dense @ m)


@pytest.mark.parametrize("symm", [True, False])
def test_square_inv(symm, matrices):
    diag, p, q, a, _, _, l, u = matrices
    N = len(p)
    if symm:
        mat = SymmQSM(diag=diag, lower=StrictTriQSM(p=p, q=q, a=a))
    else:
        mat = SquareQSM(
            diag=diag,
            lower=StrictTriQSM(p=p, q=q, a=a),
            upper=StrictTriQSM(p=p, q=q, a=a),
        )

    # Create and double check the dense reconstruction
    dense = mat.matmul(np.eye(N))
    np.testing.assert_allclose(np.tril(dense, -1), l)
    np.testing.assert_allclose(np.triu(dense, 1), u)
    np.testing.assert_allclose(np.diag(dense), diag)

    # Invert the QS matrix
    minv = mat.inv()
    np.testing.assert_allclose(minv.matmul(np.eye(N)), jnp.linalg.inv(dense), rtol=2e-6)
    np.testing.assert_allclose(minv.matmul(dense), np.eye(N), atol=1e-12)

    # In this case, we know our matrix to be symmetric - so should its inverse be!
    # This may change in the future as we expand test cases
    if not symm:
        np.testing.assert_allclose(minv.lower.p, minv.upper.p)
        np.testing.assert_allclose(minv.lower.q, minv.upper.q)
        np.testing.assert_allclose(minv.lower.a, minv.upper.a)

    # The inverse of the inverse should be itself... don't actually do this!
    # Note: we can't actually directly compare the generators because there's
    # enough degrees of freedom that they won't necessarily round trip. It's
    # good enough to check that it produces the correct dense reconstruction.
    mat2 = minv.inv()
    np.testing.assert_allclose(mat2.matmul(np.eye(N)), dense, rtol=1e-4)
