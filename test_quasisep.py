import jax.numpy as jnp
import numpy as np
import pytest
from jax.config import config

from quasisep import SquareQSM, StrictTriQSM, SymmQSM, TriQSM

config.update("jax_enable_x64", True)


@pytest.fixture(params=["random", "celerite"])
def name(request):
    return request.param


@pytest.fixture
def matrices(name):
    return get_matrices(name)


def get_matrices(name):
    N = 100
    random = np.random.default_rng(1234)
    diag = np.exp(random.normal(size=N))

    if name == "random":
        J = 5
        p = random.normal(size=(N, J))
        q = random.normal(size=(N, J))
        a = np.repeat(np.eye(J)[None, :, :], N, axis=0)
        l = np.tril(p @ q.T, -1)
        u = np.triu(q @ p.T, 1)
        diag += np.sum(p * q, axis=1)

    elif name == "celerite":
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
        dt = np.append(0, np.diff(t))
        a = np.stack([np.diag(v) for v in np.exp(-c[None] * dt[:, None])], axis=0)
        p = np.einsum("ni,nij->nj", p, a)

    else:
        assert False

    v = random.normal(size=N)
    m = random.normal(size=(N, 4))
    return diag, p, q, a, v, m, l, u


def test_strict_tri_matmul(matrices):
    _, p, q, a, v, m, l, u = matrices
    mat = StrictTriQSM(p=p, q=q, a=a)

    # Check multiplication into identity / to dense
    np.testing.assert_allclose(mat.to_dense(), l)
    np.testing.assert_allclose(mat.to_dense(lower=False), u)

    # Check matvec
    np.testing.assert_allclose(mat.matmul(v), l @ v)
    np.testing.assert_allclose(mat.matmul(v, lower=False), u @ v)

    # Check matmat
    np.testing.assert_allclose(mat.matmul(m), l @ m)
    np.testing.assert_allclose(mat.matmul(m, lower=False), u @ m)


def test_tri_matmul(matrices):
    diag, p, q, a, v, m, l, _ = matrices
    mat = TriQSM(diag=diag, lower=StrictTriQSM(p=p, q=q, a=a))
    dense = l + np.diag(diag)

    # Check multiplication into identity / to dense
    np.testing.assert_allclose(mat.to_dense(), dense)
    np.testing.assert_allclose(mat.to_dense(lower=False), dense.T)

    # Check matvec
    np.testing.assert_allclose(mat.matmul(v), dense @ v)
    np.testing.assert_allclose(mat.matmul(v, lower=False), dense.T @ v)

    # Check matmat
    np.testing.assert_allclose(mat.matmul(m), dense @ m)
    np.testing.assert_allclose(mat.matmul(m, lower=False), dense.T @ m)


@pytest.mark.parametrize("symm", [True, False])
def test_square_matmul(symm, matrices):
    diag, p, q, a, v, m, l, u = matrices
    if symm:
        mat = SymmQSM(diag=diag, lower=StrictTriQSM(p=p, q=q, a=a))
    else:
        mat = SquareQSM(
            diag=diag,
            lower=StrictTriQSM(p=p, q=q, a=a),
            upper=StrictTriQSM(p=p, q=q, a=a),
        )

    # Create and double check the dense reconstruction
    dense = mat.to_dense()
    np.testing.assert_allclose(np.tril(dense, -1), l)
    np.testing.assert_allclose(np.triu(dense, 1), u)
    np.testing.assert_allclose(np.diag(dense), diag)

    # Test matmuls
    np.testing.assert_allclose(mat.matmul(v), dense @ v)
    np.testing.assert_allclose(mat.matmul(m), dense @ m)


@pytest.mark.parametrize("name", ["celerite"])
def test_tri_inv(matrices):
    diag, p, q, a, _, _, _, _ = matrices
    mat = TriQSM(diag=diag, lower=StrictTriQSM(p=p, q=q, a=a))
    dense = mat.to_dense()
    minv = mat.inv()
    np.testing.assert_allclose(minv.to_dense(), jnp.linalg.inv(dense))
    np.testing.assert_allclose(minv.matmul(dense), np.eye(len(diag)), atol=1e-12)


@pytest.mark.parametrize("symm", [True, False])
def test_square_inv(symm, matrices):
    diag, p, q, a, _, _, l, u = matrices
    if symm:
        mat = SymmQSM(diag=diag, lower=StrictTriQSM(p=p, q=q, a=a))
    else:
        mat = SquareQSM(
            diag=diag,
            lower=StrictTriQSM(p=p, q=q, a=a),
            upper=StrictTriQSM(p=p, q=q, a=a),
        )

    # Create and double check the dense reconstruction
    dense = mat.to_dense()
    np.testing.assert_allclose(np.tril(dense, -1), l)
    np.testing.assert_allclose(np.triu(dense, 1), u)
    np.testing.assert_allclose(np.diag(dense), diag)

    # Invert the QS matrix
    minv = mat.inv()
    np.testing.assert_allclose(minv.to_dense(), jnp.linalg.inv(dense), rtol=2e-6)
    np.testing.assert_allclose(minv.matmul(dense), np.eye(len(diag)), atol=1e-12)

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
    np.testing.assert_allclose(mat2.to_dense(), dense, rtol=1e-4)


@pytest.mark.parametrize("name", ["celerite"])
def test_cholesky(matrices):
    diag, p, q, a, _, _, _, _ = matrices
    mat = SymmQSM(diag=diag, lower=StrictTriQSM(p=p, q=q, a=a))
    dense = mat.to_dense()
    chol = mat.cholesky()
    np.testing.assert_allclose(chol.to_dense(), np.linalg.cholesky(dense))

    mat = mat.inv()
    dense = mat.to_dense()
    chol = mat.cholesky()
    np.testing.assert_allclose(chol.to_dense(), np.linalg.cholesky(dense))


def test_qsmul():
    diag, p, q, a, _, _, _, _ = get_matrices("celerite")
    mat1 = SquareQSM(
        diag=diag,
        lower=StrictTriQSM(p=p, q=q, a=a),
        upper=StrictTriQSM(p=p, q=q, a=a),
    )

    diag, p, q, a, _, _, _, _ = get_matrices("random")
    mat2 = SquareQSM(
        diag=diag,
        lower=StrictTriQSM(p=p, q=q, a=a),
        upper=StrictTriQSM(p=p, q=q, a=a),
    )

    mat = mat1.qsmul(mat2)
    print(mat.to_dense())
    print(mat1.to_dense() @ mat2.to_dense())
    assert 0
