####################################################################################################
#                                          test_backend.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-09-03                                                                              #
#                                                                                                  #
# Purpose: Tests for the nifti_mrs_plus.backend module — jit and value_and_grad dispatched at      #
#          call time, covering all available frameworks plus the numpy behaviour.                  #
#                                                                                                  #
####################################################################################################

import pytest
import numpy as np

from nifti_mrs_plus.backend import jit, value_and_grad


X_NP = np.linspace(0.1, 1.0, 16).astype(np.float32)


def _loss(x):
    return (x * x).sum()


def _fd(fn, x, eps=1e-3):
    grad = np.zeros_like(x)
    for i in range(x.size):
        up, down = x.copy(), x.copy()
        up[i] += eps
        down[i] -= eps
        grad[i] = (fn(up) - fn(down)) / (2 * eps)
    return grad


# ── NumPy ─────────────────────────────────────────────────────────────────────

class TestBackendNumpy:

    def test_jit_runs_uncompiled(self):
        assert np.isclose(jit(_loss)(X_NP), _loss(X_NP))

    def test_value_and_grad_raises(self):
        with pytest.raises(NotImplementedError, match="first argument"):
            value_and_grad(_loss)(X_NP)


# ── PyTorch ───────────────────────────────────────────────────────────────────

class TestBackendTorch:

    def setup_method(self):
        self.torch = pytest.importorskip("torch")

    def _x(self):
        return self.torch.from_numpy(X_NP.copy())

    def test_value_and_grad(self):
        value, grad = value_and_grad(_loss)(self._x())
        assert np.isclose(float(value), _loss(X_NP), atol=1e-5)
        np.testing.assert_allclose(grad.numpy(), _fd(_loss, X_NP.astype(np.float64)), atol=1e-2)

    def test_jit_compiles_and_matches(self):
        assert np.isclose(float(jit(_loss)(self._x())), _loss(X_NP), atol=1e-5)


# ── JAX ───────────────────────────────────────────────────────────────────────

class TestBackendJax:

    def setup_method(self):
        self.jnp = pytest.importorskip("jax.numpy")

    def _x(self):
        return self.jnp.asarray(X_NP)

    def test_value_and_grad(self):
        value, grad = value_and_grad(_loss)(self._x())
        assert np.isclose(float(value), _loss(X_NP), atol=1e-5)
        np.testing.assert_allclose(np.asarray(grad), _fd(_loss, X_NP.astype(np.float64)), atol=1e-2)

    def test_jit_compiles_and_matches(self):
        assert np.isclose(float(jit(_loss)(self._x())), _loss(X_NP), atol=1e-5)


# ── TensorFlow ────────────────────────────────────────────────────────────────

class TestBackendTensorflow:

    def setup_method(self):
        self.tf = pytest.importorskip("tensorflow")

    def _x(self):
        return self.tf.constant(X_NP)

    def test_value_and_grad(self):
        value, grad = value_and_grad(lambda x: self.tf.reduce_sum(x * x))(self._x())
        assert np.isclose(float(value), _loss(X_NP), atol=1e-5)
        np.testing.assert_allclose(grad.numpy(), _fd(_loss, X_NP.astype(np.float64)), atol=1e-2)

    def test_jit_compiles_and_matches(self):
        fn = jit(lambda x: self.tf.reduce_sum(x * x))
        assert np.isclose(float(fn(self._x())), _loss(X_NP), atol=1e-5)


# ── Coexistence ───────────────────────────────────────────────────────────────

class TestCoexistence:

    def test_one_wrapper_serves_multiple_frameworks(self):
        jnp = pytest.importorskip("jax.numpy")
        fn = jit(_loss)
        assert np.isclose(float(fn(jnp.asarray(X_NP))), _loss(X_NP), atol=1e-5)
        assert np.isclose(fn(X_NP), _loss(X_NP), atol=1e-5)
