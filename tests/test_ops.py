####################################################################################################
#                                          test_ops.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-06-27                                                                              #
#                                                                                                  #
# Purpose: Tests for the nifti_mrs_plus.ops module — fft/ifft/fftshift, to_numpy, and              #
#          match_backend, covering all available array backends.                                   #
#                                                                                                  #
####################################################################################################

import pytest
import numpy as np

from nifti_mrs_plus.ops import fft, ifft, fftshift, ifftshift, to_numpy, match_backend


N = 128
_rng = np.random.default_rng(42)
X_NP = _rng.random(N).astype(np.float32) + 1j * _rng.random(N).astype(np.float32)


# ── NumPy ─────────────────────────────────────────────────────────────────────

class TestBackendsNumpy:

    def test_fft_roundtrip(self):
        out = ifft(fft(X_NP))
        np.testing.assert_allclose(np.real(out), np.real(X_NP), atol=1e-5)

    def test_fftshift_roundtrip(self):
        np.testing.assert_allclose(ifftshift(fftshift(X_NP)), X_NP, atol=1e-7)

    def test_fft_shape(self):
        assert fft(X_NP).shape == X_NP.shape

    def test_to_numpy_passthrough(self):
        out = to_numpy(X_NP)
        assert isinstance(out, np.ndarray)
        np.testing.assert_array_equal(out, X_NP)

    def test_match_backend_numpy(self):
        ref = np.zeros(4, dtype=np.float32)
        param = np.array([1.0, 2.0, 3.0, 4.0])
        out = match_backend(param, ref)
        assert isinstance(out, np.ndarray)


# ── PyTorch ───────────────────────────────────────────────────────────────────

class TestBackendsTorch:

    def setup_method(self):
        self.torch = pytest.importorskip("torch")

    def _x(self):
        return self.torch.from_numpy(X_NP.copy())

    def test_fft_roundtrip(self):
        x = self._x()
        out = ifft(fft(x))
        np.testing.assert_allclose(to_numpy(out).real, X_NP.real, atol=1e-5)

    def test_fftshift_roundtrip(self):
        x = self._x()
        np.testing.assert_allclose(to_numpy(ifftshift(fftshift(x))), X_NP, atol=1e-6)

    def test_to_numpy(self):
        x = self._x()
        out = to_numpy(x)
        assert isinstance(out, np.ndarray)
        np.testing.assert_array_almost_equal(out, X_NP)

    def test_match_backend(self):
        ref = self.torch.zeros(4)
        param = np.array([1.0, 2.0, 3.0, 4.0])
        out = match_backend(param, ref)
        assert isinstance(out, self.torch.Tensor)


# ── JAX ───────────────────────────────────────────────────────────────────────

class TestBackendsJax:

    def setup_method(self):
        self.jnp = pytest.importorskip("jax.numpy")

    def _x(self):
        return self.jnp.array(X_NP)

    def test_fft_roundtrip(self):
        x = self._x()
        out = ifft(fft(x))
        np.testing.assert_allclose(np.asarray(out).real, X_NP.real, atol=1e-5)

    def test_fftshift_roundtrip(self):
        x = self._x()
        np.testing.assert_allclose(np.asarray(ifftshift(fftshift(x))), X_NP, atol=1e-6)

    def test_to_numpy(self):
        x = self._x()
        out = to_numpy(x)
        assert isinstance(out, np.ndarray)

    def test_match_backend(self):
        ref = self.jnp.zeros(4)
        param = np.array([1.0, 2.0, 3.0, 4.0])
        out = match_backend(param, ref)
        assert hasattr(out, 'device')   # jax array


# ── TensorFlow ────────────────────────────────────────────────────────────────

class TestBackendsTensorflow:

    def setup_method(self):
        self.tf = pytest.importorskip("tensorflow")

    def _x(self):
        return self.tf.constant(X_NP)

    def test_fft_roundtrip(self):
        x = self._x()
        out = ifft(fft(x))
        np.testing.assert_allclose(to_numpy(out).real, X_NP.real, atol=1e-5)

    def test_fftshift_roundtrip(self):
        x = self._x()
        np.testing.assert_allclose(to_numpy(ifftshift(fftshift(x))), X_NP, atol=1e-6)

    def test_to_numpy(self):
        x = self._x()
        out = to_numpy(x)
        assert isinstance(out, np.ndarray)

    def test_match_backend(self):
        ref = self.tf.zeros(4)
        param = np.array([1.0, 2.0, 3.0, 4.0])
        out = match_backend(param, ref)
        assert isinstance(out, self.tf.Tensor)

    @pytest.mark.parametrize("fn", [fft, ifft, fftshift, ifftshift])
    def test_stays_a_tf_tensor(self, fn):
        """TF input must not silently fall through to the numpy branch.

        Before ops.py grew TensorFlow branches, every one of these dropped a TF
        tensor into "np.fft.*", which coerces via "__array__" and discards
        the graph. The roundtrip tests could not see it because they compare
        through "to_numpy".
        """
        assert isinstance(fn(self._x()), self.tf.Tensor)

    @pytest.mark.parametrize("n", [8, 9])
    def test_shift_matches_numpy_for_odd_and_even(self, n):
        """fftshift and ifftshift differ only for odd n -- pin both."""
        x = _rng.random(n) + 1j * _rng.random(n)
        xt = self.tf.constant(x.astype(np.complex64))
        np.testing.assert_array_equal(to_numpy(fftshift(xt)), np.fft.fftshift(x).astype(np.complex64))
        np.testing.assert_array_equal(to_numpy(ifftshift(xt)), np.fft.ifftshift(x).astype(np.complex64))
