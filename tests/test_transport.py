####################################################################################################
#                                       test_transport.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-08-07                                                                              #
#                                                                                                  #
# Purpose: Tests for the gradient-carrying transport path: set_data / materialize, and the         #
#          guarantee that a tensor handed to the container comes back with its autograd graph      #
#          and device placement intact.                                                            #
#                                                                                                  #
####################################################################################################

import pytest
import numpy as np

from nifti_mrs_plus import NIfTI_MRS_Plus, Backend


#**********************************#
#   Class TestSetDataMaterialize   #
#**********************************#
class TestSetDataMaterialize:
    """set_data defers the NumPy round-trip; materialize performs it."""

    def test_set_data_does_not_touch_nifti_list(self, dummy_nifti_list):
        plus = NIfTI_MRS_Plus(dummy_nifti_list, backend=Backend.NUMPY, volatile=True)
        before = plus.nifti_list[0][:].copy()

        plus.set_data(plus.numpy() * 3.0)

        # nifti_list is deliberately stale until materialize()
        assert np.allclose(plus.nifti_list[0][:], before)

    def test_materialize_flushes(self, dummy_nifti_list):
        plus = NIfTI_MRS_Plus(dummy_nifti_list, backend=Backend.NUMPY, volatile=True)
        before = plus.nifti_list[0][:].copy()

        plus.set_data(plus.numpy() * 3.0)
        plus.materialize()

        assert np.allclose(plus.nifti_list[0][:], before * 3.0)

    def test_list_triggers_materialize(self, dummy_nifti_list):
        plus = NIfTI_MRS_Plus(dummy_nifti_list, backend=Backend.NUMPY, volatile=True)
        before = plus.nifti_list[0][:].copy()

        plus.set_data(plus.numpy() * 2.0)

        assert np.allclose(plus.list()[0][:], before * 2.0)

    def test_numpy_reads_pending_tensor_not_stale_list(self, dummy_nifti_list):
        """numpy() must not restack the stale list while a tensor is pending."""
        plus = NIfTI_MRS_Plus(dummy_nifti_list, backend=Backend.NUMPY, volatile=True)
        before = plus.numpy().copy()

        plus.set_data(before * 5.0)

        assert np.allclose(plus.numpy(), before * 5.0)

    def test_materialize_is_idempotent(self, dummy_nifti_list):
        plus = NIfTI_MRS_Plus(dummy_nifti_list, backend=Backend.NUMPY, volatile=True)
        before = plus.numpy().copy()

        plus.set_data(before * 2.0)
        plus.materialize()
        plus.materialize()          # second call must not double-apply

        assert np.allclose(plus.numpy(), before * 2.0)

    def test_wrong_batch_size_rejected(self, dummy_nifti_list):
        plus = NIfTI_MRS_Plus(dummy_nifti_list, backend=Backend.NUMPY, volatile=True)
        with pytest.raises(ValueError, match="leading axis"):
            plus.set_data(plus.numpy()[:2])

    def test_unknown_type_rejected(self, dummy_nifti_list):
        plus = NIfTI_MRS_Plus(dummy_nifti_list, backend=Backend.NUMPY, volatile=True)
        with pytest.raises(TypeError, match="cannot infer a backend"):
            plus.set_data(object())


#*****************************#
#   Class TestGradientFlow    #
#*****************************#
class TestGradientFlow:
    """The reason set_data exists: an autograd graph must survive the container."""

    def test_torch_graph_survives(self, dummy_nifti_list):
        torch = pytest.importorskip("torch")

        plus = NIfTI_MRS_Plus(dummy_nifti_list, backend=Backend.PYTORCH, volatile=True)
        data = plus.get_data(Backend.PYTORCH)

        scale = torch.tensor(2.0, requires_grad=True)
        out = data * torch.complex(scale, torch.zeros_like(scale))
        plus.set_data(out, Backend.PYTORCH)

        back = plus.get_data(Backend.PYTORCH)
        assert back.grad_fn is not None, "autograd graph was severed by the container"

        torch.abs(back).sum().backward()
        assert scale.grad is not None and scale.grad.item() != 0.0

    def test_setitem_full_slice_preserves_graph(self, dummy_nifti_list):
        """`plus[:] = tensor` is the hot path and must not detach."""
        torch = pytest.importorskip("torch")

        plus = NIfTI_MRS_Plus(dummy_nifti_list, backend=Backend.PYTORCH, volatile=True)
        scale = torch.tensor(2.0, requires_grad=True)
        out = plus.get_data(Backend.PYTORCH) * torch.complex(scale, torch.zeros_like(scale))

        plus[:] = out               # must not raise, and must not detach

        assert plus.get_data(Backend.PYTORCH).grad_fn is not None

    def test_jax_tracer_survives(self, dummy_nifti_list):
        jax = pytest.importorskip("jax")

        plus = NIfTI_MRS_Plus(dummy_nifti_list, backend=Backend.JAX, volatile=True)

        def loss(scale):
            data = plus.get_data(Backend.JAX)
            fresh = NIfTI_MRS_Plus(plus.nifti_list, backend=Backend.JAX, volatile=True)
            fresh.set_data(data * scale, Backend.JAX)
            return jax.numpy.abs(fresh.get_data(Backend.JAX)).sum()

        grad = jax.grad(loss)(2.0)
        assert np.isfinite(float(grad)) and float(grad) != 0.0

    def test_get_data_torch_does_not_alias_cache(self, dummy_nifti_list):
        """torch.from_numpy would share memory with the cached NumPy array."""
        torch = pytest.importorskip("torch")

        plus = NIfTI_MRS_Plus(dummy_nifti_list, backend=Backend.PYTORCH, volatile=True)
        cached_np = plus.numpy()
        idx = (0,) * cached_np.ndim          # fixture is 7-D: (5,1,1,1,2048,8,16)
        before = cached_np[idx].copy()

        tensor = plus.get_data(Backend.PYTORCH)
        tensor[idx] = 12345.0 + 0j

        assert cached_np[idx] == before, "in-place torch op mutated the cache"
