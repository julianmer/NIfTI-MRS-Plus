####################################################################################################
#                                        test_core.py                                              #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-06-27                                                                              #
#                                                                                                  #
# Purpose: Tests for NIfTI_MRS_Plus core functionality: creation, properties, backends,           #
#          indexing, metadata, copy, and repr. Ported from the Augmentrum test suite.             #
#                                                                                                  #
####################################################################################################

import pytest
import numpy as np

from nifti_mrs_plus import NIfTI_MRS_Plus, Backend


class TestCreation:

    def test_create_from_list(self, dummy_nifti_list):
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list)
        assert nifti_plus is not None
        assert len(nifti_plus) == 5
        assert nifti_plus.n_subjects == 5

    def test_create_from_another_nifti_mrs_plus(self, nifti_mrs_plus):
        copy = NIfTI_MRS_Plus(nifti_list=nifti_mrs_plus)
        assert len(copy) == len(nifti_mrs_plus)
        assert copy.backend == nifti_mrs_plus.backend

    def test_create_with_backend(self, dummy_nifti_list):
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NUMPY)
        assert nifti_plus.backend == Backend.NUMPY

    def test_create_volatile(self, dummy_nifti_list):
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, volatile=True)
        assert nifti_plus.volatile is True
        assert nifti_plus.metadata_common == {}
        assert nifti_plus.metadata_individual == []

    def test_create_empty_list(self):
        nifti_plus = NIfTI_MRS_Plus(nifti_list=[])
        assert len(nifti_plus) == 0
        assert nifti_plus.n_subjects == 0

    def test_reject_invalid_input(self):
        with pytest.raises(ValueError, match="must be a list of NIFTI_MRS"):
            NIfTI_MRS_Plus(nifti_list="not a list")
        with pytest.raises(ValueError, match="must be a list of NIFTI_MRS"):
            NIfTI_MRS_Plus(nifti_list=[1, 2, 3])


class TestShape:

    def test_shape(self, nifti_mrs_plus):
        shape = nifti_mrs_plus.shape
        assert shape[0] == 5
        assert shape[1:] == (1, 1, 1, 2048, 8, 16)

    def test_dim_tags(self, nifti_mrs_plus):
        assert 'DIM_COIL' in nifti_mrs_plus.dim_tags
        assert 'DIM_DYN' in nifti_mrs_plus.dim_tags

    def test_dim_tags_non_uniform_warns(self, dummy_nifti_list):
        dummy_nifti_list[2].set_dim_tag(4, 'DIM_EDIT')
        with pytest.warns(UserWarning, match="non-uniform dim_tags"):
            nplus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list)
        assert nplus.n_subjects == len(dummy_nifti_list)

    def test_len(self, nifti_mrs_plus):
        assert len(nifti_mrs_plus) == 5

    def test_shape_empty(self):
        nifti_plus = NIfTI_MRS_Plus(nifti_list=[])
        assert nifti_plus.shape == (0,)
        assert nifti_plus.dim_tags is None


class TestProperties:

    def test_dim_position(self, nifti_mrs_plus):
        assert nifti_mrs_plus.dim_position('DIM_COIL') == 4
        assert nifti_mrs_plus.dim_position('DIM_DYN') == 5

    def test_dwelltime(self, nifti_mrs_plus):
        assert nifti_mrs_plus.dwelltime == pytest.approx(1/2000)

    def test_spectrometer_frequency(self, nifti_mrs_plus):
        assert nifti_mrs_plus.spectrometer_frequency == [123.0]

    def test_nucleus(self, nifti_mrs_plus):
        assert nifti_mrs_plus.nucleus is not None

    def test_ndim(self, nifti_mrs_plus):
        assert nifti_mrs_plus.ndim >= 6

    def test_dtype(self, nifti_mrs_plus):
        assert nifti_mrs_plus.dtype in (np.complex64, np.complex128)

    def test_header(self, nifti_mrs_plus):
        assert nifti_mrs_plus.header is not None

    def test_hdr_ext(self, nifti_mrs_plus):
        assert nifti_mrs_plus.hdr_ext is not None

    def test_proxy_methods_empty(self):
        nifti_plus = NIfTI_MRS_Plus(nifti_list=[])
        assert nifti_plus.dim_position('DIM_COIL') is None
        assert nifti_plus.dwelltime is None
        assert nifti_plus.ndim is None
        assert nifti_plus.dtype is None


class TestBackends:

    def test_default_backend(self, dummy_nifti_list):
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list)
        assert nifti_plus.backend == Backend.NIFTI_LIST

    def test_get_data_nifti_list(self, nifti_mrs_plus):
        data = nifti_mrs_plus.get_data(Backend.NIFTI_LIST)
        assert isinstance(data, list)
        assert len(data) == 5
        assert hasattr(data[0], 'dwelltime')

    def test_get_data_numpy(self, nifti_mrs_plus):
        data = nifti_mrs_plus.get_data(Backend.NUMPY)
        assert isinstance(data, np.ndarray)
        assert data.shape == (5, 1, 1, 1, 2048, 8, 16)
        assert np.iscomplexobj(data)

    def test_numpy_caching(self, nifti_mrs_plus):
        a = nifti_mrs_plus.numpy()
        b = nifti_mrs_plus.numpy()
        assert a is b

    def test_list_method(self, nifti_mrs_plus):
        nifti_list = nifti_mrs_plus.list()
        assert isinstance(nifti_list, list)
        assert len(nifti_list) == 5

    def test_to_nifti_list(self, nifti_mrs_plus):
        nifti_list = nifti_mrs_plus.to_nifti_list()
        assert isinstance(nifti_list, list)
        assert len(nifti_list) == 5

    def test_data_property(self, nifti_mrs_plus):
        data = nifti_mrs_plus.data
        assert isinstance(data, list)

    def test_get_data_empty_nifti_list(self):
        nifti_plus = NIfTI_MRS_Plus(nifti_list=[])
        assert nifti_plus.get_data(Backend.NIFTI_LIST) == []
        arr = nifti_plus.get_data(Backend.NUMPY)
        assert isinstance(arr, np.ndarray)
        assert len(arr) == 0

    def test_get_data_pytorch(self, nifti_mrs_plus):
        torch = pytest.importorskip("torch")
        data = nifti_mrs_plus.get_data(Backend.PYTORCH)
        assert isinstance(data, torch.Tensor)
        assert data.shape == torch.Size([5, 1, 1, 1, 2048, 8, 16])
        assert torch.is_complex(data)

    def test_get_data_jax(self, nifti_mrs_plus):
        jnp = pytest.importorskip("jax.numpy")
        data = nifti_mrs_plus.get_data(Backend.JAX)
        assert data.shape == (5, 1, 1, 1, 2048, 8, 16)
        assert jnp.iscomplexobj(data)

    def test_get_data_tensorflow(self, nifti_mrs_plus):
        tf = pytest.importorskip("tensorflow")
        data = nifti_mrs_plus.get_data(Backend.TENSORFLOW)
        assert tuple(data.shape) == (5, 1, 1, 1, 2048, 8, 16)
        assert data.dtype in (tf.complex64, tf.complex128)

    def test_get_data_keras(self, nifti_mrs_plus):
        pytest.importorskip("keras")
        data = nifti_mrs_plus.get_data(Backend.KERAS)
        assert tuple(data.shape) == (5, 1, 1, 1, 2048, 8, 16)

    def test_all_backends_same_values(self, dummy_nifti_list):
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list)
        ref = nifti_plus.get_data(Backend.NUMPY)

        try:
            import torch
            np.testing.assert_array_almost_equal(ref, nifti_plus.get_data(Backend.PYTORCH).numpy())
        except ImportError:
            pass

        try:
            import tensorflow as tf
            np.testing.assert_array_almost_equal(ref, nifti_plus.get_data(Backend.TENSORFLOW).numpy())
        except ImportError:
            pass

        try:
            import jax.numpy as jnp
            np.testing.assert_array_almost_equal(ref, np.asarray(nifti_plus.get_data(Backend.JAX)))
        except ImportError:
            pass

    def test_complex_dtype_preserved(self, dummy_nifti_list):
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list)
        assert np.iscomplexobj(nifti_plus.get_data(Backend.NUMPY))

        try:
            import torch
            assert torch.is_complex(nifti_plus.get_data(Backend.PYTORCH))
        except ImportError:
            pass

    def test_backend_setting_preserved(self, dummy_nifti_list):
        for backend in [Backend.NUMPY]:
            nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=backend)
            assert nifti_plus.backend == backend


class TestIndexing:

    def test_getitem_single(self, nifti_mrs_plus):
        item = nifti_mrs_plus[0]
        assert hasattr(item, 'dwelltime')

    def test_getitem_slice(self, nifti_mrs_plus):
        subset = nifti_mrs_plus[1:3]
        assert isinstance(subset, NIfTI_MRS_Plus)
        assert len(subset) == 2

    def test_getitem_preserves_backend(self, dummy_nifti_list):
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NUMPY)
        assert nifti_plus[0:2].backend == Backend.NUMPY

    def test_getitem_preserves_volatile(self, dummy_nifti_list):
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, volatile=True)
        assert nifti_plus[0:2].volatile is True


class TestMetadata:

    def test_metadata_initialized(self, nifti_mrs_plus):
        assert isinstance(nifti_mrs_plus.metadata_common, dict)
        assert isinstance(nifti_mrs_plus.metadata_individual, list)
        assert len(nifti_mrs_plus.metadata_individual) == 5

    def test_metadata_empty_volatile(self, dummy_nifti_list):
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, volatile=True)
        assert nifti_plus.metadata_common == {}
        assert nifti_plus.metadata_individual == []

    def test_update_metadata(self, nifti_mrs_plus):
        if not nifti_mrs_plus.volatile:
            nifti_mrs_plus.update_metadata('TestOperation', {'param': 'value'})
            assert 'common_provenance' in nifti_mrs_plus.metadata_common

    def test_update_metadata_noop_volatile(self, dummy_nifti_list):
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, volatile=True)
        nifti_plus.update_metadata('TestOperation', {'param': 'value'})
        assert nifti_plus.metadata_common == {}


class TestCopy:

    def test_copy(self, nifti_mrs_plus):
        copy = nifti_mrs_plus.copy()
        assert copy is not nifti_mrs_plus
        assert len(copy) == len(nifti_mrs_plus)
        assert copy.backend == nifti_mrs_plus.backend
        assert copy.volatile == nifti_mrs_plus.volatile

    def test_copy_independent(self, nifti_mrs_plus):
        copy = nifti_mrs_plus.copy()
        if not copy.volatile:
            copy.update_metadata('TestOp', {})
            assert len(copy.metadata_common.get('common_provenance', [])) != \
                   len(nifti_mrs_plus.metadata_common.get('common_provenance', []))


class TestRepr:

    def test_repr(self, nifti_mrs_plus):
        r = repr(nifti_mrs_plus)
        assert 'NIfTI_MRS_Plus' in r
        assert 'n_subjects=5' in r
        assert 'backend=' in r
        assert 'volatile=' in r


class TestSyncHeaders:

    def test_sync_headers(self, nifti_mrs_plus):
        nifti_mrs_plus.sync_headers(source_idx=0)
