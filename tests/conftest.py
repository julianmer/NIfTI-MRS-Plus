####################################################################################################
#                                          conftest.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-06-27                                                                              #
#                                                                                                  #
# Purpose: Shared pytest fixtures for the nifti-mrs-plus test suite. Mirrors the fixture          #
#          structure used in Augmentrum, adapted to import from nifti_mrs_plus directly.          #
#                                                                                                  #
####################################################################################################

import pytest
import numpy as np

from nifti_mrs.create_nmrs import gen_nifti_mrs
from nifti_mrs_plus import NIfTI_MRS_Plus, Backend


@pytest.fixture
def dummy_nifti_mrs():
    """Single NIfTI-MRS object — shape (1,1,1,2048,8,16), DIM_COIL + DIM_DYN."""
    data = np.random.randn(1, 1, 1, 2048, 8, 16) + 1j * np.random.randn(1, 1, 1, 2048, 8, 16)
    nifti = gen_nifti_mrs(data, dwelltime=1/2000, spec_freq=123.0)
    nifti.set_dim_tag(4, 'DIM_COIL')
    nifti.set_dim_tag(5, 'DIM_DYN')
    return nifti


@pytest.fixture
def dummy_nifti_list():
    """List of 5 NIfTI-MRS objects — same shape and dim tags."""
    nifti_list = []
    for _ in range(5):
        data = np.random.randn(1, 1, 1, 2048, 8, 16) + 1j * np.random.randn(1, 1, 1, 2048, 8, 16)
        nifti = gen_nifti_mrs(data, dwelltime=1/2000, spec_freq=123.0)
        nifti.set_dim_tag(4, 'DIM_COIL')
        nifti.set_dim_tag(5, 'DIM_DYN')
        nifti_list.append(nifti)
    return nifti_list


@pytest.fixture
def nifti_mrs_plus(dummy_nifti_list):
    """NIfTI_MRS_Plus wrapping the dummy list, NIFTI_LIST backend."""
    return NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)


@pytest.fixture
def dummy_nifti_single_coil():
    """Single-voxel, single-coil, no dynamics — minimal shape (1,1,1,2048)."""
    data = np.random.randn(1, 1, 1, 2048) + 1j * np.random.randn(1, 1, 1, 2048)
    return gen_nifti_mrs(data, dwelltime=1/2000, spec_freq=123.0)


@pytest.fixture
def dummy_nifti_water():
    """Water-reference NIfTI-MRS — same multi-coil shape as dummy_nifti_mrs."""
    data = np.random.randn(1, 1, 1, 2048, 8, 16) + 1j * np.random.randn(1, 1, 1, 2048, 8, 16)
    nifti = gen_nifti_mrs(data, dwelltime=1/2000, spec_freq=123.0)
    nifti.set_dim_tag(4, 'DIM_COIL')
    nifti.set_dim_tag(5, 'DIM_DYN')
    return nifti
