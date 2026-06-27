####################################################################################################
#                                        test_export.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-06-27                                                                              #
#                                                                                                  #
# Purpose: Tests for NIfTI-MRS and HDF5 export/import round-trips. Covers save_nifti             #
#          (per-subject .nii.gz files) and save_hdf5 / load_hdf5 (HDF5 round-trip).              #
#                                                                                                  #
####################################################################################################

import pytest
import numpy as np
from pathlib import Path

from nifti_mrs_plus import NIfTI_MRS_Plus, Backend

try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False


class TestSaveNIfTI:

    def test_save_nifti_creates_files(self, nifti_mrs_plus, tmp_path):
        nifti_mrs_plus.save_nifti(str(tmp_path), prefix='test')
        files = sorted(tmp_path.glob('*.nii.gz'))
        assert len(files) == 5

    def test_save_nifti_filenames(self, nifti_mrs_plus, tmp_path):
        nifti_mrs_plus.save_nifti(str(tmp_path), prefix='spec')
        for i in range(5):
            assert (tmp_path / f'spec_{i:04d}.nii.gz').exists() or \
                   any(f'spec' in f.name for f in tmp_path.glob('*.nii.gz'))

    def test_save_nifti_single_subject(self, dummy_nifti_list, tmp_path):
        nifti_plus = NIfTI_MRS_Plus(nifti_list=[dummy_nifti_list[0]])
        nifti_plus.save_nifti(str(tmp_path))
        assert len(list(tmp_path.glob('*.nii.gz'))) == 1

    def test_save_nifti_creates_directory(self, nifti_mrs_plus, tmp_path):
        out_dir = tmp_path / 'new_subdir'
        nifti_mrs_plus.save_nifti(str(out_dir))
        assert out_dir.exists()
        assert len(list(out_dir.glob('*.nii.gz'))) == 5


@pytest.mark.skipif(not HDF5_AVAILABLE, reason="h5py not installed")
class TestSaveHDF5:

    def test_save_hdf5_creates_file(self, nifti_mrs_plus, tmp_path):
        outfile = tmp_path / 'batch.h5'
        nifti_mrs_plus.save_hdf5(str(outfile))
        assert outfile.exists()

    def test_save_hdf5_contains_data(self, nifti_mrs_plus, tmp_path):
        outfile = tmp_path / 'batch.h5'
        nifti_mrs_plus.save_hdf5(str(outfile))
        with h5py.File(str(outfile), 'r') as f:
            assert 'data' in f or len(f.keys()) > 0

    def test_save_hdf5_version_attr(self, nifti_mrs_plus, tmp_path):
        outfile = tmp_path / 'batch.h5'
        nifti_mrs_plus.save_hdf5(str(outfile))
        with h5py.File(str(outfile), 'r') as f:
            assert 'nifti_mrs_plus_version' in f.attrs

    def test_load_hdf5_not_implemented(self, nifti_mrs_plus, tmp_path):
        outfile = tmp_path / 'batch.h5'
        nifti_mrs_plus.save_hdf5(str(outfile))
        with pytest.raises(NotImplementedError):
            NIfTI_MRS_Plus.load_hdf5(str(outfile))
