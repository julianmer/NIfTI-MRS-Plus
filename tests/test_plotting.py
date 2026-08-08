####################################################################################################
#                                       test_plotting.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-06-27                                                                              #
#                                                                                                  #
# Purpose: Tests for batch-aware plotting utilities. Mirrors the nifti-mrs test_nifti_mrs_vis      #
#          pattern: skip when fsl-mrs is absent, run when present.                                 #
#                                                                                                  #
####################################################################################################

import pytest
import matplotlib
matplotlib.use('Agg')   # non-interactive backend for CI

try:
    import fsl_mrs   # noqa: F401
    FSL_MRS_AVAILABLE = True
except ImportError:
    FSL_MRS_AVAILABLE = False


def test_plotting_import_fails_without_fsl(monkeypatch):
    """Importing plotting should raise ImportError when fsl-mrs is absent."""
    import sys
    import importlib

    if FSL_MRS_AVAILABLE:
        pytest.skip("fsl-mrs present")

    # Ensure plotting is not cached
    for mod in list(sys.modules.keys()):
        if 'nifti_mrs_plus.plotting' in mod:
            del sys.modules[mod]

    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name.startswith('fsl_mrs'):
            raise ImportError("mocked fsl_mrs missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', mock_import)
    with pytest.raises(ImportError, match="FSL-MRS"):
        importlib.reload(importlib.import_module('nifti_mrs_plus.plotting'))


@pytest.mark.skipif(not FSL_MRS_AVAILABLE, reason="fsl-mrs not installed")
class TestPlotting:

    def test_vis_returns_figure(self, nifti_mrs_plus):
        import matplotlib.figure
        from nifti_mrs_plus.plotting import vis_nifti_mrs_plus
        fig = vis_nifti_mrs_plus(nifti_mrs_plus, batch_index=0, ppmlim=(0.2, 4.2))
        assert isinstance(fig, matplotlib.figure.Figure)
        matplotlib.pyplot.close('all')

    def test_vis_batch_returns_figure(self, nifti_mrs_plus):
        import matplotlib.figure
        from nifti_mrs_plus.plotting import vis_nifti_mrs_plus
        fig = vis_nifti_mrs_plus(nifti_mrs_plus, ppmlim=(0.2, 4.2), max_batch_display=3)
        assert isinstance(fig, matplotlib.figure.Figure)
        matplotlib.pyplot.close('all')

    def test_plot_batch_comparison(self, nifti_mrs_plus):
        import matplotlib.figure
        import matplotlib.pyplot
        from nifti_mrs_plus.plotting import plot_batch_comparison
        fig = plot_batch_comparison(nifti_mrs_plus, indices=[0, 1, 2], ppmlim=(0.2, 4.2))
        assert isinstance(fig, matplotlib.figure.Figure)
        matplotlib.pyplot.close('all')

    def test_plot_batch_grid_detailed(self, nifti_mrs_plus):
        import matplotlib.figure
        import matplotlib.pyplot
        from nifti_mrs_plus.plotting import plot_batch_grid_detailed
        fig = plot_batch_grid_detailed(nifti_mrs_plus, max_display=4, ppmlim=(0.2, 4.2))
        assert isinstance(fig, matplotlib.figure.Figure)
        matplotlib.pyplot.close('all')

    def test_quick_plot(self, nifti_mrs_plus):
        import matplotlib.figure
        import matplotlib.pyplot
        from nifti_mrs_plus.plotting import quick_plot
        fig = quick_plot(nifti_mrs_plus, index=0, ppmlim=(0.2, 4.2))
        assert isinstance(fig, matplotlib.figure.Figure)
        matplotlib.pyplot.close('all')

    def test_vis_bad_index_raises(self, nifti_mrs_plus):
        from nifti_mrs_plus.plotting import vis_nifti_mrs_plus
        with pytest.raises(IndexError):
            vis_nifti_mrs_plus(nifti_mrs_plus, batch_index=999)

    def test_vis_wrong_type_raises(self):
        from nifti_mrs_plus.plotting import vis_nifti_mrs_plus
        with pytest.raises(TypeError):
            vis_nifti_mrs_plus("not a NIfTI_MRS_Plus object")
