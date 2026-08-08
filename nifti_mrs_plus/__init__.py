####################################################################################################
#                                          __init__.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-06-27                                                                              #
#                                                                                                  #
# Purpose: Public API for the nifti-mrs-plus package.                                              #
#          Exposes the NIfTI_MRS_Plus batched wrapper and the Backend enum.                        #
#                                                                                                  #
####################################################################################################

# Defined before importing submodules so core.py can read it without a circular import.
__version__ = "0.1.0"

from nifti_mrs_plus.core import NIfTI_MRS_Plus, Backend, set_provenance, get_provenance

__all__ = ["NIfTI_MRS_Plus", "Backend", "set_provenance", "get_provenance", "__version__"]
