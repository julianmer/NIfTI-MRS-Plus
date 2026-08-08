####################################################################################################
#                                         test_state.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-08-08                                                                              #
#                                                                                                  #
# Purpose: Holds DataState to the one property that makes it worth having: it is carried even on   #
#          the volatile path, where the provenance record deliberately is not.                     #
#                                                                                                  #
####################################################################################################

"""
Tests for the small always-carried record of where the data is.

The distinction being guarded is between this and provenance. Provenance grows
with every operation, so volatile mode skips it for speed. State is a fixed
handful of facts replaced wholesale, so it costs the same however long the
pipeline is and must survive that same fast path - which is exactly where a
module needs it.
"""

#*************#
#   imports   #
#*************#
import numpy as np
import pytest

from fsl_mrs.core.nifti_mrs import gen_nifti_mrs
from nifti_mrs_plus import NIfTI_MRS_Plus, Backend
from nifti_mrs_plus.core import DataState


#**************#
#   fixtures   #
#**************#
@pytest.fixture
def batch():
    """A small batch, in the NIfTI-MRS canonical form."""
    objs = [gen_nifti_mrs(np.ones((1, 1, 1, 64), np.complex64), 1 / 2000.0, 123.0)
            for _ in range(3)]
    return lambda volatile=False: NIfTI_MRS_Plus(nifti_list=objs, backend=Backend.NUMPY,
                                                 volatile=volatile)


#***********#
#   state   #
#***********#
def test_a_fresh_batch_starts_canonical(batch):
    """Time domain, image space, fully sampled - what a NIfTI-MRS file holds."""
    state = batch().state

    assert (state.spectral, state.spatial, state.sampling) == ('time', 'image', 'full')
    assert state.last == ''


def test_state_cannot_be_mutated_by_accident(batch):
    """Frozen, so a module cannot change what its caller is holding."""
    with pytest.raises(Exception):
        batch().state.spatial = 'kspace'


def test_having_leaves_the_original_alone(batch):
    """Updating is making a new one, which is what makes it safe to pass around."""
    before = batch().state
    after = before.having(spatial='kspace', last='Something')

    assert before.spatial == 'image', "the original was modified"
    assert (after.spatial, after.last) == ('kspace', 'Something')
    assert after.spectral == before.spectral, "unrelated fields should carry over"


#*******************#
#   carried where   #
#*******************#
@pytest.mark.parametrize("volatile", [True, False])
def test_state_survives_the_volatile_path(batch, volatile):
    """The point of the exercise: volatile drops provenance, never state."""
    plus = batch(volatile=volatile)
    plus.set_state(plus.state.having(spatial='kspace', sampling='undersampled'))

    assert plus.state.spatial == 'kspace'
    assert plus.state.sampling == 'undersampled'


def test_state_follows_the_data_through_copies_and_slices(batch):
    """It describes the data, so anything derived from it carries it."""
    plus = batch()
    plus.set_state(plus.state.having(spectral='frequency', last='DomainTransform'))

    assert plus.copy().state.spectral == 'frequency'
    assert plus[0:2].state.last == 'DomainTransform'


def test_the_two_domain_axes_are_independent(batch):
    """
    Spectral and spatial are separate because their transforms are.

    They act on different axes of the same array and commute, so the data can
    sit in any combination of the two. One flag could not express that, and
    would force transforms that are not needed.
    """
    plus = batch()
    plus.set_state(plus.state.having(spectral='frequency', spatial='kspace'))

    assert (plus.state.spectral, plus.state.spatial) == ('frequency', 'kspace')
